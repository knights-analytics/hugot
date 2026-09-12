package pipelines

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"sync/atomic"
	"time"

	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/util/safeconv"
	"github.com/knights-analytics/hugot/util/vectorutil"
)

// ZeroShotAudioClassificationConfig configures a zero-shot audio pipeline.
type (
	ZeroShotAudioClassificationConfig = backends.PipelineConfig[*ZeroShotAudioClassificationPipeline]
	ZeroShotAudioClassificationOption = backends.PipelineOption[*ZeroShotAudioClassificationPipeline]
)

// ZeroShotAudioClassificationResult is the score for one candidate label.
type ZeroShotAudioClassificationResult struct {
	Label      string
	Score      float32
	ClassIndex int
}

type ZeroShotAudioClassificationOutput struct {
	Predictions [][]ZeroShotAudioClassificationResult
}

func (o *ZeroShotAudioClassificationOutput) GetOutput() []any {
	out := make([]any, len(o.Predictions))
	for i := range o.Predictions {
		out[i] = o.Predictions[i]
	}
	return out
}

// ZeroShotAudioClassificationPipeline scores candidate labels against audio.
// The model contract is an audio input and one logit per candidate label.
type ZeroShotAudioClassificationPipeline struct {
	*backends.BasePipeline
	Labels []string
	TopK   int
	Input  backends.InputOutputInfo
}

func WithAudioLabels(labels []string) ZeroShotAudioClassificationOption {
	return func(p *ZeroShotAudioClassificationPipeline) error {
		p.Labels = append([]string(nil), labels...)
		return nil
	}
}

// WithZeroShotAudioLabels is the family-specific spelling of WithAudioLabels.
func WithZeroShotAudioLabels(labels []string) ZeroShotAudioClassificationOption {
	return WithAudioLabels(labels)
}

func WithZeroShotAudioTopK(k int) ZeroShotAudioClassificationOption {
	return func(p *ZeroShotAudioClassificationPipeline) error {
		if k < 1 {
			return errors.New("zero-shot audio classification top-k must be greater than zero")
		}
		p.TopK = k
		return nil
	}
}

func NewZeroShotAudioClassificationPipeline(ctx context.Context, config ZeroShotAudioClassificationConfig, model *backends.Model) (*ZeroShotAudioClassificationPipeline, error) {
	if model == nil {
		return nil, errors.New("zero-shot audio classification requires a model")
	}
	p := &ZeroShotAudioClassificationPipeline{BasePipeline: backends.NewBasePipeline(ctx, config, model), TopK: 5}
	for _, option := range config.Options {
		if err := option(p); err != nil {
			return nil, err
		}
	}
	if err := p.Validate(); err != nil {
		return nil, err
	}
	return p, nil
}

func (p *ZeroShotAudioClassificationPipeline) IsGenerative() bool        { return false }
func (p *ZeroShotAudioClassificationPipeline) GetModel() *backends.Model { return p.Model }
func (p *ZeroShotAudioClassificationPipeline) GetMetadata() backends.PipelineMetadata {
	return backends.PipelineMetadata{OutputsInfo: []backends.OutputInfo{{Name: p.Model.OutputsMeta[0].Name, Dimensions: p.Model.OutputsMeta[0].Dimensions}}}
}

func (p *ZeroShotAudioClassificationPipeline) GetStatistics() backends.PipelineStatistics {
	s := backends.PipelineStatistics{}
	s.ComputeOnnxStatistics(p.ONNXTimings)
	return s
}

func (p *ZeroShotAudioClassificationPipeline) Validate() error {
	if p.Model == nil {
		return errors.New("zero-shot audio classification requires a model")
	}
	var errs []error
	if len(p.Labels) == 0 {
		errs = append(errs, errors.New("zero-shot audio classification requires at least one candidate label"))
	}
	if len(p.Model.InputsMeta) == 0 {
		errs = append(errs, errors.New("zero-shot audio classification model has no inputs"))
	} else {
		p.Input = p.Model.InputsMeta[0]
		for _, input := range p.Model.InputsMeta {
			if input.Name == "input_values" || input.Name == "waveform" || input.Name == "audio" {
				p.Input = input
				break
			}
		}
		if len(p.Input.Dimensions) != 2 && len(p.Input.Dimensions) != 3 {
			errs = append(errs, fmt.Errorf("unsupported zero-shot audio input layout %q: expected rank 2 or 3, got %v", p.Input.Name, p.Input.Dimensions))
		}
	}
	if len(p.Model.OutputsMeta) == 0 {
		errs = append(errs, errors.New("zero-shot audio classification model has no outputs"))
	} else {
		dims := p.Model.OutputsMeta[0].Dimensions
		if len(dims) != 2 {
			errs = append(errs, fmt.Errorf("zero-shot audio classification output must have shape [batch, labels], got %v", dims))
		} else if dims[1] > 0 && len(p.Labels) > 0 && int(dims[1]) != len(p.Labels) {
			errs = append(errs, fmt.Errorf("candidate labels (%d) do not match audio logits (%d)", len(p.Labels), dims[1]))
		}
	}
	if p.TopK < 1 {
		errs = append(errs, errors.New("zero-shot audio classification top-k must be greater than zero"))
	}
	return errors.Join(errs...)
}

func (p *ZeroShotAudioClassificationPipeline) preprocess(batch *backends.PipelineBatch, inputs [][]float32) error {
	if len(inputs) == 0 {
		return errors.New("zero-shot audio classification requires at least one waveform")
	}
	for i, samples := range inputs {
		if len(samples) == 0 {
			return fmt.Errorf("waveform %d is empty", i)
		}
	}
	creator, ok := p.Model.Backend.(audioTensorCreator)
	if !ok {
		return errors.New("audio tensor creation is unavailable for the configured backend")
	}
	return creator.CreateAudioTensors(batch, p.Model, inputs)
}

func (p *ZeroShotAudioClassificationPipeline) forward(ctx context.Context, batch *backends.PipelineBatch) error {
	start := time.Now()
	if err := backends.RunSessionOnBatch(ctx, batch, p.BasePipeline); err != nil {
		return err
	}
	atomic.AddUint64(&p.ONNXTimings.NumCalls, 1)
	atomic.AddUint64(&p.ONNXTimings.TotalNS, safeconv.DurationToU64(time.Since(start)))
	return nil
}

func (p *ZeroShotAudioClassificationPipeline) postprocess(batch *backends.PipelineBatch) (*ZeroShotAudioClassificationOutput, error) {
	if len(batch.OutputValues) == 0 {
		return nil, errors.New("zero-shot audio classification produced no outputs")
	}
	logits, ok := batch.OutputValues[0].([][]float32)
	if !ok {
		return nil, fmt.Errorf("zero-shot audio classification output type %T is not supported", batch.OutputValues[0])
	}
	out := &ZeroShotAudioClassificationOutput{Predictions: make([][]ZeroShotAudioClassificationResult, len(logits))}
	for i, row := range logits {
		scores := vectorutil.SoftMax(row)
		indices := make([]int, len(row))
		for j := range indices {
			indices[j] = j
		}
		sort.SliceStable(indices, func(a, b int) bool { return scores[indices[a]] > scores[indices[b]] })
		k := min(p.TopK, len(indices))
		out.Predictions[i] = make([]ZeroShotAudioClassificationResult, k)
		for j, index := range indices[:k] {
			out.Predictions[i][j] = ZeroShotAudioClassificationResult{Label: p.Labels[index], Score: scores[index], ClassIndex: index}
		}
	}
	return out, nil
}

func (p *ZeroShotAudioClassificationPipeline) Run(ctx context.Context, inputs []string) (backends.PipelineBatchOutput, error) {
	return p.RunFiles(ctx, inputs)
}

func (p *ZeroShotAudioClassificationPipeline) RunFiles(ctx context.Context, paths []string) (*ZeroShotAudioClassificationOutput, error) {
	waveforms := make([]AudioWaveform, len(paths))
	for i, path := range paths {
		waveform, err := loadWAV(path)
		if err != nil {
			return nil, fmt.Errorf("failed to load audio %q: %w", path, err)
		}
		waveforms[i] = waveform
	}
	return p.RunWaveforms(ctx, waveforms)
}

func (p *ZeroShotAudioClassificationPipeline) RunWaveforms(ctx context.Context, inputs []AudioWaveform) (*ZeroShotAudioClassificationOutput, error) {
	samples := make([][]float32, len(inputs))
	for i := range inputs {
		samples[i] = inputs[i].Samples
	}
	return p.RunWithAudio(ctx, samples)
}

func (p *ZeroShotAudioClassificationPipeline) RunWithAudio(ctx context.Context, inputs [][]float32) (*ZeroShotAudioClassificationOutput, error) {
	return backends.RunPipeline(ctx, len(inputs), func(batch *backends.PipelineBatch) error { return p.preprocess(batch, inputs) }, p.forward, p.postprocess)
}
