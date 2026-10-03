package pipelines

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"
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
// The model compares an audio input with tokenized candidate text and returns pairwise similarities.
type ZeroShotAudioClassificationPipeline struct {
	*backends.BasePipeline
	Labels             []string
	TopK               int
	Input              backends.InputOutputInfo
	SampleRate         int
	HypothesisTemplate string
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

// WithZeroShotAudioHypothesisTemplate sets the text template used to describe each candidate label.
func WithZeroShotAudioHypothesisTemplate(template string) ZeroShotAudioClassificationOption {
	return func(p *ZeroShotAudioClassificationPipeline) error {
		if !strings.Contains(template, "{}") {
			return errors.New("zero-shot audio hypothesis template must contain {}")
		}
		p.HypothesisTemplate = template
		return nil
	}
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

// WithZeroShotAudioSampleRate sets the sample rate expected by the model.
func WithZeroShotAudioSampleRate(sampleRate int) ZeroShotAudioClassificationOption {
	return func(p *ZeroShotAudioClassificationPipeline) error {
		if sampleRate <= 0 {
			return errors.New("zero-shot audio sample rate must be greater than zero")
		}
		p.SampleRate = sampleRate
		return nil
	}
}

func NewZeroShotAudioClassificationPipeline(ctx context.Context, config ZeroShotAudioClassificationConfig, model *backends.Model) (*ZeroShotAudioClassificationPipeline, error) {
	if model == nil {
		return nil, errors.New("zero-shot audio classification requires a model")
	}
	p := &ZeroShotAudioClassificationPipeline{
		BasePipeline:       backends.NewBasePipeline(ctx, config, model),
		TopK:               5,
		SampleRate:         48000,
		HypothesisTemplate: "This is a sound of {}.",
	}
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
	if p.BasePipeline == nil || p.Model == nil {
		return errors.New("zero-shot audio classification requires a model")
	}
	var errs []error
	if len(p.Labels) == 0 {
		errs = append(errs, errors.New("zero-shot audio classification requires at least one candidate label"))
	}
	inputFeaturesFound, inputIDsFound := false, false
	if len(p.Model.InputsMeta) == 0 {
		errs = append(errs, errors.New("zero-shot audio classification model has no inputs"))
	} else {
		for _, input := range p.Model.InputsMeta {
			switch input.Name {
			case "input_features":
				p.Input = input
				inputFeaturesFound = true
				if len(input.Dimensions) != 4 {
					errs = append(errs, fmt.Errorf("CLAP input_features must have rank 4, got %v", input.Dimensions))
				}
			case "is_longer":
			case "input_ids":
				inputIDsFound = true
			}
		}
		if !inputFeaturesFound {
			errs = append(errs, errors.New("zero-shot audio classification requires a CLAP input_features tensor"))
		}
		if !inputIDsFound {
			errs = append(errs, errors.New("zero-shot audio classification requires text input_ids"))
		}
	}
	if len(p.Model.OutputsMeta) == 0 {
		errs = append(errs, errors.New("zero-shot audio classification model has no outputs"))
	} else {
		dims := p.Model.OutputsMeta[0].Dimensions
		if len(dims) != 2 {
			errs = append(errs, fmt.Errorf("zero-shot audio classification output must have shape [audio, text], got %v", dims))
		}
	}
	if p.TopK < 1 {
		errs = append(errs, errors.New("zero-shot audio classification top-k must be greater than zero"))
	}
	if p.SampleRate <= 0 {
		errs = append(errs, errors.New("zero-shot audio sample rate must be greater than zero"))
	}
	if !strings.Contains(p.HypothesisTemplate, "{}") {
		errs = append(errs, errors.New("zero-shot audio hypothesis template must contain {}"))
	}
	return errors.Join(errs...)
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

func zeroShotAudioSimilarity(batch *backends.PipelineBatch) (float32, error) {
	if len(batch.OutputValues) == 0 {
		return 0, errors.New("zero-shot audio classification produced no outputs")
	}
	logits, ok := batch.OutputValues[0].([][]float32)
	if !ok {
		return 0, fmt.Errorf("zero-shot audio classification output type %T is not supported", batch.OutputValues[0])
	}
	if len(logits) != 1 || len(logits[0]) != 1 {
		return 0, fmt.Errorf("CLAP pairwise output must contain one audio/text score, got %v", logits)
	}
	return logits[0][0], nil
}

func rankZeroShotAudio(labels []string, similarities []float32, topK int) [][]ZeroShotAudioClassificationResult {
	scores := vectorutil.SoftMax(similarities)
	indices := make([]int, len(labels))
	for i := range indices {
		indices[i] = i
	}
	sort.SliceStable(indices, func(i, j int) bool { return scores[indices[i]] > scores[indices[j]] })
	k := min(topK, len(indices))
	results := make([]ZeroShotAudioClassificationResult, k)
	for i, index := range indices[:k] {
		results[i] = ZeroShotAudioClassificationResult{Label: labels[index], Score: scores[index], ClassIndex: index}
	}
	return [][]ZeroShotAudioClassificationResult{results}
}

func (p *ZeroShotAudioClassificationPipeline) Run(ctx context.Context, inputs []string) (backends.PipelineBatchOutput, error) {
	return p.RunFiles(ctx, inputs)
}

func (p *ZeroShotAudioClassificationPipeline) RunFiles(ctx context.Context, paths []string) (*ZeroShotAudioClassificationOutput, error) {
	waveforms, err := loadAudioFiles(paths)
	if err != nil {
		return nil, err
	}
	for i := range waveforms {
		waveforms[i].Samples = resampleAudio(waveforms[i].Samples, waveforms[i].SampleRate, p.SampleRate)
		waveforms[i].SampleRate = p.SampleRate
	}
	return p.RunWaveforms(ctx, waveforms)
}

func (p *ZeroShotAudioClassificationPipeline) RunWaveforms(ctx context.Context, inputs []AudioWaveform) (*ZeroShotAudioClassificationOutput, error) {
	samples, err := audioSamples(inputs, p.SampleRate)
	if err != nil {
		return nil, err
	}
	return p.RunWithAudio(ctx, samples)
}

// RunWithAudio executes mono samples already sampled at the pipeline's configured SampleRate.
func (p *ZeroShotAudioClassificationPipeline) RunWithAudio(ctx context.Context, inputs [][]float32) (*ZeroShotAudioClassificationOutput, error) {
	if len(inputs) == 0 {
		return nil, errors.New("zero-shot audio classification requires at least one waveform")
	}
	if len(p.Labels) == 0 {
		return nil, errors.New("zero-shot audio classification requires candidate labels")
	}
	if p.Model.Tokenizer == nil {
		return nil, errors.New("zero-shot audio classification requires a tokenizer for candidate labels")
	}
	output := &ZeroShotAudioClassificationOutput{Predictions: make([][]ZeroShotAudioClassificationResult, len(inputs))}
	for audioIndex, samples := range inputs {
		features, isLonger, err := clapAudioFeatures(samples, p.SampleRate)
		if err != nil {
			return nil, fmt.Errorf("audio %d: %w", audioIndex, err)
		}
		similarities := make([]float32, len(p.Labels))
		for labelIndex, label := range p.Labels {
			if strings.TrimSpace(label) == "" {
				return nil, fmt.Errorf("candidate label %d is empty", labelIndex)
			}
			text := strings.Replace(p.HypothesisTemplate, "{}", label, 1)
			pair, err := backends.RunPipeline(ctx, 1, func(batch *backends.PipelineBatch) error {
				backends.TokenizeInputs(batch, p.Model.Tokenizer, []string{text})
				return backends.CreateAudioTextTensors(batch, p.Model, [][][][]float32{features}, []bool{isLonger}, batch.Input)
			}, p.forward, func(batch *backends.PipelineBatch) (*ZeroShotAudioClassificationOutput, error) {
				similarity, err := zeroShotAudioSimilarity(batch)
				if err != nil {
					return nil, err
				}
				return &ZeroShotAudioClassificationOutput{Predictions: [][]ZeroShotAudioClassificationResult{{{Score: similarity}}}}, nil
			})
			if err != nil {
				return nil, err
			}
			similarities[labelIndex] = pair.Predictions[0][0].Score
		}
		output.Predictions[audioIndex] = rankZeroShotAudio(p.Labels, similarities, p.TopK)[0]
	}
	return output, nil
}
