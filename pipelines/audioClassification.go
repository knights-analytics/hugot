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

// AudioWaveform is a mono waveform and its sampling rate. Samples are expected
// to be normalized to the range [-1, 1].
type AudioWaveform struct {
	Samples    []float32
	SampleRate int
}

type AudioClassificationResult struct {
	Label      string
	Score      float32
	ClassIndex int
}

type AudioClassificationOutput struct {
	Predictions [][]AudioClassificationResult
}

// AudioClassificationPipeline classifies mono audio waveforms.
type AudioClassificationPipeline struct {
	*backends.BasePipeline
	IDLabelMap map[int]string
	Input      backends.InputOutputInfo
	TopK       int
	SampleRate int
}

func (o *AudioClassificationOutput) GetOutput() []any {
	out := make([]any, len(o.Predictions))
	for i, predictions := range o.Predictions {
		out[i] = any(predictions)
	}
	return out
}

func (p *AudioClassificationPipeline) IsGenerative() bool { return false }

// WithAudioTopK sets the number of classifications returned per waveform.
func WithAudioTopK(topK int) backends.PipelineOption[*AudioClassificationPipeline] {
	return func(pipeline *AudioClassificationPipeline) error {
		if topK < 1 {
			return fmt.Errorf("audio classification top-k must be greater than zero")
		}
		pipeline.TopK = topK
		return nil
	}
}

func NewAudioClassificationPipeline(ctx context.Context, config backends.PipelineConfig[*AudioClassificationPipeline], model *backends.Model) (*AudioClassificationPipeline, error) {
	if model == nil {
		return nil, errors.New("audio classification pipeline requires a model")
	}
	pipeline := &AudioClassificationPipeline{
		BasePipeline: backends.NewBasePipeline(ctx, config, model),
		IDLabelMap:   model.IDLabelMap,
		TopK:         5,
		SampleRate:   defaultAudioSampleRate,
	}
	for _, option := range config.Options {
		if err := option(pipeline); err != nil {
			return nil, err
		}
	}
	if err := pipeline.Validate(); err != nil {
		return nil, err
	}
	return pipeline, nil
}

// WithAudioClassificationSampleRate sets the sample rate expected by the model.
func WithAudioClassificationSampleRate(sampleRate int) backends.PipelineOption[*AudioClassificationPipeline] {
	return func(pipeline *AudioClassificationPipeline) error {
		if sampleRate <= 0 {
			return errors.New("audio classification sample rate must be greater than zero")
		}
		pipeline.SampleRate = sampleRate
		return nil
	}
}

func (p *AudioClassificationPipeline) GetModel() *backends.Model { return p.Model }

func (p *AudioClassificationPipeline) GetMetadata() backends.PipelineMetadata {
	if p == nil || p.Model == nil || len(p.Model.OutputsMeta) == 0 {
		return backends.PipelineMetadata{}
	}
	return backends.PipelineMetadata{OutputsInfo: []backends.OutputInfo{{
		Name: p.Model.OutputsMeta[0].Name, Dimensions: p.Model.OutputsMeta[0].Dimensions,
	}}}
}

func (p *AudioClassificationPipeline) GetStatistics() backends.PipelineStatistics {
	statistics := backends.PipelineStatistics{}
	statistics.ComputeOnnxStatistics(p.ONNXTimings)
	return statistics
}

func (p *AudioClassificationPipeline) Validate() error {
	var validationErrors []error
	if len(p.Model.InputsMeta) == 0 {
		validationErrors = append(validationErrors, errors.New("audio classification model has no inputs"))
	} else {
		input := p.Model.InputsMeta[0]
		for _, candidate := range p.Model.InputsMeta {
			name := strings.ToLower(candidate.Name)
			if strings.Contains(name, "input_values") || strings.Contains(name, "waveform") || strings.Contains(name, "audio") {
				input = candidate
				break
			}
		}
		p.Input = input
		dims := input.Dimensions
		if len(dims) != 2 && len(dims) != 3 {
			validationErrors = append(validationErrors, fmt.Errorf("unsupported audio input layout %q: expected rank 2 or 3, got %v", input.Name, dims))
		} else if dims[0] != -1 && dims[0] != 1 {
			validationErrors = append(validationErrors, fmt.Errorf("unsupported audio input batch dimension %d: expected -1 or 1", dims[0]))
		} else if dims[len(dims)-1] == 0 {
			validationErrors = append(validationErrors, fmt.Errorf("unsupported audio input sample dimension 0"))
		}
	}
	if len(p.Model.OutputsMeta) == 0 {
		validationErrors = append(validationErrors, errors.New("audio classification model has no outputs"))
	} else {
		dims := p.Model.OutputsMeta[0].Dimensions
		if len(dims) != 2 {
			validationErrors = append(validationErrors, fmt.Errorf("audio classification output must have shape [batch, classes], got %v", dims))
		} else if dims[1] > 0 && len(p.IDLabelMap) > 0 && int(dims[1]) != len(p.IDLabelMap) {
			validationErrors = append(validationErrors, fmt.Errorf("audio classification labels (%d) do not match logits (%d)", len(p.IDLabelMap), dims[1]))
		}
	}
	if p.TopK < 1 {
		validationErrors = append(validationErrors, errors.New("audio classification top-k must be greater than zero"))
	}
	return errors.Join(validationErrors...)
}

func (p *AudioClassificationPipeline) preprocess(batch *backends.PipelineBatch, inputs [][]float32) error {
	return createAudioInputTensors(batch, p.Model, inputs)
}

func (p *AudioClassificationPipeline) forward(ctx context.Context, batch *backends.PipelineBatch) error {
	start := time.Now()
	if err := backends.RunSessionOnBatch(ctx, batch, p.BasePipeline); err != nil {
		return err
	}
	atomic.AddUint64(&p.ONNXTimings.NumCalls, 1)
	atomic.AddUint64(&p.ONNXTimings.TotalNS, safeconv.DurationToU64(time.Since(start)))
	return nil
}

func (p *AudioClassificationPipeline) postprocess(batch *backends.PipelineBatch) (*AudioClassificationOutput, error) {
	if len(batch.OutputValues) == 0 {
		return nil, errors.New("audio classification produced no outputs")
	}
	logits, ok := batch.OutputValues[0].([][]float32)
	if !ok {
		return nil, fmt.Errorf("audio classification output type %T is not supported", batch.OutputValues[0])
	}
	output := &AudioClassificationOutput{Predictions: make([][]AudioClassificationResult, len(logits))}
	for i, row := range logits {
		output.Predictions[i] = topKAudio(vectorutil.SoftMax(row), p.TopK, p.IDLabelMap)
	}
	return output, nil
}

func topKAudio(logits []float32, k int, labels map[int]string) []AudioClassificationResult {
	indices := make([]int, len(logits))
	for i := range indices {
		indices[i] = i
	}
	sort.SliceStable(indices, func(i, j int) bool { return logits[indices[i]] > logits[indices[j]] })
	if k > len(indices) {
		k = len(indices)
	}
	results := make([]AudioClassificationResult, k)
	for i, index := range indices[:k] {
		label := fmt.Sprintf("class_%d", index)
		if value, ok := labels[index]; ok {
			label = value
		}
		results[i] = AudioClassificationResult{Label: label, Score: logits[index], ClassIndex: index}
	}
	return results
}

// Run executes the pipeline on PCM WAV file paths.
func (p *AudioClassificationPipeline) Run(ctx context.Context, inputs []string) (backends.PipelineBatchOutput, error) {
	return p.RunFiles(ctx, inputs)
}

func (p *AudioClassificationPipeline) RunFiles(ctx context.Context, paths []string) (*AudioClassificationOutput, error) {
	waveforms, err := loadAudioFiles(paths)
	if err != nil {
		return nil, err
	}
	return p.RunWaveforms(ctx, waveforms)
}

func (p *AudioClassificationPipeline) RunWaveforms(ctx context.Context, inputs []AudioWaveform) (*AudioClassificationOutput, error) {
	samples, err := audioSamples(inputs, p.SampleRate)
	if err != nil {
		return nil, err
	}
	return p.RunWithAudio(ctx, samples)
}

// RunWithAudio executes mono samples already sampled at the pipeline's configured SampleRate.
func (p *AudioClassificationPipeline) RunWithAudio(ctx context.Context, inputs [][]float32) (*AudioClassificationOutput, error) {
	return backends.RunPipeline(ctx, len(inputs), func(batch *backends.PipelineBatch) error {
		return p.preprocess(batch, inputs)
	}, p.forward, p.postprocess)
}
