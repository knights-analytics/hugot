package pipelines

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync/atomic"
	"time"

	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/util/safeconv"
)

// TextToAudioOutput contains one mono waveform for every input text.
type TextToAudioOutput struct {
	Audio []AudioWaveform
}

func (o *TextToAudioOutput) GetOutput() []any {
	output := make([]any, len(o.Audio))
	for i := range o.Audio {
		output[i] = o.Audio[i]
	}
	return output
}

// TextToAudioPipeline generates audio waveforms from text inputs.
type TextToAudioPipeline struct {
	*backends.BasePipeline
	Input      backends.InputOutputInfo
	Output     backends.InputOutputInfo
	SampleRate int
}

// TextToAudioConfig configures a TextToAudioPipeline.
type TextToAudioConfig = backends.PipelineConfig[*TextToAudioPipeline]

// TextToAudioOption configures a TextToAudioPipeline.
type TextToAudioOption = backends.PipelineOption[*TextToAudioPipeline]

// WithTextToAudioSampleRate sets the sample rate attached to generated waveforms.
func WithTextToAudioSampleRate(sampleRate int) TextToAudioOption {
	return func(pipeline *TextToAudioPipeline) error {
		if sampleRate <= 0 {
			return errors.New("text-to-audio sample rate must be greater than zero")
		}
		pipeline.SampleRate = sampleRate
		return nil
	}
}

func NewTextToAudioPipeline(ctx context.Context, config TextToAudioConfig, model *backends.Model) (*TextToAudioPipeline, error) {
	if model == nil {
		return nil, errors.New("text-to-audio pipeline requires a model")
	}
	pipeline := &TextToAudioPipeline{BasePipeline: backends.NewBasePipeline(ctx, config, model), SampleRate: 16000}
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

func (p *TextToAudioPipeline) IsGenerative() bool        { return false }
func (p *TextToAudioPipeline) GetModel() *backends.Model { return p.Model }
func (p *TextToAudioPipeline) GetMetadata() backends.PipelineMetadata {
	if p.Output.Name == "" {
		return backends.PipelineMetadata{}
	}
	return backends.PipelineMetadata{OutputsInfo: []backends.OutputInfo{{Name: p.Output.Name, Dimensions: p.Output.Dimensions}}}
}

func (p *TextToAudioPipeline) GetStatistics() backends.PipelineStatistics {
	statistics := backends.PipelineStatistics{}
	if p.TokenizerTimings != nil {
		statistics.ComputeTokenizerStatistics(p.TokenizerTimings)
	}
	if p.ONNXTimings != nil {
		statistics.ComputeOnnxStatistics(p.ONNXTimings)
	}
	return statistics
}

func (p *TextToAudioPipeline) Validate() error {
	if p.Model == nil {
		return errors.New("text-to-audio pipeline requires a model")
	}
	var validationErrors []error
	if p.Model.Tokenizer == nil {
		validationErrors = append(validationErrors, errors.New("text-to-audio pipeline requires a tokenizer"))
	}
	if len(p.Model.InputsMeta) == 0 {
		validationErrors = append(validationErrors, errors.New("text-to-audio model has no inputs"))
	} else {
		p.Input = p.Model.InputsMeta[0]
		for _, input := range p.Model.InputsMeta {
			name := strings.ToLower(input.Name)
			if strings.Contains(name, "input_ids") || strings.Contains(name, "input") {
				p.Input = input
				break
			}
		}
		if len(p.Input.Dimensions) != 2 {
			validationErrors = append(validationErrors, fmt.Errorf("unsupported text-to-audio input layout %q: expected rank 2, got %v", p.Input.Name, p.Input.Dimensions))
		}
	}
	if len(p.Model.OutputsMeta) == 0 {
		validationErrors = append(validationErrors, errors.New("text-to-audio model has no outputs"))
	} else {
		p.Output = p.Model.OutputsMeta[0]
		if len(p.Output.Dimensions) != 2 && len(p.Output.Dimensions) != 3 {
			validationErrors = append(validationErrors, fmt.Errorf("unsupported text-to-audio output layout %q: expected rank 2 or 3, got %v", p.Output.Name, p.Output.Dimensions))
		}
	}
	if p.SampleRate <= 0 {
		validationErrors = append(validationErrors, errors.New("text-to-audio sample rate must be greater than zero"))
	}
	return errors.Join(validationErrors...)
}

func (p *TextToAudioPipeline) preprocess(batch *backends.PipelineBatch, inputs []string) error {
	if len(inputs) == 0 {
		return errors.New("text-to-audio requires at least one input")
	}
	for i, input := range inputs {
		if strings.TrimSpace(input) == "" {
			return fmt.Errorf("text input %d is empty", i)
		}
	}
	start := time.Now()
	backends.TokenizeInputs(batch, p.Model.Tokenizer, inputs)
	if p.TokenizerTimings != nil {
		atomic.AddUint64(&p.TokenizerTimings.NumCalls, 1)
		atomic.AddUint64(&p.TokenizerTimings.TotalNS, safeconv.DurationToU64(time.Since(start)))
	}
	return backends.CreateInputTensors(batch, p.Model)
}

func (p *TextToAudioPipeline) forward(ctx context.Context, batch *backends.PipelineBatch) error {
	start := time.Now()
	if err := backends.RunSessionOnBatch(ctx, batch, p.BasePipeline); err != nil {
		return err
	}
	if p.ONNXTimings != nil {
		atomic.AddUint64(&p.ONNXTimings.NumCalls, 1)
		atomic.AddUint64(&p.ONNXTimings.TotalNS, safeconv.DurationToU64(time.Since(start)))
	}
	return nil
}

func (p *TextToAudioPipeline) postprocess(batch *backends.PipelineBatch) (*TextToAudioOutput, error) {
	if len(batch.OutputValues) == 0 {
		return nil, errors.New("text-to-audio produced no outputs")
	}
	waveforms, err := audioOutputWaveforms(batch.OutputValues[0], batch.Size, p.SampleRate)
	if err != nil {
		return nil, err
	}
	return &TextToAudioOutput{Audio: waveforms}, nil
}

// Run executes the pipeline for a batch of text inputs.
func (p *TextToAudioPipeline) Run(ctx context.Context, inputs []string) (backends.PipelineBatchOutput, error) {
	return p.RunText(ctx, inputs)
}

// RunText generates waveforms from text inputs.
func (p *TextToAudioPipeline) RunText(ctx context.Context, inputs []string) (*TextToAudioOutput, error) {
	return backends.RunPipeline(ctx, len(inputs), func(batch *backends.PipelineBatch) error {
		return p.preprocess(batch, inputs)
	}, p.forward, p.postprocess)
}

func audioOutputWaveforms(value any, batchSize, sampleRate int) ([]AudioWaveform, error) {
	result := make([]AudioWaveform, batchSize)
	switch audio := value.(type) {
	case [][]float32:
		if len(audio) != batchSize {
			return nil, fmt.Errorf("text-to-audio output batch size %d does not match input batch size %d", len(audio), batchSize)
		}
		for i := range audio {
			result[i] = AudioWaveform{Samples: audio[i], SampleRate: sampleRate}
		}
	case [][][]float32:
		if len(audio) != batchSize {
			return nil, fmt.Errorf("text-to-audio output batch size %d does not match input batch size %d", len(audio), batchSize)
		}
		for i, channels := range audio {
			if len(channels) != 1 {
				return nil, fmt.Errorf("text-to-audio output waveform %d has %d channels; only mono audio is supported", i, len(channels))
			}
			result[i] = AudioWaveform{Samples: channels[0], SampleRate: sampleRate}
		}
	default:
		return nil, fmt.Errorf("unsupported text-to-audio output type %T", value)
	}
	for i, waveform := range result {
		if len(waveform.Samples) == 0 {
			return nil, fmt.Errorf("text-to-audio output waveform %d is empty", i)
		}
	}
	return result, nil
}

// TextToSpeechOutput is the typed output of TextToSpeechPipeline.
type TextToSpeechOutput = TextToAudioOutput

// TextToSpeechPipeline generates speech waveforms from text inputs.
type TextToSpeechPipeline struct{ TextToAudioPipeline }

type (
	TextToSpeechConfig = backends.PipelineConfig[*TextToSpeechPipeline]
	TextToSpeechOption = backends.PipelineOption[*TextToSpeechPipeline]
)

// WithTextToSpeechSampleRate sets the sample rate attached to generated speech waveforms.
func WithTextToSpeechSampleRate(sampleRate int) TextToSpeechOption {
	return func(pipeline *TextToSpeechPipeline) error {
		return WithTextToAudioSampleRate(sampleRate)(&pipeline.TextToAudioPipeline)
	}
}

func NewTextToSpeechPipeline(ctx context.Context, config TextToSpeechConfig, model *backends.Model) (*TextToSpeechPipeline, error) {
	if model == nil {
		return nil, errors.New("text-to-speech pipeline requires a model")
	}
	pipeline := &TextToSpeechPipeline{BasePipeline: backends.NewBasePipeline(ctx, config, model), SampleRate: 16000}
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

func (p *TextToSpeechPipeline) IsGenerative() bool { return false }

func (p *TextToSpeechPipeline) Run(ctx context.Context, inputs []string) (backends.PipelineBatchOutput, error) {
	return p.RunText(ctx, inputs)
}

func (p *TextToSpeechPipeline) RunText(ctx context.Context, inputs []string) (*TextToSpeechOutput, error) {
	return p.TextToAudioPipeline.RunText(ctx, inputs)
}
