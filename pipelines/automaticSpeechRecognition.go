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

// AutomaticSpeechRecognitionOutput contains one greedy CTC transcription per waveform.
type AutomaticSpeechRecognitionOutput struct {
	Text  []string
	Words [][]string
}

func (o *AutomaticSpeechRecognitionOutput) GetOutput() []any {
	result := make([]any, len(o.Text))
	for i, text := range o.Text {
		result[i] = text
	}
	return result
}

// AutomaticSpeechRecognitionPipeline decodes frame-level CTC logits.
type AutomaticSpeechRecognitionPipeline struct {
	*backends.BasePipeline
	IDLabelMap map[int]string
	BlankID    int
	Input      backends.InputOutputInfo
}

func (p *AutomaticSpeechRecognitionPipeline) IsGenerative() bool { return false }

func NewAutomaticSpeechRecognitionPipeline(ctx context.Context, config backends.PipelineConfig[*AutomaticSpeechRecognitionPipeline], model *backends.Model) (*AutomaticSpeechRecognitionPipeline, error) {
	pipeline := &AutomaticSpeechRecognitionPipeline{BasePipeline: backends.NewBasePipeline(ctx, config, model), IDLabelMap: model.IDLabelMap, BlankID: 0}
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

func WithASRBlankID(id int) backends.PipelineOption[*AutomaticSpeechRecognitionPipeline] {
	return func(p *AutomaticSpeechRecognitionPipeline) error {
		if id < 0 {
			return errors.New("ASR blank ID must not be negative")
		}
		p.BlankID = id
		return nil
	}
}

func (p *AutomaticSpeechRecognitionPipeline) Validate() error {
	if len(p.Model.InputsMeta) == 0 {
		return errors.New("automatic speech recognition model has no inputs")
	}
	p.Input = p.Model.InputsMeta[0]
	for _, input := range p.Model.InputsMeta {
		name := strings.ToLower(input.Name)
		if strings.Contains(name, "input_values") || strings.Contains(name, "waveform") || strings.Contains(name, "audio") {
			p.Input = input
			break
		}
	}
	if len(p.Input.Dimensions) != 2 && len(p.Input.Dimensions) != 3 {
		return fmt.Errorf("unsupported ASR input layout %q: expected rank 2 or 3, got %v", p.Input.Name, p.Input.Dimensions)
	}
	if len(p.Model.OutputsMeta) == 0 {
		return errors.New("automatic speech recognition model has no outputs")
	}
	if len(p.Model.OutputsMeta[0].Dimensions) != 3 {
		return fmt.Errorf("unsupported ASR output layout: expected [batch,time,vocabulary], got %v", p.Model.OutputsMeta[0].Dimensions)
	}
	return nil
}

func (p *AutomaticSpeechRecognitionPipeline) GetModel() *backends.Model { return p.Model }
func (p *AutomaticSpeechRecognitionPipeline) GetMetadata() backends.PipelineMetadata {
	return backends.PipelineMetadata{}
}

func (p *AutomaticSpeechRecognitionPipeline) GetStatistics() backends.PipelineStatistics {
	s := backends.PipelineStatistics{}
	s.ComputeOnnxStatistics(p.ONNXTimings)
	return s
}

func (p *AutomaticSpeechRecognitionPipeline) preprocess(batch *backends.PipelineBatch, inputs [][]float32) error {
	for i, input := range inputs {
		if len(input) == 0 {
			return fmt.Errorf("waveform %d is empty", i)
		}
	}
	return p.Model.Backend.CreateAudioTensors(batch, p.Model, inputs)
}

func (p *AutomaticSpeechRecognitionPipeline) forward(ctx context.Context, batch *backends.PipelineBatch) error {
	start := time.Now()
	if err := backends.RunSessionOnBatch(ctx, batch, p.BasePipeline); err != nil {
		return err
	}
	atomic.AddUint64(&p.ONNXTimings.NumCalls, 1)
	atomic.AddUint64(&p.ONNXTimings.TotalNS, safeconv.DurationToU64(time.Since(start)))
	return nil
}

func (p *AutomaticSpeechRecognitionPipeline) postprocess(batch *backends.PipelineBatch) (*AutomaticSpeechRecognitionOutput, error) {
	if len(batch.OutputValues) == 0 {
		return nil, errors.New("automatic speech recognition produced no outputs")
	}
	logits, ok := batch.OutputValues[0].([][][]float32)
	if !ok {
		return nil, fmt.Errorf("unsupported ASR output type %T", batch.OutputValues[0])
	}
	output := &AutomaticSpeechRecognitionOutput{Text: make([]string, len(logits)), Words: make([][]string, len(logits))}
	for i, frames := range logits {
		ids := make([]uint32, 0, len(frames))
		words := make([]string, 0, len(frames))
		previous := p.BlankID
		for _, frame := range frames {
			if len(frame) == 0 {
				continue
			}
			best := 0
			for j := 1; j < len(frame); j++ {
				if frame[j] > frame[best] {
					best = j
				}
			}
			if best == p.BlankID || best == previous {
				previous = best
				continue
			}
			previous = best
			if label, exists := p.IDLabelMap[best]; exists {
				words = append(words, label)
			} else {
				ids = append(ids, uint32(best))
			}
		}
		if len(ids) > 0 && p.Model.Tokenizer != nil {
			decoded, err := backends.Decode(ids, p.Model.Tokenizer)
			if err != nil {
				return nil, err
			}
			output.Text[i] = decoded
		} else {
			output.Text[i] = strings.Join(words, "")
		}
		output.Words[i] = words
	}
	return output, nil
}

func (p *AutomaticSpeechRecognitionPipeline) Run(ctx context.Context, inputs []string) (backends.PipelineBatchOutput, error) {
	return p.RunFiles(ctx, inputs)
}

func (p *AutomaticSpeechRecognitionPipeline) RunFiles(ctx context.Context, paths []string) (*AutomaticSpeechRecognitionOutput, error) {
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

func (p *AutomaticSpeechRecognitionPipeline) RunWaveforms(ctx context.Context, inputs []AudioWaveform) (*AutomaticSpeechRecognitionOutput, error) {
	samples := make([][]float32, len(inputs))
	for i := range inputs {
		samples[i] = inputs[i].Samples
	}
	return p.RunWithAudio(ctx, samples)
}

func (p *AutomaticSpeechRecognitionPipeline) RunWithAudio(ctx context.Context, inputs [][]float32) (*AutomaticSpeechRecognitionOutput, error) {
	return backends.RunPipeline(ctx, len(inputs), func(batch *backends.PipelineBatch) error { return p.preprocess(batch, inputs) }, p.forward, p.postprocess)
}
