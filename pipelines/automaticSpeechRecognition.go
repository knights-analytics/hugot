package pipelines

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"strings"
	"sync/atomic"
	"time"

	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/util/safeconv"
)

// AutomaticSpeechRecognitionOutput contains one transcription per waveform.
type AutomaticSpeechRecognitionOutput struct {
	Text              []string
	Words             [][]string
	WordTimestamps    [][]ASRTimestamp
	SegmentTimestamps [][]ASRTimestamp
}

// ASRTimestamp identifies the text and time range of a recognized word or segment.
type ASRTimestamp = backends.ASRTimestamp

func (o *AutomaticSpeechRecognitionOutput) GetOutput() []any {
	result := make([]any, len(o.Text))
	for i, text := range o.Text {
		result[i] = text
	}
	return result
}

// AutomaticSpeechRecognitionPipeline recognizes speech with CTC or a GenAI model.
type AutomaticSpeechRecognitionPipeline struct {
	*backends.BasePipeline
	IDLabelMap       map[int]string
	BlankID          int
	Input            backends.InputOutputInfo
	SampleRate       int
	Prompt           string
	StreamingOptions StreamingASROptions
}

// StreamingASROptions configures PCM chunk sizing, VAD, and language settings.
type StreamingASROptions = backends.StreamingASROptions

// WithASRStreamingOptions configures a package loaded with WithGenAIStreamingASR.
// ChunkSamples must be the package's maximum chunk size at its native sample rate.
func WithASRStreamingOptions(opts StreamingASROptions) backends.PipelineOption[*AutomaticSpeechRecognitionPipeline] {
	return func(p *AutomaticSpeechRecognitionPipeline) error {
		if opts.ChunkSamples <= 0 {
			return errors.New("streaming ASR ChunkSamples must be greater than zero")
		}
		opts.ProcessorOptions = maps.Clone(opts.ProcessorOptions)
		opts.RuntimeOptions = maps.Clone(opts.RuntimeOptions)
		p.StreamingOptions = opts
		return nil
	}
}

func (p *AutomaticSpeechRecognitionPipeline) IsGenerative() bool {
	return p != nil && p.Model != nil && p.Model.IsGenerative
}

// SupportsGenerativeModelLoading allows callers to select GenAI explicitly
// while keeping ONNX/CTC as the default loading mode for this pipeline.
func (*AutomaticSpeechRecognitionPipeline) SupportsGenerativeModelLoading() bool { return true }

func NewAutomaticSpeechRecognitionPipeline(ctx context.Context, config backends.PipelineConfig[*AutomaticSpeechRecognitionPipeline], model *backends.Model) (*AutomaticSpeechRecognitionPipeline, error) {
	pipeline := &AutomaticSpeechRecognitionPipeline{BasePipeline: backends.NewBasePipeline(ctx, config, model), IDLabelMap: model.IDLabelMap, BlankID: 0, SampleRate: defaultAudioSampleRate, Prompt: "Transcribe the audio."}
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

// WithASRPrompt sets the model-specific user prompt used for generative ASR.
func WithASRPrompt(prompt string) backends.PipelineOption[*AutomaticSpeechRecognitionPipeline] {
	return func(pipeline *AutomaticSpeechRecognitionPipeline) error {
		if strings.TrimSpace(prompt) == "" {
			return errors.New("ASR prompt must not be empty")
		}
		pipeline.Prompt = prompt
		return nil
	}
}

// WithASRSampleRate sets the sample rate expected by the model.
func WithASRSampleRate(sampleRate int) backends.PipelineOption[*AutomaticSpeechRecognitionPipeline] {
	return func(pipeline *AutomaticSpeechRecognitionPipeline) error {
		if sampleRate <= 0 {
			return errors.New("ASR sample rate must be greater than zero")
		}
		pipeline.SampleRate = sampleRate
		return nil
	}
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
	if p.Model != nil && p.Model.UsesStreamingASR() {
		if p.StreamingOptions.ChunkSamples <= 0 {
			return errors.New("WithGenAIStreamingASR requires WithASRStreamingOptions with a positive ChunkSamples setting")
		}
		if p.SampleRate <= 0 {
			return errors.New("ASR sample rate must be greater than zero")
		}
		return nil
	}
	if p.StreamingOptions.ChunkSamples != 0 || len(p.StreamingOptions.ProcessorOptions) > 0 || len(p.StreamingOptions.RuntimeOptions) > 0 {
		return errors.New("ASR streaming options require WithGenAIStreamingASR")
	}
	if p.Model != nil && p.Model.IsGenerative {
		if err := p.Model.ValidateMultimodalSession(); err != nil {
			return err
		}
		if strings.TrimSpace(p.Prompt) == "" {
			return errors.New("ASR prompt must not be empty")
		}
		return nil
	}
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
	if p.Model != nil && p.Model.ORTModel != nil && p.Model.ORTModel.Generative != nil {
		return p.Model.ORTModel.Generative.Statistics()
	}
	s := backends.PipelineStatistics{}
	s.ComputeOnnxStatistics(p.ONNXTimings)
	return s
}

func (p *AutomaticSpeechRecognitionPipeline) preprocess(batch *backends.PipelineBatch, inputs [][]float32) error {
	return createAudioInputTensors(batch, p.Model, inputs)
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
			ids = append(ids, uint32(best))
			if label, exists := p.IDLabelMap[best]; exists {
				words = append(words, label)
			} else {
				words = append(words, fmt.Sprintf("%d", best))
			}
		}
		tokens := words
		if p.Model.Tokenizer != nil && p.Model.Tokenizer.GoTokenizer != nil {
			tokens = make([]string, len(ids))
			for j, id := range ids {
				if token, found := p.Model.Tokenizer.GoTokenizer.Tokenizer.IDToToken(int(id)); found {
					tokens[j] = token
				} else {
					tokens[j] = words[j]
				}
			}
		}
		output.Text[i] = decodeCTCText(tokens)
		output.Words[i] = words
	}
	return output, nil
}

func decodeCTCText(tokens []string) string {
	text := strings.ReplaceAll(strings.Join(tokens, ""), "|", " ")
	return strings.Join(strings.Fields(text), " ")
}

func (p *AutomaticSpeechRecognitionPipeline) Run(ctx context.Context, inputs []string) (backends.PipelineBatchOutput, error) {
	return p.RunFiles(ctx, inputs)
}

func (p *AutomaticSpeechRecognitionPipeline) RunFiles(ctx context.Context, paths []string) (*AutomaticSpeechRecognitionOutput, error) {
	waveforms, err := loadAudioFiles(paths)
	if err != nil {
		return nil, err
	}
	return p.RunWaveforms(ctx, waveforms)
}

func (p *AutomaticSpeechRecognitionPipeline) RunWaveforms(ctx context.Context, inputs []AudioWaveform) (*AutomaticSpeechRecognitionOutput, error) {
	samples, err := audioSamples(inputs, p.SampleRate)
	if err != nil {
		return nil, err
	}
	return p.RunWithAudio(ctx, samples)
}

// RunWithAudio executes mono samples already sampled at the pipeline's configured SampleRate.
func (p *AutomaticSpeechRecognitionPipeline) RunWithAudio(ctx context.Context, inputs [][]float32) (*AutomaticSpeechRecognitionOutput, error) {
	if p.Model != nil && p.Model.IsGenerative {
		var results []backends.ASRResult
		var err error
		if p.Model.UsesStreamingASR() {
			results, err = p.runStreamingAudio(ctx, inputs)
		} else {
			results, err = backends.RunGenerativeASR(ctx, p.BasePipeline, inputs, p.SampleRate, p.Prompt)
		}
		if err != nil {
			return nil, err
		}
		output := &AutomaticSpeechRecognitionOutput{
			Text:              make([]string, len(results)),
			Words:             make([][]string, len(results)),
			WordTimestamps:    make([][]ASRTimestamp, len(results)),
			SegmentTimestamps: make([][]ASRTimestamp, len(results)),
		}
		for i, result := range results {
			output.Text[i] = result.Text
			output.WordTimestamps[i] = result.Words
			output.SegmentTimestamps[i] = result.Segments
			output.Words[i] = make([]string, len(result.Words))
			for j, word := range result.Words {
				output.Words[i][j] = word.Text
			}
		}
		return output, nil
	}
	return backends.RunPipeline(ctx, len(inputs), func(batch *backends.PipelineBatch) error { return p.preprocess(batch, inputs) }, p.forward, p.postprocess)
}

// RunAudioStream accepts mono PCM at SampleRate and emits incremental transcript
// fragments. Ownership of each chunk transfers to the pipeline; do not mutate it.
// Chunks larger than ChunkSamples are split automatically. Close chunks to flush
// the stream, and consume both output channels until closed. Cancel ctx to stop early.
func (p *AutomaticSpeechRecognitionPipeline) RunAudioStream(ctx context.Context, chunks <-chan []float32) (<-chan backends.ASRResult, <-chan error, error) {
	if p == nil || p.Model == nil || !p.Model.UsesStreamingASR() {
		return nil, nil, errors.New("RunAudioStream requires a model loaded with WithGenAIStreamingASR")
	}
	return backends.RunStreamingASR(ctx, p.BasePipeline, chunks, p.StreamingOptions)
}

func (p *AutomaticSpeechRecognitionPipeline) runStreamingAudio(ctx context.Context, inputs [][]float32) ([]backends.ASRResult, error) {
	if len(inputs) == 0 {
		return nil, errors.New("at least one ASR waveform is required")
	}
	results := make([]backends.ASRResult, len(inputs))
	for i, waveform := range inputs {
		if len(waveform) == 0 {
			return nil, fmt.Errorf("waveform %d is empty", i)
		}
		chunks := make(chan []float32, 1)
		chunks <- waveform
		close(chunks)
		stream, errs, err := p.RunAudioStream(ctx, chunks)
		if err != nil {
			return nil, err
		}
		var streamErr error
		for stream != nil || errs != nil {
			select {
			case fragment, ok := <-stream:
				if !ok {
					stream = nil
					continue
				}
				results[i].Text += fragment.Text
				results[i].Words = append(results[i].Words, fragment.Words...)
				results[i].Segments = append(results[i].Segments, fragment.Segments...)
			case err, ok := <-errs:
				if !ok {
					errs = nil
					continue
				}
				streamErr = errors.Join(streamErr, err)
			}
		}
		if streamErr != nil {
			return nil, fmt.Errorf("waveform %d: %w", i, streamErr)
		}
	}
	return results, nil
}
