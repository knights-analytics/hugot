package pipelines

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"os"
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
}

type audioTensorCreator interface {
	CreateAudioTensors(*backends.PipelineBatch, *backends.Model, [][]float32) error
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
	pipeline := &AudioClassificationPipeline{
		BasePipeline: backends.NewBasePipeline(ctx, config, model),
		IDLabelMap:   model.IDLabelMap,
		TopK:         5,
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

func (p *AudioClassificationPipeline) GetModel() *backends.Model { return p.Model }

func (p *AudioClassificationPipeline) GetMetadata() backends.PipelineMetadata {
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

func (p *AudioClassificationPipeline) RunWaveforms(ctx context.Context, inputs []AudioWaveform) (*AudioClassificationOutput, error) {
	samples := make([][]float32, len(inputs))
	for i, input := range inputs {
		samples[i] = input.Samples
	}
	return p.RunWithAudio(ctx, samples)
}

// RunWithAudio executes the pipeline on mono waveform samples.
func (p *AudioClassificationPipeline) RunWithAudio(ctx context.Context, inputs [][]float32) (*AudioClassificationOutput, error) {
	return backends.RunPipeline(ctx, len(inputs), func(batch *backends.PipelineBatch) error {
		return p.preprocess(batch, inputs)
	}, p.forward, p.postprocess)
}

func loadWAV(path string) (AudioWaveform, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return AudioWaveform{}, err
	}
	if len(data) < 44 || string(data[:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return AudioWaveform{}, errors.New("unsupported audio file: expected RIFF/WAVE")
	}
	var format, channels, bits uint16
	var sampleRate uint32
	var audio []byte
	for offset := 12; offset+8 <= len(data); {
		size := int(binary.LittleEndian.Uint32(data[offset+4 : offset+8]))
		start, end := offset+8, offset+8+size
		if end > len(data) {
			return AudioWaveform{}, errors.New("invalid WAV chunk size")
		}
		switch string(data[offset : offset+4]) {
		case "fmt ":
			if size < 16 {
				return AudioWaveform{}, errors.New("invalid WAV format chunk")
			}
			format = binary.LittleEndian.Uint16(data[start : start+2])
			channels = binary.LittleEndian.Uint16(data[start+2 : start+4])
			sampleRate = binary.LittleEndian.Uint32(data[start+4 : start+8])
			bits = binary.LittleEndian.Uint16(data[start+14 : start+16])
		case "data":
			audio = data[start:end]
		}
		offset = end + size%2
	}
	if format != 1 || channels != 1 || len(audio) == 0 {
		return AudioWaveform{}, errors.New("unsupported WAV layout: only mono PCM is supported")
	}
	bytesPerSample := int(bits / 8)
	if bytesPerSample != 1 && bytesPerSample != 2 && bytesPerSample != 3 && bytesPerSample != 4 {
		return AudioWaveform{}, fmt.Errorf("unsupported PCM bit depth %d", bits)
	}
	samples := make([]float32, len(audio)/bytesPerSample)
	for i := range samples {
		chunk := audio[i*bytesPerSample : (i+1)*bytesPerSample]
		switch bytesPerSample {
		case 1:
			samples[i] = (float32(chunk[0]) - 128) / 128
		case 2:
			value := int32(binary.LittleEndian.Uint16(chunk))
			if value >= 1<<15 {
				value -= 1 << 16
			}
			samples[i] = float32(value) / 32768
		case 3:
			value := int32(chunk[0]) | int32(chunk[1])<<8 | int32(chunk[2])<<16
			if value&0x800000 != 0 {
				value |= ^0xffffff
			}
			samples[i] = float32(value) / 8388608
		case 4:
			value := int64(binary.LittleEndian.Uint32(chunk))
			if value >= 1<<31 {
				value -= 1 << 32
			}
			samples[i] = float32(value) / 2147483648
		}
	}
	return AudioWaveform{Samples: samples, SampleRate: int(sampleRate)}, nil
}
