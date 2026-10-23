package pipelines

import (
	"context"
	"errors"
	"fmt"
	"image"
	"math"
	"time"

	"github.com/knights-analytics/hugot/backends"
)

// VideoInput contains decoded frames in presentation order. Timestamps, when
// present, must have one entry per frame. Sampling uses frame indices, as in
// the v5.20.2 VideoMAE image-processor path, even for variable frame rates.
type VideoInput struct {
	Frames     []image.Image
	Timestamps []time.Duration
}

// VideoDecoder decodes at most maxFrames frames from the start of a video.
// Implementations must honor cancellation and return frames in presentation order.
type VideoDecoder interface {
	Decode(context.Context, string, int) (VideoInput, error)
}

type VideoClassificationPipeline struct {
	*backends.BasePipeline
	NumFrames         int
	FrameSamplingRate int
	TopK              int
	FunctionToApply   string
	Layout            string
	Decoder           VideoDecoder
	processor         visionProcessor
}

type VideoClassificationConfig = backends.PipelineConfig[*VideoClassificationPipeline]
type VideoClassificationOption = backends.PipelineOption[*VideoClassificationPipeline]
type VideoClassificationOutput = ImageClassificationOutput

func WithVideoDecoder(decoder VideoDecoder) VideoClassificationOption {
	return func(p *VideoClassificationPipeline) error { p.Decoder = decoder; return nil }
}
func WithVideoSampling(numFrames, rate int) VideoClassificationOption {
	return func(p *VideoClassificationPipeline) error {
		p.NumFrames, p.FrameSamplingRate = numFrames, rate
		return nil
	}
}
func WithVideoClassificationTopK(k int) VideoClassificationOption {
	return func(p *VideoClassificationPipeline) error { p.TopK = k; return nil }
}
func WithVideoClassificationFunction(function string) VideoClassificationOption {
	return func(p *VideoClassificationPipeline) error { p.FunctionToApply = function; return nil }
}

// WithVideoTensorLayout explicitly selects BTCHW (VideoMAE) or BCTHW.
func WithVideoTensorLayout(layout string) VideoClassificationOption {
	return func(p *VideoClassificationPipeline) error { p.Layout = layout; return nil }
}

func NewVideoClassificationPipeline(ctx context.Context, config VideoClassificationConfig, model *backends.Model) (*VideoClassificationPipeline, error) {
	if model == nil {
		return nil, errors.New("video classification requires a VideoMAE ONNX model")
	}
	p := &VideoClassificationPipeline{BasePipeline: backends.NewBasePipeline(ctx, config, model), FrameSamplingRate: 1, TopK: 5, FunctionToApply: "softmax", Layout: "BTCHW"}
	var cfg struct {
		ModelType string `json:"model_type"`
		NumFrames int    `json:"num_frames"`
	}
	if err := readNativeConfig(ctx, model.Path, "config.json", &cfg); err != nil {
		return nil, err
	}
	if cfg.ModelType != "videomae" {
		return nil, errors.New("video classification currently requires model_type videomae")
	}
	p.NumFrames = cfg.NumFrames
	if err := readNativeConfig(ctx, model.Path, "preprocessor_config.json", &p.processor); err != nil {
		return nil, err
	}
	if p.processor.ProcessorType != "VideoMAEImageProcessor" {
		return nil, errors.New("video classification requires the VideoMAE image processor; video-processor sampling is not supported")
	}
	for _, option := range config.Options {
		if err := option(p); err != nil {
			return nil, err
		}
	}
	return p, p.Validate()
}

func (*VideoClassificationPipeline) IsGenerative() bool          { return false }
func (p *VideoClassificationPipeline) GetModel() *backends.Model { return p.Model }
func (p *VideoClassificationPipeline) GetMetadata() backends.PipelineMetadata {
	return nativeMetadata(p.Model)
}
func (p *VideoClassificationPipeline) GetStatistics() backends.PipelineStatistics {
	return nativeStatistics(p.BasePipeline)
}
func (p *VideoClassificationPipeline) Validate() error {
	if p == nil || p.BasePipeline == nil {
		return errors.New("video classification requires a model")
	}
	if err := requireTensorNames(p.Model, []string{"pixel_values"}, []string{"logits"}); err != nil {
		return err
	}
	if p.NumFrames <= 0 || p.FrameSamplingRate <= 0 || p.NumFrames > math.MaxInt/p.FrameSamplingRate || p.TopK <= 0 {
		return errors.New("video frame count, sampling rate and top-k must be positive and fit in an int")
	}
	if p.Layout != "BTCHW" && p.Layout != "BCTHW" {
		return errors.New("video layout must be BTCHW or BCTHW")
	}
	if p.FunctionToApply != "softmax" && p.FunctionToApply != "sigmoid" && p.FunctionToApply != "none" {
		return errors.New("video function must be softmax, sigmoid or none")
	}
	w, h, err := p.processor.dimensions()
	if err != nil {
		return err
	}
	if len(p.Model.InputsMeta) != 1 {
		return errors.New("video graph must have only pixel_values as input")
	}
	want := p.shape(w, h)
	if err := tensorContract(p.Model.InputsMeta[0], want); err != nil {
		return err
	}
	return namedTensorContract(p.Model.OutputsMeta, "logits", []int64{1, -1})
}

func (p *VideoClassificationPipeline) shape(w, h int) []int64 {
	if p.Layout == "BCTHW" {
		return []int64{1, 3, int64(p.NumFrames), int64(h), int64(w)}
	}
	return []int64{1, int64(p.NumFrames), 3, int64(h), int64(w)}
}

func (p *VideoClassificationPipeline) preprocessVideo(input VideoInput) (backends.Tensor, error) {
	if len(input.Timestamps) != 0 {
		if len(input.Timestamps) != len(input.Frames) {
			return backends.Tensor{}, errors.New("video timestamps must match frame count")
		}
		for i, t := range input.Timestamps {
			if t < 0 || i > 0 && t < input.Timestamps[i-1] {
				return backends.Tensor{}, errors.New("video timestamps must be nonnegative and ordered")
			}
		}
	}
	// np.linspace(0, num_frames * sampling_rate - 1, num_frames, dtype=int64).
	end := p.NumFrames*p.FrameSamplingRate - 1
	if len(input.Frames) <= end {
		return backends.Tensor{}, fmt.Errorf("short video: need at least %d decoded frames, got %d; frames are not repeated", end+1, len(input.Frames))
	}
	w, h, err := p.processor.dimensions()
	if err != nil {
		return backends.Tensor{}, err
	}
	data := make([]float32, p.NumFrames*3*w*h)
	for t := 0; t < p.NumFrames; t++ {
		index := 0
		if p.NumFrames > 1 {
			index = int(float64(t) * float64(end) / float64(p.NumFrames-1))
		}
		pixels, err := p.processor.pixels(input.Frames[index])
		if err != nil {
			return backends.Tensor{}, fmt.Errorf("frame %d: %w", index, err)
		}
		if p.Layout == "BTCHW" {
			copy(data[t*3*w*h:], pixels)
		} else {
			for c := 0; c < 3; c++ {
				copy(data[(c*p.NumFrames+t)*w*h:], pixels[c*w*h:(c+1)*w*h])
			}
		}
	}
	return backends.Tensor{Shape: p.shape(w, h), Data: data}, nil
}

func (p *VideoClassificationPipeline) Run(ctx context.Context, inputs []string) (backends.PipelineBatchOutput, error) {
	return p.RunFiles(ctx, inputs)
}
func (p *VideoClassificationPipeline) RunFiles(ctx context.Context, paths []string) (*VideoClassificationOutput, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	if err := validatePipelineContext(ctx, p.BasePipeline); err != nil {
		return nil, err
	}
	if p.Decoder == nil {
		return nil, errors.New("video file input requires WithVideoDecoder; install FFmpeg and use FFmpegVideoDecoder, or use RunFrames")
	}
	if len(paths) == 0 {
		return nil, errors.New("video classification requires at least one video")
	}
	// Link file decoding to session cancellation as well as the caller context.
	if p.SessionContext != nil {
		merged, cancel := context.WithCancel(ctx)
		stop := context.AfterFunc(p.SessionContext, cancel)
		defer func() { stop(); cancel() }()
		ctx = merged
	}
	out := &VideoClassificationOutput{}
	for i, path := range paths {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if p.SessionContext != nil {
			if err := p.SessionContext.Err(); err != nil {
				return nil, err
			}
		}
		input, err := p.Decoder.Decode(ctx, path, p.NumFrames*p.FrameSamplingRate)
		if err != nil {
			return nil, fmt.Errorf("video %d: %w", i, err)
		}
		result, err := p.RunFrames(ctx, []VideoInput{input})
		if err != nil {
			return nil, fmt.Errorf("video %d: %w", i, err)
		}
		out.Predictions = append(out.Predictions, result.Predictions...)
	}
	return out, nil
}

// RunFrames processes videos individually so each graph call has batch size one.
func (p *VideoClassificationPipeline) RunFrames(ctx context.Context, inputs []VideoInput) (*VideoClassificationOutput, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	if err := validatePipelineContext(ctx, p.BasePipeline); err != nil {
		return nil, err
	}
	if len(inputs) == 0 {
		return nil, errors.New("video classification requires at least one video")
	}
	out := &VideoClassificationOutput{Predictions: make([][]ImageClassificationResult, len(inputs))}
	for i, input := range inputs {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		pixels, err := p.preprocessVideo(input)
		if err != nil {
			return nil, fmt.Errorf("video %d: %w", i, err)
		}
		result, err := runNativeTensors(ctx, p.BasePipeline, p.Model, map[string]backends.Tensor{"pixel_values": pixels})
		if err != nil {
			return nil, err
		}
		out.Predictions[i], err = p.classify(result["logits"])
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

func (p *VideoClassificationPipeline) classify(logits backends.Tensor) ([]ImageClassificationResult, error) {
	values, ok := logits.Data.([]float32)
	if !ok || len(logits.Shape) != 2 || logits.Shape[0] != 1 || logits.Shape[1] <= 0 || int64(len(values)) != logits.Shape[1] {
		return nil, errors.New("video logits must be FP32 [1,classes]")
	}
	scores := append([]float32(nil), values...)
	max := scores[0]
	for _, s := range scores {
		if !finite32(s) {
			return nil, errors.New("video logits must be finite")
		}
		if s > max {
			max = s
		}
	}
	var sum float64
	for i, s := range scores {
		switch p.FunctionToApply {
		case "softmax":
			scores[i] = float32(math.Exp(float64(s - max)))
			sum += float64(scores[i])
		case "sigmoid":
			scores[i] = float32(1 / (1 + math.Exp(-float64(s))))
		}
	}
	if p.FunctionToApply == "softmax" {
		for i := range scores {
			scores[i] = float32(float64(scores[i]) / sum)
		}
	}
	return getTopK(scores, p.TopK, p.Model.IDLabelMap), nil
}
