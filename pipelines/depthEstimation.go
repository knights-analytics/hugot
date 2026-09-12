package pipelines

import (
	"context"
	"errors"
	"fmt"
	"image"
	"strings"
	"sync/atomic"
	"time"

	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/util/imageutil"
	"github.com/knights-analytics/hugot/util/safeconv"
)

// DepthEstimationPipeline estimates one depth value for each source-image pixel.
type DepthEstimationPipeline struct {
	*backends.BasePipeline
	imageFormat     string
	preprocessSteps []imageutil.PreprocessStep
	normalizeSteps  []imageutil.NormalizationStep
	DepthOutput     string
}

type (
	DepthEstimationConfig = backends.PipelineConfig[*DepthEstimationPipeline]
	DepthEstimationOption = backends.PipelineOption[*DepthEstimationPipeline]
)

type DepthEstimationResult struct {
	Width    int
	Height   int
	DepthMap [][]float32
}

type DepthEstimationOutput struct {
	Results []DepthEstimationResult
}

func (o *DepthEstimationOutput) GetOutput() []any {
	out := make([]any, len(o.Results))
	for i, result := range o.Results {
		out[i] = result
	}
	return out
}

// WithDepthOutput selects the model output containing predicted depth values.
func WithDepthOutput(name string) DepthEstimationOption {
	return func(pipeline *DepthEstimationPipeline) error {
		pipeline.DepthOutput = name
		return nil
	}
}

// WithDepthEstimationOutput is an explicit-name alias for WithDepthOutput.
func WithDepthEstimationOutput(name string) DepthEstimationOption { return WithDepthOutput(name) }

func NewDepthEstimationPipeline(ctx context.Context, config DepthEstimationConfig, model *backends.Model) (*DepthEstimationPipeline, error) {
	pipeline := &DepthEstimationPipeline{BasePipeline: backends.NewBasePipeline(ctx, config, model)}
	for _, option := range config.Options {
		if err := option(pipeline); err != nil {
			return nil, err
		}
	}
	if pipeline.imageFormat == "" {
		format, err := backends.DetectImageTensorFormat(model)
		if err != nil {
			return nil, err
		}
		pipeline.imageFormat = format
	}
	if len(pipeline.normalizeSteps) == 0 {
		pipeline.normalizeSteps = []imageutil.NormalizationStep{
			imageutil.RescaleStep(), imageutil.ImagenetPixelNormalizationStep(),
		}
	}
	if len(pipeline.preprocessSteps) == 0 && len(model.InputsMeta) > 0 && len(model.InputsMeta[0].Dimensions) == 4 {
		dimensions := model.InputsMeta[0].Dimensions
		if dimensions[2] > 0 && dimensions[3] > 0 {
			pipeline.preprocessSteps = []imageutil.PreprocessStep{
				imageutil.ResizeStep(int(dimensions[2])),
				imageutil.CenterCropStep(int(dimensions[3]), int(dimensions[2])),
			}
		} else if model.ImageSize > 0 {
			pipeline.preprocessSteps = []imageutil.PreprocessStep{
				imageutil.ResizeStep(model.ImageSize),
				imageutil.CenterCropStep(model.ImageSize, model.ImageSize),
			}
		}
	}
	if err := pipeline.Validate(); err != nil {
		return nil, err
	}
	return pipeline, nil
}

func (p *DepthEstimationPipeline) IsGenerative() bool        { return false }
func (p *DepthEstimationPipeline) GetModel() *backends.Model { return p.Model }
func (p *DepthEstimationPipeline) GetMetadata() backends.PipelineMetadata {
	outputs := make([]backends.OutputInfo, len(p.Model.OutputsMeta))
	for i, output := range p.Model.OutputsMeta {
		outputs[i] = backends.OutputInfo{Name: output.Name, Dimensions: output.Dimensions}
	}
	return backends.PipelineMetadata{OutputsInfo: outputs}
}

func (p *DepthEstimationPipeline) GetStatistics() backends.PipelineStatistics {
	statistics := backends.PipelineStatistics{}
	statistics.ComputeOnnxStatistics(p.ONNXTimings)
	return statistics
}

func (p *DepthEstimationPipeline) Validate() error {
	if p.Model == nil {
		return errors.New("depth estimation pipeline requires a model")
	}
	var validationErrors []error
	if len(p.Model.InputsMeta) == 0 {
		validationErrors = append(validationErrors, errors.New("depth estimation pipeline requires model inputs"))
	}
	for _, input := range p.Model.InputsMeta {
		if len(input.Dimensions) != 4 {
			validationErrors = append(validationErrors, fmt.Errorf("image input %s must have four dimensions", input.Name))
		}
	}
	if len(p.Model.OutputsMeta) == 0 {
		validationErrors = append(validationErrors, errors.New("depth estimation pipeline requires model outputs"))
	}
	if p.DepthOutput == "" {
		for _, output := range p.Model.OutputsMeta {
			name := strings.ToLower(output.Name)
			if strings.Contains(name, "depth") {
				p.DepthOutput = output.Name
				break
			}
		}
	}
	if p.DepthOutput == "" {
		validationErrors = append(validationErrors, errors.New("could not infer depth output; set WithDepthOutput"))
	}
	return errors.Join(validationErrors...)
}

func (p *DepthEstimationPipeline) addPreprocessSteps(steps ...imageutil.PreprocessStep) {
	p.preprocessSteps = append(p.preprocessSteps, steps...)
}

func (p *DepthEstimationPipeline) addNormalizationSteps(steps ...imageutil.NormalizationStep) {
	p.normalizeSteps = append(p.normalizeSteps, steps...)
}

func (p *DepthEstimationPipeline) setImageFormat(format string) { p.imageFormat = format }

func (p *DepthEstimationPipeline) preprocess(batch *backends.PipelineBatch, images []image.Image) error {
	if len(images) == 0 {
		return errors.New("depth estimation requires at least one image")
	}
	sizes := make([]image.Point, len(images))
	for i, img := range images {
		sizes[i] = image.Point{X: img.Bounds().Dx(), Y: img.Bounds().Dy()}
	}
	batch.InputMetadata = sizes
	processed, err := backends.PreprocessImages(p.imageFormat, images, p.preprocessSteps, p.normalizeSteps)
	if err != nil {
		return fmt.Errorf("failed to preprocess images: %w", err)
	}
	return backends.CreateImageTensors(batch, p.Model, processed)
}

func (p *DepthEstimationPipeline) forward(ctx context.Context, batch *backends.PipelineBatch) error {
	start := time.Now()
	if err := backends.RunSessionOnBatch(ctx, batch, p.BasePipeline); err != nil {
		return err
	}
	atomic.AddUint64(&p.ONNXTimings.NumCalls, 1)
	atomic.AddUint64(&p.ONNXTimings.TotalNS, safeconv.DurationToU64(time.Since(start)))
	return nil
}

func (p *DepthEstimationPipeline) postprocess(batch *backends.PipelineBatch) (*DepthEstimationOutput, error) {
	outputIndex := -1
	for i, output := range p.Model.OutputsMeta {
		if output.Name == p.DepthOutput {
			outputIndex = i
			break
		}
	}
	if outputIndex < 0 || outputIndex >= len(batch.OutputValues) {
		return nil, fmt.Errorf("depth output %q was not returned by the model", p.DepthOutput)
	}
	depthMaps, err := decodeDepthOutput(batch.OutputValues[outputIndex])
	if err != nil {
		return nil, err
	}
	sizes, ok := batch.InputMetadata.([]image.Point)
	if !ok || len(sizes) != len(depthMaps) {
		return nil, errors.New("depth source image dimensions are unavailable")
	}
	results := make([]DepthEstimationResult, len(depthMaps))
	for i, depthMap := range depthMaps {
		if len(depthMap) == 0 || len(depthMap[0]) == 0 {
			return nil, fmt.Errorf("depth output for image %d is empty", i)
		}
		results[i] = DepthEstimationResult{
			Width: sizes[i].X, Height: sizes[i].Y,
			DepthMap: resizeDepthMap(depthMap, sizes[i].X, sizes[i].Y),
		}
	}
	return &DepthEstimationOutput{Results: results}, nil
}

func decodeDepthOutput(output any) ([][][]float32, error) {
	switch value := output.(type) {
	case [][][]float32:
		return value, nil
	case [][][][]float32:
		maps := make([][][]float32, len(value))
		for i, channels := range value {
			if len(channels) != 1 {
				return nil, fmt.Errorf("depth output has %d channels for image %d; expected one", len(channels), i)
			}
			maps[i] = channels[0]
		}
		return maps, nil
	default:
		return nil, fmt.Errorf("depth output type %T is unsupported; expected [batch][height][width] depth values", output)
	}
}

func resizeDepthMap(depthMap [][]float32, width, height int) [][]float32 {
	resized := make([][]float32, height)
	if width == 0 || height == 0 || len(depthMap) == 0 || len(depthMap[0]) == 0 {
		return resized
	}
	for y := range resized {
		resized[y] = make([]float32, width)
		sourceY := y * len(depthMap) / height
		for x := range resized[y] {
			sourceX := x * len(depthMap[0]) / width
			resized[y][x] = depthMap[sourceY][sourceX]
		}
	}
	return resized
}

func (p *DepthEstimationPipeline) Run(ctx context.Context, inputs []string) (backends.PipelineBatchOutput, error) {
	return p.RunPipeline(ctx, inputs)
}

func (p *DepthEstimationPipeline) RunPipeline(ctx context.Context, inputs []string) (*DepthEstimationOutput, error) {
	return backends.RunPipeline(ctx, len(inputs), func(batch *backends.PipelineBatch) error {
		images, err := imageutil.LoadImagesFromPaths(p.SessionContext, inputs)
		if err != nil {
			return fmt.Errorf("failed to load images: %w", err)
		}
		return p.preprocess(batch, images)
	}, p.forward, p.postprocess)
}

func (p *DepthEstimationPipeline) RunWithImages(ctx context.Context, images []image.Image) (*DepthEstimationOutput, error) {
	return backends.RunPipeline(ctx, len(images), func(batch *backends.PipelineBatch) error {
		return p.preprocess(batch, images)
	}, p.forward, p.postprocess)
}

func (p *DepthEstimationPipeline) RunWithImagePaths(ctx context.Context, paths []string) (*DepthEstimationOutput, error) {
	return p.RunPipeline(ctx, paths)
}
