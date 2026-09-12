package pipelines

import (
	"context"
	"errors"
	"fmt"
	"image"
	"math"
	"strings"
	"sync/atomic"
	"time"

	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/util/imageutil"
	"github.com/knights-analytics/hugot/util/safeconv"
)

// ImageSegmentationPipeline decodes semantic segmentation logits into masks in
// the coordinate space of the source image.
type ImageSegmentationPipeline struct {
	*backends.BasePipeline
	IDLabelMap      map[int]string
	imageFormat     string
	LogitsOutput    string
	ScoreThreshold  float32
	preprocessSteps []imageutil.PreprocessStep
	normalizeSteps  []imageutil.NormalizationStep
}

type ImageSegmentationSegment struct {
	Label string
	Class int
	Score float32
	Mask  [][]bool
}

type ImageSegmentationResult struct {
	Width    int
	Height   int
	Segments []ImageSegmentationSegment
}

type ImageSegmentationOutput struct {
	Results []ImageSegmentationResult
}

func (o *ImageSegmentationOutput) GetOutput() []any {
	out := make([]any, len(o.Results))
	for i, result := range o.Results {
		out[i] = result
	}
	return out
}

func WithSegmentationLogitsOutput(name string) backends.PipelineOption[*ImageSegmentationPipeline] {
	return func(pipeline *ImageSegmentationPipeline) error {
		pipeline.LogitsOutput = name
		return nil
	}
}

func WithSegmentationScoreThreshold(threshold float32) backends.PipelineOption[*ImageSegmentationPipeline] {
	return func(pipeline *ImageSegmentationPipeline) error {
		pipeline.ScoreThreshold = threshold
		return nil
	}
}

func NewImageSegmentationPipeline(ctx context.Context, config backends.PipelineConfig[*ImageSegmentationPipeline], model *backends.Model) (*ImageSegmentationPipeline, error) {
	pipeline := &ImageSegmentationPipeline{
		BasePipeline:   backends.NewBasePipeline(ctx, config, model),
		IDLabelMap:     model.IDLabelMap,
		ScoreThreshold: 0.5,
	}
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
		}
	}
	if err := pipeline.Validate(); err != nil {
		return nil, err
	}
	return pipeline, nil
}

func (p *ImageSegmentationPipeline) IsGenerative() bool        { return false }
func (p *ImageSegmentationPipeline) GetModel() *backends.Model { return p.Model }
func (p *ImageSegmentationPipeline) GetMetadata() backends.PipelineMetadata {
	outputs := make([]backends.OutputInfo, len(p.Model.OutputsMeta))
	for i, output := range p.Model.OutputsMeta {
		outputs[i] = backends.OutputInfo{Name: output.Name, Dimensions: output.Dimensions}
	}
	return backends.PipelineMetadata{OutputsInfo: outputs}
}

func (p *ImageSegmentationPipeline) GetStatistics() backends.PipelineStatistics {
	statistics := backends.PipelineStatistics{}
	statistics.ComputeOnnxStatistics(p.ONNXTimings)
	return statistics
}

func (p *ImageSegmentationPipeline) Validate() error {
	var validationErrors []error
	if p.ScoreThreshold < 0 || p.ScoreThreshold > 1 {
		validationErrors = append(validationErrors, fmt.Errorf("segmentation threshold must be between 0 and 1, got %f", p.ScoreThreshold))
	}
	for _, input := range p.Model.InputsMeta {
		if len(input.Dimensions) != 4 {
			validationErrors = append(validationErrors, fmt.Errorf("image input %s must have four dimensions", input.Name))
		}
	}
	if p.LogitsOutput == "" {
		for _, output := range p.Model.OutputsMeta {
			lower := strings.ToLower(output.Name)
			if strings.Contains(lower, "logit") || strings.Contains(lower, "mask") || strings.Contains(lower, "segment") {
				p.LogitsOutput = output.Name
				break
			}
		}
	}
	if p.LogitsOutput == "" {
		validationErrors = append(validationErrors, errors.New("could not infer segmentation logits output; set WithSegmentationLogitsOutput"))
	}
	return errors.Join(validationErrors...)
}

func (p *ImageSegmentationPipeline) addPreprocessSteps(steps ...imageutil.PreprocessStep) {
	p.preprocessSteps = append(p.preprocessSteps, steps...)
}

func (p *ImageSegmentationPipeline) addNormalizationSteps(steps ...imageutil.NormalizationStep) {
	p.normalizeSteps = append(p.normalizeSteps, steps...)
}
func (p *ImageSegmentationPipeline) setImageFormat(format string) { p.imageFormat = format }

func (p *ImageSegmentationPipeline) preprocess(batch *backends.PipelineBatch, images []image.Image) error {
	if len(images) == 0 {
		return errors.New("image segmentation requires at least one image")
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

func (p *ImageSegmentationPipeline) forward(ctx context.Context, batch *backends.PipelineBatch) error {
	start := time.Now()
	if err := backends.RunSessionOnBatch(ctx, batch, p.BasePipeline); err != nil {
		return err
	}
	atomic.AddUint64(&p.ONNXTimings.NumCalls, 1)
	atomic.AddUint64(&p.ONNXTimings.TotalNS, safeconv.DurationToU64(time.Since(start)))
	return nil
}

func (p *ImageSegmentationPipeline) postprocess(batch *backends.PipelineBatch) (*ImageSegmentationOutput, error) {
	outputIndex := -1
	for i, output := range p.Model.OutputsMeta {
		if output.Name == p.LogitsOutput {
			outputIndex = i
			break
		}
	}
	if outputIndex < 0 || outputIndex >= len(batch.OutputValues) {
		return nil, fmt.Errorf("segmentation output %q was not returned by the model", p.LogitsOutput)
	}
	logits, ok := batch.OutputValues[outputIndex].([][][][]float32)
	if !ok {
		return nil, fmt.Errorf("segmentation output type %T is unsupported; expected [batch][class][height][width] logits", batch.OutputValues[outputIndex])
	}
	sizes, ok := batch.InputMetadata.([]image.Point)
	if !ok || len(sizes) != len(logits) {
		return nil, errors.New("segmentation source image dimensions are unavailable")
	}
	results := make([]ImageSegmentationResult, len(logits))
	for batchIndex, imageLogits := range logits {
		if len(imageLogits) == 0 || len(imageLogits[0]) == 0 || len(imageLogits[0][0]) == 0 {
			return nil, fmt.Errorf("segmentation output for image %d is empty", batchIndex)
		}
		height, width := len(imageLogits[0]), len(imageLogits[0][0])
		masks := make([][][]bool, len(imageLogits))
		scores := make([]float32, len(imageLogits))
		for classIndex := range imageLogits {
			masks[classIndex] = make([][]bool, height)
			for y := range height {
				masks[classIndex][y] = make([]bool, width)
			}
		}
		for y := range height {
			for x := range width {
				bestClass := 0
				bestLogit := imageLogits[0][y][x]
				for classIndex := 1; classIndex < len(imageLogits); classIndex++ {
					if imageLogits[classIndex][y][x] > bestLogit {
						bestClass, bestLogit = classIndex, imageLogits[classIndex][y][x]
					}
				}
				maxLogit := bestLogit
				var denominator float64
				for _, classLogits := range imageLogits {
					denominator += math.Exp(float64(classLogits[y][x] - maxLogit))
				}
				confidence := float32(1 / denominator)
				scores[bestClass] += confidence
				if confidence >= p.ScoreThreshold {
					masks[bestClass][y][x] = true
				}
			}
		}
		segments := make([]ImageSegmentationSegment, 0, len(masks))
		for classIndex, mask := range masks {
			count := 0
			for _, row := range mask {
				for _, value := range row {
					if value {
						count++
					}
				}
			}
			if count == 0 {
				continue
			}
			label := fmt.Sprintf("class_%d", classIndex)
			if mapped, found := p.IDLabelMap[classIndex]; found {
				label = mapped
			}
			segments = append(segments, ImageSegmentationSegment{
				Label: label, Class: classIndex, Score: scores[classIndex] / float32(width*height),
				Mask: resizeMask(mask, sizes[batchIndex].X, sizes[batchIndex].Y),
			})
		}
		results[batchIndex] = ImageSegmentationResult{Width: sizes[batchIndex].X, Height: sizes[batchIndex].Y, Segments: segments}
	}
	return &ImageSegmentationOutput{Results: results}, nil
}

func resizeMask(mask [][]bool, width, height int) [][]bool {
	resized := make([][]bool, height)
	if height == 0 || width == 0 {
		return resized
	}
	for y := range resized {
		resized[y] = make([]bool, width)
		sourceY := y * len(mask) / height
		for x := range resized[y] {
			sourceX := x * len(mask[0]) / width
			resized[y][x] = mask[sourceY][sourceX]
		}
	}
	return resized
}

func (p *ImageSegmentationPipeline) Run(ctx context.Context, inputs []string) (backends.PipelineBatchOutput, error) {
	return p.RunPipeline(ctx, inputs)
}

func (p *ImageSegmentationPipeline) RunPipeline(ctx context.Context, inputs []string) (*ImageSegmentationOutput, error) {
	return backends.RunPipeline(ctx, len(inputs), func(batch *backends.PipelineBatch) error {
		images, err := imageutil.LoadImagesFromPaths(p.SessionContext, inputs)
		if err != nil {
			return fmt.Errorf("failed to load images: %w", err)
		}
		return p.preprocess(batch, images)
	}, p.forward, p.postprocess)
}

func (p *ImageSegmentationPipeline) RunWithImages(ctx context.Context, images []image.Image) (*ImageSegmentationOutput, error) {
	return backends.RunPipeline(ctx, len(images), func(batch *backends.PipelineBatch) error { return p.preprocess(batch, images) }, p.forward, p.postprocess)
}
