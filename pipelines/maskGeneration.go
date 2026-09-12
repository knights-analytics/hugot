package pipelines

import (
	"context"
	"errors"
	"image"

	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/util/imageutil"
)

// MaskGenerationPipeline generates source-sized masks from image segmentation
// models. Its output contract is intentionally the same as image segmentation.
type MaskGenerationPipeline struct {
	*ImageSegmentationPipeline
}

type (
	MaskGenerationOutput = ImageSegmentationOutput
	MaskGenerationResult = ImageSegmentationResult
	MaskGenerationMask   = ImageSegmentationSegment
)

func NewMaskGenerationPipeline(ctx context.Context, config backends.PipelineConfig[*MaskGenerationPipeline], model *backends.Model) (*MaskGenerationPipeline, error) {
	baseConfig := backends.PipelineConfig[*ImageSegmentationPipeline]{
		Name: config.Name, ModelPath: config.ModelPath, OnnxFilename: config.OnnxFilename,
	}
	p := &MaskGenerationPipeline{ImageSegmentationPipeline: &ImageSegmentationPipeline{BasePipeline: backends.NewBasePipeline(ctx, baseConfig, model), ScoreThreshold: 0.5}}
	for _, option := range config.Options {
		if err := option(p); err != nil {
			return nil, err
		}
	}
	if p.imageFormat == "" {
		format, err := backends.DetectImageTensorFormat(model)
		if err != nil {
			return nil, err
		}
		p.imageFormat = format
	}
	if len(p.normalizeSteps) == 0 {
		p.normalizeSteps = []imageutil.NormalizationStep{
			imageutil.RescaleStep(), imageutil.ImagenetPixelNormalizationStep(),
		}
	}
	if err := p.Validate(); err != nil {
		return nil, err
	}
	return p, nil
}

func (p *MaskGenerationPipeline) IsGenerative() bool { return false }

func (p *MaskGenerationPipeline) Run(ctx context.Context, inputs []string) (backends.PipelineBatchOutput, error) {
	return p.RunPipeline(ctx, inputs)
}

func (p *MaskGenerationPipeline) RunPipeline(ctx context.Context, inputs []string) (*MaskGenerationOutput, error) {
	if len(inputs) == 0 {
		return nil, errors.New("mask generation requires at least one image")
	}
	return p.ImageSegmentationPipeline.RunPipeline(ctx, inputs)
}

func (p *MaskGenerationPipeline) RunWithImages(ctx context.Context, images []image.Image) (*MaskGenerationOutput, error) {
	if len(images) == 0 {
		return nil, errors.New("mask generation requires at least one image")
	}
	return p.ImageSegmentationPipeline.RunWithImages(ctx, images)
}
