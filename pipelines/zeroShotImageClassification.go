package pipelines

import (
	"context"
	"errors"
	"fmt"
	"image"
	"sort"
	"strings"

	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/util/imageutil"
)

type ZeroShotImageClassificationPipeline struct {
	*backends.BasePipeline
	Labels []string
	TopK   int
}

type ZeroShotImageClassificationResult struct {
	Label string
	Score float32
}
type ZeroShotImageClassificationOutput struct {
	Predictions [][]ZeroShotImageClassificationResult
}

func (o *ZeroShotImageClassificationOutput) GetOutput() []any {
	out := make([]any, len(o.Predictions))
	for i, v := range o.Predictions {
		out[i] = v
	}
	return out
}

func WithImageLabels(labels []string) backends.PipelineOption[*ZeroShotImageClassificationPipeline] {
	return func(p *ZeroShotImageClassificationPipeline) error {
		p.Labels = append([]string(nil), labels...)
		return nil
	}
}

func WithImageTopK(k int) backends.PipelineOption[*ZeroShotImageClassificationPipeline] {
	return func(p *ZeroShotImageClassificationPipeline) error {
		if k < 1 {
			return errors.New("zero-shot image classification top-k must be greater than zero")
		}
		p.TopK = k
		return nil
	}
}

func NewZeroShotImageClassificationPipeline(ctx context.Context, config backends.PipelineConfig[*ZeroShotImageClassificationPipeline], model *backends.Model) (*ZeroShotImageClassificationPipeline, error) {
	p := &ZeroShotImageClassificationPipeline{BasePipeline: backends.NewBasePipeline(ctx, config, model), TopK: 5}
	for _, o := range config.Options {
		if err := o(p); err != nil {
			return nil, err
		}
	}
	if p.TopK == 0 {
		p.TopK = 5
	}
	if err := p.Validate(); err != nil {
		return nil, err
	}
	return p, nil
}
func (p *ZeroShotImageClassificationPipeline) IsGenerative() bool        { return false }
func (p *ZeroShotImageClassificationPipeline) GetModel() *backends.Model { return p.Model }
func (p *ZeroShotImageClassificationPipeline) GetMetadata() backends.PipelineMetadata {
	out := make([]backends.OutputInfo, len(p.Model.OutputsMeta))
	for i, v := range p.Model.OutputsMeta {
		out[i] = backends.OutputInfo{Name: v.Name, Dimensions: v.Dimensions}
	}
	return backends.PipelineMetadata{OutputsInfo: out}
}

func (p *ZeroShotImageClassificationPipeline) GetStatistics() backends.PipelineStatistics {
	return backends.PipelineStatistics{}
}

func (p *ZeroShotImageClassificationPipeline) Validate() error {
	var errs []error
	if p.Model == nil {
		return errors.New("zero-shot image classification requires a model")
	}
	if len(p.Labels) == 0 {
		errs = append(errs, errors.New("zero-shot image classification requires at least one candidate label"))
	}
	if len(p.Model.InputsMeta) == 0 || len(p.Model.InputsMeta[0].Dimensions) != 4 {
		errs = append(errs, errors.New("zero-shot image classification requires a four-dimensional image input"))
	}
	if p.Model.Tokenizer == nil {
		errs = append(errs, errors.New("zero-shot image classification requires a tokenizer for candidate labels"))
	}
	return errors.Join(errs...)
}

// Run is intentionally unavailable because []string cannot express paired image/text inputs.
func (p *ZeroShotImageClassificationPipeline) Run(context.Context, []string) (backends.PipelineBatchOutput, error) {
	return nil, errors.New("zero-shot image classification requires paired image/text inputs; use RunWithImageText")
}

func (p *ZeroShotImageClassificationPipeline) RunWithImageText(ctx context.Context, inputs []backends.ImageTextInput) (*ZeroShotImageClassificationOutput, error) {
	if len(inputs) == 0 {
		return nil, errors.New("zero-shot image classification requires at least one input")
	}
	images := make([]image.Image, len(inputs))
	texts := make([]string, len(inputs))
	for i, input := range inputs {
		if err := input.Validate(); err != nil {
			return nil, fmt.Errorf("input %d: %w", i, err)
		}
		if input.Image == nil {
			loaded, err := imageutil.LoadImagesFromPaths(ctx, []string{input.ImagePath})
			if err != nil {
				return nil, fmt.Errorf("failed to load image %q: %w", input.ImagePath, err)
			}
			images[i] = loaded[0]
		} else {
			images[i] = input.Image
		}
		texts[i] = input.Text
	}
	return p.run(ctx, images, texts)
}

func (p *ZeroShotImageClassificationPipeline) RunWithImagesAndLabels(ctx context.Context, images []image.Image, labels []string) (*ZeroShotImageClassificationOutput, error) {
	if len(labels) == 0 {
		labels = p.Labels
	}
	if len(images) == 0 || len(labels) == 0 {
		return nil, errors.New("zero-shot image classification requires images and labels")
	}
	pairedImages := make([]image.Image, 0, len(images)*len(labels))
	pairedTexts := make([]string, 0, len(images)*len(labels))
	for _, img := range images {
		if img == nil {
			return nil, errors.New("zero-shot image classification received a nil image")
		}
		for _, label := range labels {
			if strings.TrimSpace(label) == "" {
				return nil, errors.New("zero-shot image classification received an empty label")
			}
			pairedImages = append(pairedImages, img)
			pairedTexts = append(pairedTexts, label)
		}
	}
	flat, err := p.run(ctx, pairedImages, pairedTexts)
	if err != nil {
		return nil, err
	}
	result := &ZeroShotImageClassificationOutput{Predictions: make([][]ZeroShotImageClassificationResult, len(images))}
	for i := range images {
		for _, prediction := range flat.Predictions[i*len(labels) : (i+1)*len(labels)] {
			result.Predictions[i] = append(result.Predictions[i], prediction...)
		}
		sort.Slice(result.Predictions[i], func(a, b int) bool { return result.Predictions[i][a].Score > result.Predictions[i][b].Score })
		if p.TopK > 0 && len(result.Predictions[i]) > p.TopK {
			result.Predictions[i] = result.Predictions[i][:p.TopK]
		}
	}
	return result, nil
}

func (p *ZeroShotImageClassificationPipeline) run(ctx context.Context, images []image.Image, texts []string) (*ZeroShotImageClassificationOutput, error) {
	return backends.RunPipeline(ctx, len(images), func(batch *backends.PipelineBatch) error {
		format, err := backends.DetectImageTensorFormat(p.Model)
		if err != nil {
			return err
		}
		processed, err := backends.PreprocessImages(format, images, nil, nil)
		if err != nil {
			return err
		}
		backends.TokenizeInputs(batch, p.Model.Tokenizer, texts)
		return p.Model.Backend.CreateImageTextTensors(batch, p.Model, processed, batch.Input)
	}, func(ctx context.Context, batch *backends.PipelineBatch) error {
		return backends.RunSessionOnBatch(ctx, batch, p.BasePipeline)
	}, func(batch *backends.PipelineBatch) (*ZeroShotImageClassificationOutput, error) {
		if len(batch.OutputValues) == 0 {
			return nil, errors.New("zero-shot image classification produced no outputs")
		}
		predictions := make([][]ZeroShotImageClassificationResult, len(texts))
		switch values := batch.OutputValues[0].(type) {
		case [][]float32:
			if len(values) != len(texts) {
				return nil, fmt.Errorf("output batch size %d does not match input batch size %d", len(values), len(texts))
			}
			for i, row := range values {
				score := float32(0)
				if len(row) > 0 {
					score = row[0]
				}
				predictions[i] = []ZeroShotImageClassificationResult{{Label: texts[i], Score: score}}
			}
		case []float32:
			if len(values) != len(texts) {
				return nil, fmt.Errorf("output batch size %d does not match input batch size %d", len(values), len(texts))
			}
			for i, score := range values {
				predictions[i] = []ZeroShotImageClassificationResult{{Label: texts[i], Score: score}}
			}
		default:
			return nil, fmt.Errorf("unsupported zero-shot image classification output type %T", batch.OutputValues[0])
		}
		return &ZeroShotImageClassificationOutput{Predictions: predictions}, nil
	})
}
