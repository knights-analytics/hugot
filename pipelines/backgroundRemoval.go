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

type BackgroundRemovalPipeline struct {
	*backends.BasePipeline
	imageFormat string
	OutputName  string
	preprocess  []imageutil.PreprocessStep
	normalize   []imageutil.NormalizationStep
}

type BackgroundRemovalResult struct {
	Width, Height int
	Mask          [][]float32
}

type BackgroundRemovalOutput struct{ Results []BackgroundRemovalResult }

func (o *BackgroundRemovalOutput) GetOutput() []any {
	out := make([]any, len(o.Results))
	for i := range o.Results {
		out[i] = o.Results[i]
	}
	return out
}

func WithBackgroundRemovalOutput(name string) backends.PipelineOption[*BackgroundRemovalPipeline] {
	return func(p *BackgroundRemovalPipeline) error { p.OutputName = name; return nil }
}

func NewBackgroundRemovalPipeline(ctx context.Context, config backends.PipelineConfig[*BackgroundRemovalPipeline], model *backends.Model) (*BackgroundRemovalPipeline, error) {
	p := &BackgroundRemovalPipeline{BasePipeline: backends.NewBasePipeline(ctx, config, model), normalize: []imageutil.NormalizationStep{imageutil.RescaleStep(), imageutil.ImagenetPixelNormalizationStep()}}
	for _, option := range config.Options {
		if err := option(p); err != nil {
			return nil, err
		}
	}
	format, err := backends.DetectImageTensorFormat(model)
	if err != nil {
		return nil, err
	}
	p.imageFormat = format
	if err := p.Validate(); err != nil {
		return nil, err
	}
	return p, nil
}

func (p *BackgroundRemovalPipeline) IsGenerative() bool        { return false }
func (p *BackgroundRemovalPipeline) GetModel() *backends.Model { return p.Model }
func (p *BackgroundRemovalPipeline) GetMetadata() backends.PipelineMetadata {
	out := make([]backends.OutputInfo, len(p.Model.OutputsMeta))
	for i, v := range p.Model.OutputsMeta {
		out[i] = backends.OutputInfo{Name: v.Name, Dimensions: v.Dimensions}
	}
	return backends.PipelineMetadata{OutputsInfo: out}
}

func (p *BackgroundRemovalPipeline) GetStatistics() backends.PipelineStatistics {
	s := backends.PipelineStatistics{}
	s.ComputeOnnxStatistics(p.ONNXTimings)
	return s
}

func (p *BackgroundRemovalPipeline) Validate() error {
	var errs []error
	if len(p.Model.InputsMeta) == 0 || len(p.Model.InputsMeta[0].Dimensions) != 4 {
		errs = append(errs, errors.New("background removal requires a four-dimensional image input"))
	}
	if p.OutputName == "" {
		for _, out := range p.Model.OutputsMeta {
			n := strings.ToLower(out.Name)
			if strings.Contains(n, "mask") || strings.Contains(n, "alpha") || strings.Contains(n, "matte") {
				p.OutputName = out.Name
				break
			}
		}
		if p.OutputName == "" && len(p.Model.OutputsMeta) == 1 {
			p.OutputName = p.Model.OutputsMeta[0].Name
		}
	}
	if p.OutputName == "" {
		errs = append(errs, errors.New("could not infer background-removal mask output; set WithBackgroundRemovalOutput"))
	}
	return errors.Join(errs...)
}

func (p *BackgroundRemovalPipeline) preprocessBatch(batch *backends.PipelineBatch, images []image.Image) error {
	if len(images) == 0 {
		return errors.New("background removal requires at least one image")
	}
	sizes := make([]image.Point, len(images))
	for i, img := range images {
		sizes[i] = image.Point{X: img.Bounds().Dx(), Y: img.Bounds().Dy()}
	}
	batch.InputMetadata = sizes
	processed, err := backends.PreprocessImages(p.imageFormat, images, p.preprocess, p.normalize)
	if err != nil {
		return err
	}
	return backends.CreateImageTensors(batch, p.Model, processed)
}

func (p *BackgroundRemovalPipeline) forward(ctx context.Context, batch *backends.PipelineBatch) error {
	start := time.Now()
	if err := backends.RunSessionOnBatch(ctx, batch, p.BasePipeline); err != nil {
		return err
	}
	atomic.AddUint64(&p.ONNXTimings.NumCalls, 1)
	atomic.AddUint64(&p.ONNXTimings.TotalNS, safeconv.DurationToU64(time.Since(start)))
	return nil
}

func (p *BackgroundRemovalPipeline) postprocess(batch *backends.PipelineBatch) (*BackgroundRemovalOutput, error) {
	idx := -1
	for i, o := range p.Model.OutputsMeta {
		if o.Name == p.OutputName {
			idx = i
			break
		}
	}
	if idx < 0 || idx >= len(batch.OutputValues) {
		return nil, fmt.Errorf("background-removal output %q was not returned", p.OutputName)
	}
	sizes, ok := batch.InputMetadata.([]image.Point)
	if !ok {
		return nil, errors.New("background-removal source dimensions unavailable")
	}
	results := make([]BackgroundRemovalResult, len(sizes))
	raw := batch.OutputValues[idx]
	for i, size := range sizes {
		var mask [][]float32
		switch v := raw.(type) {
		case [][][]float32:
			if i >= len(v) {
				return nil, errors.New("background-removal output batch mismatch")
			}
			mask = v[i]
		case [][][][]float32:
			if i >= len(v) || len(v[i]) == 0 {
				return nil, errors.New("background-removal output is empty")
			}
			mask = v[i][0]
		default:
			return nil, fmt.Errorf("unsupported background-removal output type %T", raw)
		}
		results[i] = BackgroundRemovalResult{Width: size.X, Height: size.Y, Mask: resizeFloatMask(mask, size.X, size.Y)}
	}
	return &BackgroundRemovalOutput{Results: results}, nil
}

func resizeFloatMask(src [][]float32, w, h int) [][]float32 {
	out := make([][]float32, h)
	if h == 0 || w == 0 || len(src) == 0 || len(src[0]) == 0 {
		return out
	}
	for y := range out {
		out[y] = make([]float32, w)
		sy := y * len(src) / h
		for x := range out[y] {
			out[y][x] = src[sy][x*len(src[0])/w]
		}
	}
	return out
}

func (p *BackgroundRemovalPipeline) Run(ctx context.Context, inputs []string) (backends.PipelineBatchOutput, error) {
	return p.RunPipeline(ctx, inputs)
}

func (p *BackgroundRemovalPipeline) RunPipeline(ctx context.Context, inputs []string) (*BackgroundRemovalOutput, error) {
	return backends.RunPipeline(ctx, len(inputs), func(b *backends.PipelineBatch) error {
		imgs, e := imageutil.LoadImagesFromPaths(p.SessionContext, inputs)
		if e != nil {
			return e
		}
		return p.preprocessBatch(b, imgs)
	}, p.forward, p.postprocess)
}

func (p *BackgroundRemovalPipeline) RunWithImages(ctx context.Context, inputs []image.Image) (*BackgroundRemovalOutput, error) {
	return backends.RunPipeline(ctx, len(inputs), func(b *backends.PipelineBatch) error { return p.preprocessBatch(b, inputs) }, p.forward, p.postprocess)
}
