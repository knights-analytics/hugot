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
	"github.com/knights-analytics/hugot/util/vectorutil"
)

// ImageFeatureExtractionPipeline returns one pooled embedding per image.
type ImageFeatureExtractionPipeline struct {
	*backends.BasePipeline
	imageFormat        string
	preprocessSteps    []imageutil.PreprocessStep
	normalizationSteps []imageutil.NormalizationStep
	OutputIndex        int
	Normalization      bool
	modelPooler        bool
}

type ImageFeatureExtractionOutput struct{ Embeddings [][]float32 }

func (o *ImageFeatureExtractionOutput) GetOutput() []any {
	out := make([]any, len(o.Embeddings))
	for i, embedding := range o.Embeddings {
		out[i] = embedding
	}
	return out
}

func WithImageEmbeddingNormalization() backends.PipelineOption[*ImageFeatureExtractionPipeline] {
	return func(pipeline *ImageFeatureExtractionPipeline) error { pipeline.Normalization = true; return nil }
}

// WithImageModelPooler selects the model's rank-two pooler_output. The default
// pools rank-three hidden states with an arithmetic mean, not the model pooler.
func WithImageModelPooler() backends.PipelineOption[*ImageFeatureExtractionPipeline] {
	return func(pipeline *ImageFeatureExtractionPipeline) error {
		pipeline.modelPooler = true
		return nil
	}
}

func NewImageFeatureExtractionPipeline(ctx context.Context, config backends.PipelineConfig[*ImageFeatureExtractionPipeline], model *backends.Model) (*ImageFeatureExtractionPipeline, error) {
	if model == nil {
		return nil, errors.New("image feature extraction pipeline requires a model")
	}
	pipeline := &ImageFeatureExtractionPipeline{BasePipeline: backends.NewBasePipeline(ctx, config, model)}
	for i, output := range model.OutputsMeta {
		if output.Name == "last_hidden_state" {
			pipeline.OutputIndex = i
			break
		}
	}
	for _, option := range config.Options {
		if err := option(pipeline); err != nil {
			return nil, err
		}
	}
	if pipeline.modelPooler {
		index, err := modelPoolerIndex(model)
		if err != nil {
			return nil, err
		}
		pipeline.OutputIndex = index
	}
	if pipeline.imageFormat == "" {
		format, err := backends.DetectImageTensorFormat(model)
		if err != nil {
			return nil, err
		}
		pipeline.imageFormat = format
	}
	if err := pipeline.Validate(); err != nil {
		return nil, err
	}
	return pipeline, nil
}

func (p *ImageFeatureExtractionPipeline) IsGenerative() bool        { return false }
func (p *ImageFeatureExtractionPipeline) GetModel() *backends.Model { return p.Model }
func (p *ImageFeatureExtractionPipeline) GetMetadata() backends.PipelineMetadata {
	if p == nil || p.Model == nil || p.OutputIndex < 0 || p.OutputIndex >= len(p.Model.OutputsMeta) {
		return backends.PipelineMetadata{}
	}
	return backends.PipelineMetadata{OutputsInfo: []backends.OutputInfo{{Name: p.Model.OutputsMeta[p.OutputIndex].Name, Dimensions: p.Model.OutputsMeta[p.OutputIndex].Dimensions}}}
}

func (p *ImageFeatureExtractionPipeline) GetStatistics() backends.PipelineStatistics {
	statistics := backends.PipelineStatistics{}
	statistics.ComputeOnnxStatistics(p.ONNXTimings)
	return statistics
}

func (p *ImageFeatureExtractionPipeline) Validate() error {
	if p == nil || p.BasePipeline == nil || p.Model == nil {
		return errors.New("image feature extraction requires a model")
	}
	if len(p.Model.InputsMeta) == 0 {
		return errors.New("image feature extraction pipeline requires model inputs")
	}
	if len(p.Model.OutputsMeta) == 0 {
		return errors.New("image feature extraction pipeline requires model outputs")
	}
	if p.OutputIndex < 0 || p.OutputIndex >= len(p.Model.OutputsMeta) {
		return errors.New("image feature output index is out of range")
	}
	output := p.Model.OutputsMeta[p.OutputIndex]
	if strings.Contains(strings.ToLower(output.Name), "logits") {
		return errors.New("image feature extraction requires base-model hidden features, not classifier logits")
	}
	if len(output.Dimensions) != 2 && len(output.Dimensions) != 3 {
		return errors.New("image features require a rank-two or rank-three output")
	}
	for _, input := range p.Model.InputsMeta {
		if len(input.Dimensions) != 4 {
			return fmt.Errorf("image input %s must have four dimensions", input.Name)
		}
	}
	return nil
}

func (p *ImageFeatureExtractionPipeline) addPreprocessSteps(steps ...imageutil.PreprocessStep) {
	p.preprocessSteps = append(p.preprocessSteps, steps...)
}

func (p *ImageFeatureExtractionPipeline) addNormalizationSteps(steps ...imageutil.NormalizationStep) {
	p.normalizationSteps = append(p.normalizationSteps, steps...)
}
func (p *ImageFeatureExtractionPipeline) setImageFormat(format string) { p.imageFormat = format }

func (p *ImageFeatureExtractionPipeline) preprocess(batch *backends.PipelineBatch, images []image.Image) error {
	processed, err := backends.PreprocessImages(p.imageFormat, images, p.preprocessSteps, p.normalizationSteps)
	if err != nil {
		return fmt.Errorf("failed to preprocess images: %w", err)
	}
	return backends.CreateImageTensors(batch, p.Model, processed)
}

func (p *ImageFeatureExtractionPipeline) forward(ctx context.Context, batch *backends.PipelineBatch) error {
	start := time.Now()
	if err := backends.RunSessionOnBatch(ctx, batch, p.BasePipeline); err != nil {
		return err
	}
	atomic.AddUint64(&p.ONNXTimings.NumCalls, 1)
	atomic.AddUint64(&p.ONNXTimings.TotalNS, safeconv.DurationToU64(time.Since(start)))
	return nil
}

func (p *ImageFeatureExtractionPipeline) postprocess(batch *backends.PipelineBatch) (*ImageFeatureExtractionOutput, error) {
	if p.OutputIndex < 0 || len(batch.OutputValues) <= p.OutputIndex {
		return nil, errors.New("image feature extraction model returned no selected output")
	}
	output := batch.OutputValues[p.OutputIndex]
	var embeddings [][]float32
	switch value := output.(type) {
	case [][]float32:
		embeddings = value
	case [][][]float32:
		embeddings = make([][]float32, len(value))
		for i, tokens := range value {
			if len(tokens) == 0 {
				return nil, fmt.Errorf("image embedding %d is empty", i)
			}
			embeddings[i] = make([]float32, len(tokens[0]))
			for _, token := range tokens {
				if len(token) != len(embeddings[i]) {
					return nil, errors.New("image feature tokens have inconsistent dimensions")
				}
				for j, component := range token {
					embeddings[i][j] += component
				}
			}
			for j := range embeddings[i] {
				embeddings[i][j] /= float32(len(tokens))
			}
		}
	default:
		return nil, fmt.Errorf("image feature extraction output type %T is not supported", output)
	}
	for _, embedding := range embeddings {
		if len(embedding) == 0 {
			return nil, errors.New("image feature vector is empty")
		}
		for _, value := range embedding {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return nil, errors.New("image feature vector is nonfinite")
			}
		}
	}
	if p.Normalization {
		embeddings = append([][]float32(nil), embeddings...)
		for i := range embeddings {
			embeddings[i] = vectorutil.Normalize(append([]float32(nil), embeddings[i]...), 2)
		}
	}
	return &ImageFeatureExtractionOutput{Embeddings: embeddings}, nil
}

func (p *ImageFeatureExtractionPipeline) Run(ctx context.Context, inputs []string) (backends.PipelineBatchOutput, error) {
	return p.RunPipeline(ctx, inputs)
}

func (p *ImageFeatureExtractionPipeline) RunPipeline(ctx context.Context, inputs []string) (*ImageFeatureExtractionOutput, error) {
	return backends.RunPipeline(ctx, len(inputs), func(batch *backends.PipelineBatch) error {
		images, err := imageutil.LoadImagesFromPaths(p.SessionContext, inputs)
		if err != nil {
			return fmt.Errorf("failed to load images: %w", err)
		}
		return p.preprocess(batch, images)
	}, p.forward, p.postprocess)
}

func (p *ImageFeatureExtractionPipeline) RunWithImages(ctx context.Context, images []image.Image) (*ImageFeatureExtractionOutput, error) {
	return backends.RunPipeline(ctx, len(images), func(batch *backends.PipelineBatch) error { return p.preprocess(batch, images) }, p.forward, p.postprocess)
}

func (p *ImageFeatureExtractionPipeline) postprocessRaw(batch *backends.PipelineBatch) (*RawFeatureOutput, error) {
	if p.OutputIndex < 0 || p.OutputIndex >= len(p.Model.OutputsMeta) {
		return nil, errors.New("raw image feature output index is out of range")
	}
	return rawFeatures(batch, p.Model.OutputsMeta[p.OutputIndex], p.OutputIndex)
}

// RunRaw loads image paths and returns unpooled, unnormalized model features.
func (p *ImageFeatureExtractionPipeline) RunRaw(ctx context.Context, inputs []string) (*RawFeatureOutput, error) {
	if len(inputs) == 0 {
		return nil, errors.New("raw features require a nonempty batch")
	}
	return backends.RunPipeline(ctx, len(inputs), func(batch *backends.PipelineBatch) error {
		images, err := imageutil.LoadImagesFromPaths(p.SessionContext, inputs)
		if err != nil {
			return fmt.Errorf("failed to load images: %w", err)
		}
		return p.preprocess(batch, images)
	}, p.forward, p.postprocessRaw)
}

// RunRawWithImages retains the batch, sequence and hidden dimensions of image features.
func (p *ImageFeatureExtractionPipeline) RunRawWithImages(ctx context.Context, images []image.Image) (*RawFeatureOutput, error) {
	if len(images) == 0 {
		return nil, errors.New("raw features require a nonempty batch")
	}
	return backends.RunPipeline(ctx, len(images), func(batch *backends.PipelineBatch) error {
		return p.preprocess(batch, images)
	}, p.forward, p.postprocessRaw)
}
