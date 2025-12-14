package pipelines

import (
	"errors"
	"fmt"
	"image"
	_ "image/gif"  // adds gif support
	_ "image/jpeg" // adds jpeg support
	_ "image/png"  // adds png support
	"strings"
	"sync/atomic"
	"time"

	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/util/imageutil"
	"github.com/knights-analytics/hugot/util/safeconv"
	"github.com/knights-analytics/hugot/util/vectorutil"
	_ "golang.org/x/image/webp" // adds webp support
)

// FeatureExtractionPipeline A feature extraction pipeline is a go version of
// https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/feature_extraction.py
// It supports both text and image inputs for embedding extraction.
type FeatureExtractionPipeline struct {
	*backends.BasePipeline
	OutputName    string
	Output        backends.InputOutputInfo
	OutputIndex   int // Record the index of the output selected, defaults to first (0)
	Normalization bool

	// Image mode fields (for vision encoders like CLIP visual)
	ImageMode          bool // true if this is a vision model
	imageFormat        string
	preprocessSteps    []imageutil.PreprocessStep
	normalizationSteps []imageutil.NormalizationStep
}
type FeatureExtractionOutput struct {
	Embeddings [][]float32
}

func (t *FeatureExtractionOutput) GetOutput() []any {
	out := make([]any, len(t.Embeddings))
	for i, embedding := range t.Embeddings {
		out[i] = any(embedding)
	}
	return out
}

// PIPELINE OPTIONS

// WithNormalization applies normalization to the mean pooled output of the feature pipeline.
func WithNormalization() backends.PipelineOption[*FeatureExtractionPipeline] {
	return func(pipeline *FeatureExtractionPipeline) error {
		pipeline.Normalization = true
		return nil
	}
}

// WithOutputName if there are multiple outputs from the underlying model, which output should
// be returned. If not passed, the first output from the feature pipeline is returned.
func WithOutputName(outputName string) backends.PipelineOption[*FeatureExtractionPipeline] {
	return func(pipeline *FeatureExtractionPipeline) error {
		pipeline.OutputName = outputName
		return nil
	}
}

// WithImageMode enables image feature extraction mode for vision encoders (e.g., CLIP visual encoder).
// When enabled, the pipeline accepts images instead of text and skips tokenization.
func WithImageMode() backends.PipelineOption[*FeatureExtractionPipeline] {
	return func(pipeline *FeatureExtractionPipeline) error {
		pipeline.ImageMode = true
		return nil
	}
}

// WithImagePreprocessSteps sets the image preprocessing steps (resize, crop, etc.).
func WithImagePreprocessSteps(steps ...imageutil.PreprocessStep) backends.PipelineOption[*FeatureExtractionPipeline] {
	return func(p *FeatureExtractionPipeline) error {
		p.preprocessSteps = append(p.preprocessSteps, steps...)
		return nil
	}
}

// WithImageNormalizationSteps sets the pixel normalization steps (rescale, normalize).
func WithImageNormalizationSteps(steps ...imageutil.NormalizationStep) backends.PipelineOption[*FeatureExtractionPipeline] {
	return func(p *FeatureExtractionPipeline) error {
		p.normalizationSteps = append(p.normalizationSteps, steps...)
		return nil
	}
}

// WithImageFormat sets the tensor format for image inputs.
// Use "NCHW" (channels first, default) or "NHWC" (channels last).
func WithImageFormat(format string) backends.PipelineOption[*FeatureExtractionPipeline] {
	return func(pipeline *FeatureExtractionPipeline) error {
		pipeline.imageFormat = format
		return nil
	}
}

// NewFeatureExtractionPipeline init a feature extraction pipeline.
func NewFeatureExtractionPipeline(config backends.PipelineConfig[*FeatureExtractionPipeline], s *options.Options, model *backends.Model) (*FeatureExtractionPipeline, error) {
	defaultPipeline, err := backends.NewBasePipeline(config, s, model)
	if err != nil {
		return nil, err
	}
	pipeline := &FeatureExtractionPipeline{BasePipeline: defaultPipeline}
	for _, o := range config.Options {
		err = o(pipeline)
		if err != nil {
			return nil, err
		}
	}

	// Set default image format if in image mode
	if pipeline.ImageMode && pipeline.imageFormat == "" {
		// Try to detect format from model inputs
		if len(model.InputsMeta) > 0 {
			shape := model.InputsMeta[0].Dimensions
			if len(shape) == 4 {
				if shape[1] == 3 && shape[3] != 3 {
					pipeline.imageFormat = "NCHW"
				} else if shape[3] == 3 {
					pipeline.imageFormat = "NHWC"
				} else {
					pipeline.imageFormat = "NCHW" // default
				}
			}
		}
		if pipeline.imageFormat == "" {
			pipeline.imageFormat = "NCHW" // default fallback
		}
	}

	// filter outputs
	if pipeline.OutputName != "" {
		for index, output := range model.OutputsMeta {
			if output.Name == pipeline.OutputName {
				pipeline.Output = output
				pipeline.OutputIndex = index
				break
			}
		}
		if pipeline.Output.Name == "" {
			return nil, fmt.Errorf("output %s is not available, outputs are: %s", pipeline.OutputName, strings.Join(backends.GetNames(model.OutputsMeta), ", "))
		}
	} else {
		if len(model.OutputsMeta) == 0 {
			return nil, fmt.Errorf("no model outputs metadata available for %s", model.Path)
		}
		pipeline.Output = model.OutputsMeta[0] // we take the first output otherwise, like transformers does
	}
	// validate pipeline
	err = pipeline.Validate()
	if err != nil {
		return nil, err
	}
	return pipeline, nil
}

// INTERFACE IMPLEMENTATIONS

func (p *FeatureExtractionPipeline) GetModel() *backends.Model {
	return p.Model
}

// GetMetadata returns metadata information about the pipeline, in particular:
// OutputInfo: names and dimensions of the output layer.
func (p *FeatureExtractionPipeline) GetMetadata() backends.PipelineMetadata {
	return backends.PipelineMetadata{
		OutputsInfo: []backends.OutputInfo{
			{
				Name:       p.OutputName,
				Dimensions: p.Output.Dimensions,
			},
		},
	}
}

// GetStatistics returns the runtime statistics for the pipeline.
func (p *FeatureExtractionPipeline) GetStatistics() backends.PipelineStatistics {
	statistics := backends.PipelineStatistics{}
	if p.Model.Tokenizer != nil && p.Model.Tokenizer.TokenizerTimings != nil {
		statistics.ComputeTokenizerStatistics(p.Model.Tokenizer.TokenizerTimings)
	}
	statistics.ComputeOnnxStatistics(p.PipelineTimings)
	return statistics
}

// Validate checks that the pipeline is valid.
func (p *FeatureExtractionPipeline) Validate() error {
	var validationErrors []error

	// Tokenizer is only required for text mode
	if !p.ImageMode && p.Model.Tokenizer == nil {
		validationErrors = append(validationErrors, fmt.Errorf("feature extraction pipeline requires a tokenizer for text mode"))
	}

	for _, input := range p.Model.InputsMeta {
		dims := []int64(input.Dimensions)
		maxDims := 3
		if p.ImageMode {
			maxDims = 4 // Image inputs are 4D: [batch, channels, height, width]
		}
		if len(dims) > maxDims {
			validationErrors = append(validationErrors, fmt.Errorf("inputs and outputs currently can have at most %d dimensions", maxDims))
		}
		nDynamicDimensions := 0
		for _, d := range dims {
			if d == -1 {
				nDynamicDimensions++
			}
		}
		if nDynamicDimensions > 2 {
			validationErrors = append(validationErrors, fmt.Errorf(`input %s has dimensions: %s. 
			There can only be max 2 dynamic dimensions (batch size and sequence length)`,
				input.Name, input.Dimensions.String()))
		}
	}
	return errors.Join(validationErrors...)
}

// Preprocess tokenizes the input strings.
func (p *FeatureExtractionPipeline) Preprocess(batch *backends.PipelineBatch, inputs []string) error {
	start := time.Now()
	backends.TokenizeInputs(batch, p.Model.Tokenizer, inputs)
	atomic.AddUint64(&p.Model.Tokenizer.TokenizerTimings.NumCalls, 1)
	atomic.AddUint64(&p.Model.Tokenizer.TokenizerTimings.TotalNS, safeconv.DurationToU64(time.Since(start)))
	err := backends.CreateInputTensors(batch, p.Model, p.Runtime)
	return err
}

// Forward performs the forward inference of the feature extraction pipeline.
func (p *FeatureExtractionPipeline) Forward(batch *backends.PipelineBatch) error {
	start := time.Now()
	err := backends.RunSessionOnBatch(batch, p.BasePipeline)
	if err != nil {
		return err
	}
	atomic.AddUint64(&p.PipelineTimings.NumCalls, 1)
	atomic.AddUint64(&p.PipelineTimings.TotalNS, safeconv.DurationToU64(time.Since(start)))
	return nil
}

// Postprocess parses the first output from the network similar to the transformers' implementation.
func (p *FeatureExtractionPipeline) Postprocess(batch *backends.PipelineBatch) (*FeatureExtractionOutput, error) {
	// TODO: this works if token embeddings are returned or sentence embeddings are returned.
	// in the former case embeddings are mean pooled. In the latter they are just returned.
	// to make this more general for other pipelines and to allow return of raw token embeddings,
	// we need an ndarray type that can be the return type of this pipeline. Need to think
	// about how to do this in a lightweight manner.
	output := batch.OutputValues[p.OutputIndex] // Use the index of the output we want to return
	batchEmbeddings := make([][]float32, batch.Size)
	outputDimensions := []int64(p.Output.Dimensions)
	embeddingDimension := outputDimensions[len(outputDimensions)-1]
	switch v := output.(type) {
	case [][]float32:
		batchEmbeddings = v
	case [][][]float32:
		for batchIndex, tokens := range v {
			batchEmbeddings[batchIndex] = meanPooling(tokens, batch.Input[batchIndex], batch.MaxSequenceLength, int(embeddingDimension))
		}
	default:
		return nil, fmt.Errorf("output type %T is not supported", output)
	}
	// Normalize embeddings (if asked), like in https://huggingface.co/sentence-transformers/all-mpnet-base-v2
	if p.Normalization {
		for i, embedding := range batchEmbeddings {
			batchEmbeddings[i] = vectorutil.Normalize(embedding, 2)
		}
	}
	return &FeatureExtractionOutput{Embeddings: batchEmbeddings}, nil
}

func meanPooling(tokens [][]float32, input backends.TokenizedInput, maxSequence int, dimensions int) []float32 {
	length := len(input.AttentionMask)
	vector := make([]float32, dimensions)
	for j := 0; j < maxSequence; j++ {
		// if there is no attention mask, take all tokens
		if length == 0 || (j+1 <= length && input.AttentionMask[j] != 0) {
			for k, vectorValue := range tokens[j] {
				vector[k] = vector[k] + vectorValue
			}
		}
	}
	numAttentionTokens := float32(input.MaxAttentionIndex + 1)
	for v, vectorValue := range vector {
		vector[v] = vectorValue / numAttentionTokens
	}
	return vector
}

// Run the pipeline on a batch of strings.
func (p *FeatureExtractionPipeline) Run(inputs []string) (backends.PipelineBatchOutput, error) {
	return p.RunPipeline(inputs)
}

// RunPipeline is like Run, but returns the concrete feature extraction output type rather than the interface.
func (p *FeatureExtractionPipeline) RunPipeline(inputs []string) (*FeatureExtractionOutput, error) {
	var runErrors []error
	batch := backends.NewBatch(len(inputs))
	defer func(*backends.PipelineBatch) {
		runErrors = append(runErrors, batch.Destroy())
	}(batch)
	runErrors = append(runErrors, p.Preprocess(batch, inputs))
	if e := errors.Join(runErrors...); e != nil {
		return nil, e
	}
	runErrors = append(runErrors, p.Forward(batch))
	if e := errors.Join(runErrors...); e != nil {
		return nil, e
	}
	result, postErr := p.Postprocess(batch)
	runErrors = append(runErrors, postErr)
	return result, errors.Join(runErrors...)
}

// IMAGE MODE METHODS

// PreprocessImages converts images to input tensors for vision models.
func (p *FeatureExtractionPipeline) PreprocessImages(batch *backends.PipelineBatch, images []image.Image) error {
	preprocessed, err := p.preprocessImages(images)
	if err != nil {
		return fmt.Errorf("failed to preprocess images: %w", err)
	}
	return backends.CreateImageTensors(batch, preprocessed, p.Runtime)
}

// preprocessImages applies preprocessing steps and converts images to tensor format.
func (p *FeatureExtractionPipeline) preprocessImages(images []image.Image) ([][][][]float32, error) {
	batchSize := len(images)
	out := make([][][][]float32, batchSize)

	for i, img := range images {
		processed := img

		// Apply preprocessing steps (resize, crop, etc.)
		for _, step := range p.preprocessSteps {
			var err error
			processed, err = step.Apply(processed)
			if err != nil {
				return nil, fmt.Errorf("failed to apply preprocessing step: %w", err)
			}
		}

		hh := processed.Bounds().Dy()
		ww := processed.Bounds().Dx()
		c := 3

		switch strings.ToUpper(p.imageFormat) {
		case "NHWC":
			// Height × Width × Channels
			tensor := make([][][]float32, hh)
			for y := range hh {
				tensor[y] = make([][]float32, ww)
				for x := range ww {
					tensor[y][x] = make([]float32, c)
				}
			}
			for y := range hh {
				for x := range ww {
					r, g, b, _ := processed.At(x, y).RGBA()
					rf := float32(r >> 8)
					gf := float32(g >> 8)
					bf := float32(b >> 8)
					for _, step := range p.normalizationSteps {
						rf, gf, bf = step.Apply(rf, gf, bf)
					}
					tensor[y][x][0] = rf
					tensor[y][x][1] = gf
					tensor[y][x][2] = bf
				}
			}
			out[i] = tensor

		case "NCHW":
			// Channels × Height × Width
			tensor := make([][][]float32, c)
			for ch := range c {
				tensor[ch] = make([][]float32, hh)
				for y := range hh {
					tensor[ch][y] = make([]float32, ww)
				}
			}
			for y := range hh {
				for x := range ww {
					r, g, b, _ := processed.At(x, y).RGBA()
					rf := float32(r >> 8)
					gf := float32(g >> 8)
					bf := float32(b >> 8)
					for _, step := range p.normalizationSteps {
						rf, gf, bf = step.Apply(rf, gf, bf)
					}
					tensor[0][y][x] = rf
					tensor[1][y][x] = gf
					tensor[2][y][x] = bf
				}
			}
			out[i] = tensor

		default:
			return nil, fmt.Errorf("unsupported image format: %s", p.imageFormat)
		}
	}
	return out, nil
}

// RunWithImages runs the pipeline on a batch of images (for vision models).
// Use this method when ImageMode is enabled.
func (p *FeatureExtractionPipeline) RunWithImages(images []image.Image) (*FeatureExtractionOutput, error) {
	if !p.ImageMode {
		return nil, fmt.Errorf("RunWithImages requires ImageMode to be enabled; use WithImageMode() option")
	}

	var runErrors []error
	batch := backends.NewBatch(len(images))
	defer func(*backends.PipelineBatch) {
		runErrors = append(runErrors, batch.Destroy())
	}(batch)

	runErrors = append(runErrors, p.PreprocessImages(batch, images))
	if e := errors.Join(runErrors...); e != nil {
		return nil, e
	}

	runErrors = append(runErrors, p.Forward(batch))
	if e := errors.Join(runErrors...); e != nil {
		return nil, e
	}

	result, postErr := p.Postprocess(batch)
	runErrors = append(runErrors, postErr)
	return result, errors.Join(runErrors...)
}

// RunWithImagePaths loads images from file paths and runs the pipeline.
// Convenience method that combines image loading with RunWithImages.
func (p *FeatureExtractionPipeline) RunWithImagePaths(paths []string) (*FeatureExtractionOutput, error) {
	images, err := imageutil.LoadImagesFromPaths(paths)
	if err != nil {
		return nil, fmt.Errorf("failed to load images: %w", err)
	}
	return p.RunWithImages(images)
}
