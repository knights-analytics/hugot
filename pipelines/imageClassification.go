package pipelines

import (
	"errors"
	"fmt"
	"image"
	_ "image/jpeg" // adds jpeg support to image
	_ "image/png"  // adds png support to image
	"sort"
	"sync/atomic"
	"time"

	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/util/imageutil"
	"github.com/knights-analytics/hugot/util/safeconv"
)

// ImageClassificationPipeline is a go version of
// https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/image_classification.py
// It takes images (as file paths or image.Image) and returns top-k class predictions.
type ImageClassificationPipeline struct {
	*backends.BasePipeline
	IDLabelMap         map[int]string
	format             string
	Output             backends.InputOutputInfo
	preprocessSteps    []imageutil.PreprocessStep
	normalizationSteps []imageutil.NormalizationStep
	TopK               int
}
type ImageClassificationResult struct {
	Label      string
	Score      float32
	ClassIndex int
}
type ImageClassificationOutput struct {
	Predictions [][]ImageClassificationResult // batch of results
}

func (o *ImageClassificationOutput) GetOutput() []any {
	out := make([]any, len(o.Predictions))
	for i, preds := range o.Predictions {
		out[i] = any(preds)
	}
	return out
}

func WithPreprocessSteps(steps ...imageutil.PreprocessStep) backends.PipelineOption[*ImageClassificationPipeline] {
	return func(p *ImageClassificationPipeline) error {
		p.preprocessSteps = append(p.preprocessSteps, steps...)
		return nil
	}
}

func WithNormalizationSteps(steps ...imageutil.NormalizationStep) backends.PipelineOption[*ImageClassificationPipeline] {
	return func(p *ImageClassificationPipeline) error {
		p.normalizationSteps = append(p.normalizationSteps, steps...)
		return nil
	}
}

func WithNHWCFormat() backends.PipelineOption[*ImageClassificationPipeline] {
	return func(pipeline *ImageClassificationPipeline) error {
		pipeline.format = "NHWC"
		return nil
	}
}

func WithNCHWFormat() backends.PipelineOption[*ImageClassificationPipeline] {
	return func(pipeline *ImageClassificationPipeline) error {
		pipeline.format = "NCHW"
		return nil
	}
}

// WithTopK sets the number of top classifications to return.
func WithTopK(topK int) backends.PipelineOption[*ImageClassificationPipeline] {
	return func(pipeline *ImageClassificationPipeline) error {
		pipeline.TopK = topK
		return nil
	}
}

// NewImageClassificationPipeline initializes an image classification pipeline.
func NewImageClassificationPipeline(config backends.PipelineConfig[*ImageClassificationPipeline], s *options.Options, model *backends.Model) (*ImageClassificationPipeline, error) {
	defaultPipeline, err := backends.NewBasePipeline(config, s, model)
	if err != nil {
		return nil, err
	}
	pipeline := &ImageClassificationPipeline{BasePipeline: defaultPipeline, TopK: 5} // default topK=5
	for _, o := range config.Options {
		err = o(pipeline)
		if err != nil {
			return nil, err
		}
	}
	pipeline.IDLabelMap = model.IDLabelMap
	if pipeline.format == "" {
		detectedFormat, err := backends.DetectImageTensorFormat(model)
		if err != nil {
			return nil, err
		}
		pipeline.format = detectedFormat
	}
	// validate pipeline
	err = pipeline.Validate()
	if err != nil {
		return nil, err
	}
	return pipeline, nil
}

// INTERFACE IMPLEMENTATIONS

func (p *ImageClassificationPipeline) GetModel() *backends.Model {
	return p.Model
}

func (p *ImageClassificationPipeline) GetMetadata() backends.PipelineMetadata {
	return backends.PipelineMetadata{
		OutputsInfo: []backends.OutputInfo{
			{
				Name:       p.Model.OutputsMeta[0].Name,
				Dimensions: p.Model.OutputsMeta[0].Dimensions,
			},
		},
	}
}

func (p *ImageClassificationPipeline) GetStatistics() backends.PipelineStatistics {
	statistics := backends.PipelineStatistics{}
	backends.ComputeOnnxStatistics(&statistics, p.PipelineTimings)
	return statistics
}

func (p *ImageClassificationPipeline) Validate() error {
	var validationErrors []error
	for _, input := range p.Model.InputsMeta {
		dims := []int64(input.Dimensions)
		if len(dims) != 4 {
			validationErrors = append(validationErrors, fmt.Errorf("input %s: expected 4 dimensions (batch, channels, height, width), got %d", input.Name, len(dims)))
		}
	}
	return errors.Join(validationErrors...)
}

// Preprocess decodes images from file paths or image.Image and creates input tensors.
// Preprocess loads images from file paths and creates input tensors.
func (p *ImageClassificationPipeline) Preprocess(batch *backends.PipelineBatch, inputs []image.Image) error {
	preprocessed, err := backends.PreprocessImages(p.format, inputs, p.preprocessSteps, p.normalizationSteps)
	if err != nil {
		return fmt.Errorf("failed to preprocess images: %w", err)
	}
	return backends.CreateImageTensors(batch, p.Model, preprocessed, p.Runtime)
}

// Forward runs inference.
func (p *ImageClassificationPipeline) Forward(batch *backends.PipelineBatch) error {
	start := time.Now()
	if err := backends.RunSessionOnBatch(batch, p.BasePipeline); err != nil {
		return err
	}
	atomic.AddUint64(&p.PipelineTimings.NumCalls, 1)
	atomic.AddUint64(&p.PipelineTimings.TotalNS, safeconv.DurationToU64(time.Since(start)))
	return nil
}

// Postprocess parses logits and returns top-k predictions for each image.
func (p *ImageClassificationPipeline) Postprocess(batch *backends.PipelineBatch) (*ImageClassificationOutput, error) {
	output := batch.OutputValues[0]
	var batchPreds [][]ImageClassificationResult
	logits, ok := output.([][]float32)
	if !ok {
		return nil, fmt.Errorf("output type %T is not supported", output)
	}
	for _, logit := range logits {
		preds := getTopK(logit, p.TopK, p.IDLabelMap)
		batchPreds = append(batchPreds, preds)
	}
	return &ImageClassificationOutput{Predictions: batchPreds}, nil
}

func getTopK(logits []float32, k int, labels map[int]string) []ImageClassificationResult {
	type kv struct {
		Idx int
		Val float32
	}
	var arr []kv
	for i, v := range logits {
		arr = append(arr, kv{i, v})
	}
	sort.Slice(arr, func(i, j int) bool { return arr[i].Val > arr[j].Val })
	if k > len(arr) {
		k = len(arr)
	}
	var results []ImageClassificationResult
	for i := 0; i < k; i++ {
		label := fmt.Sprintf("class_%d", arr[i].Idx)
		if labels != nil {
			if l, ok := labels[arr[i].Idx]; ok {
				label = l
			}
		}
		results = append(results, ImageClassificationResult{
			Label:      label,
			Score:      arr[i].Val,
			ClassIndex: arr[i].Idx,
		})
	}
	return results
}

// Run runs the pipeline on a batch of image file paths.
func (p *ImageClassificationPipeline) Run(inputs []string) (backends.PipelineBatchOutput, error) {
	return p.RunPipeline(inputs)
}

// RunPipeline returns the concrete output type.
func (p *ImageClassificationPipeline) RunPipeline(inputs []string) (*ImageClassificationOutput, error) {
	var runErrors []error
	batch := backends.NewBatch(len(inputs))
	defer func(*backends.PipelineBatch) {
		runErrors = append(runErrors, batch.Destroy())
	}(batch)
	images, err := imageutil.LoadImagesFromPaths(inputs)
	if err != nil {
		return nil, fmt.Errorf("failed to load images: %w", err)
	}
	runErrors = append(runErrors, p.Preprocess(batch, images))
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

func (p *ImageClassificationPipeline) RunWithImages(inputs []image.Image) (*ImageClassificationOutput, error) {
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
