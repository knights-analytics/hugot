package pipelines

import (
	"errors"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"sort"
	"strings"
	"sync/atomic"
	"time"

	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/pipelineBackends"
	"github.com/knights-analytics/hugot/util"
)

// ImageClassificationPipeline is a go version of
// https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/image_classification.py
// It takes images (as file paths or image.Image) and returns top-k class predictions.
type ImageClassificationPipeline struct {
	*pipelineBackends.BasePipeline
	Output             pipelineBackends.InputOutputInfo
	TopK               int
	IDLabelMap         map[int]string
	preprocessSteps    []util.PreprocessStep
	normalizationSteps []util.NormalizationStep
	format             string
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

func WithPreprocessSteps(steps ...util.PreprocessStep) pipelineBackends.PipelineOption[*ImageClassificationPipeline] {
	return func(p *ImageClassificationPipeline) error {
		for _, step := range steps {
			p.preprocessSteps = append(p.preprocessSteps, step)
		}
		return nil
	}
}

func WithNormalizationSteps(steps ...util.NormalizationStep) pipelineBackends.PipelineOption[*ImageClassificationPipeline] {
	return func(p *ImageClassificationPipeline) error {
		for _, step := range steps {
			p.normalizationSteps = append(p.normalizationSteps, step)
		}
		return nil
	}
}

func WithNHWCFormat() pipelineBackends.PipelineOption[*ImageClassificationPipeline] {
	return func(pipeline *ImageClassificationPipeline) error {
		pipeline.format = "NHWC"
		return nil
	}
}

func WithNCHWFormat() pipelineBackends.PipelineOption[*ImageClassificationPipeline] {
	return func(pipeline *ImageClassificationPipeline) error {
		pipeline.format = "NCHW"
		return nil
	}
}

// WithTopK sets the number of top classifications to return.
func WithTopK(topK int) pipelineBackends.PipelineOption[*ImageClassificationPipeline] {
	return func(pipeline *ImageClassificationPipeline) error {
		pipeline.TopK = topK
		return nil
	}
}

func detectInputFormat(model *pipelineBackends.Model) (string, error) {
	inputInfo := model.InputsMeta
	if len(inputInfo) == 0 {
		return "", fmt.Errorf("no inputs found in model")
	}
	info := inputInfo[0]
	shape := info.Dimensions
	if len(shape) != 4 {
		return "", fmt.Errorf("expected 4D input, got %dD", len(shape))
	}
	if shape[1] == 3 && shape[3] != 3 {
		return "NCHW", nil
	} else if shape[3] == 3 {
		return "NHWC", nil
	}
	return "", fmt.Errorf("unable to infer format from shape %v", shape)
}

// NewImageClassificationPipeline initializes an image classification pipeline.
func NewImageClassificationPipeline(config pipelineBackends.PipelineConfig[*ImageClassificationPipeline], s *options.Options, model *pipelineBackends.Model) (*ImageClassificationPipeline, error) {
	defaultPipeline, err := pipelineBackends.NewBasePipeline(config, s, model)
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
		detectedFormat, err := detectInputFormat(model)
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

func (p *ImageClassificationPipeline) GetModel() *pipelineBackends.Model {
	return p.BasePipeline.Model
}

func (p *ImageClassificationPipeline) GetMetadata() pipelineBackends.PipelineMetadata {
	return pipelineBackends.PipelineMetadata{
		OutputsInfo: []pipelineBackends.OutputInfo{
			{
				Name:       p.Model.OutputsMeta[0].Name,
				Dimensions: p.Model.OutputsMeta[0].Dimensions,
			},
		},
	}
}

func (p *ImageClassificationPipeline) GetStats() []string {
	return []string{
		fmt.Sprintf("Statistics for pipeline: %s", p.PipelineName),
		fmt.Sprintf("ONNX: Total time=%s, Execution count=%d, Average query time=%s",
			time.Duration(p.PipelineTimings.TotalNS),
			p.PipelineTimings.NumCalls,
			time.Duration(float64(p.PipelineTimings.TotalNS)/math.Max(1, float64(p.PipelineTimings.NumCalls)))),
	}
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
func (p *ImageClassificationPipeline) Preprocess(batch *pipelineBackends.PipelineBatch, inputs []image.Image) error {
	preprocessed, err := p.preprocessImages(inputs)
	if err != nil {
		return fmt.Errorf("failed to preprocess images: %w", err)
	}
	return pipelineBackends.CreateImageTensors(batch, preprocessed, p.Runtime)
}

func (p *ImageClassificationPipeline) preprocessImages(images []image.Image) ([][][][]float32, error) {
	batchSize := len(images)
	out := make([][][][]float32, batchSize)

	for i, img := range images {
		processed := img
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

		switch strings.ToUpper(p.format) {
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
			return nil, fmt.Errorf("unsupported format: %s", p.format)
		}
	}
	return out, nil
}

// Forward runs inference.
func (p *ImageClassificationPipeline) Forward(batch *pipelineBackends.PipelineBatch) error {
	start := time.Now()
	if err := pipelineBackends.RunSessionOnBatch(batch, p.BasePipeline); err != nil {
		return err
	}
	atomic.AddUint64(&p.PipelineTimings.NumCalls, 1)
	atomic.AddUint64(&p.PipelineTimings.TotalNS, uint64(time.Since(start)))
	return nil
}

// Postprocess parses logits and returns top-k predictions for each image.
func (p *ImageClassificationPipeline) Postprocess(batch *pipelineBackends.PipelineBatch) (*ImageClassificationOutput, error) {
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
func (p *ImageClassificationPipeline) Run(inputs []string) (pipelineBackends.PipelineBatchOutput, error) {
	return p.RunPipeline(inputs)
}

// RunPipeline returns the concrete output type.
func (p *ImageClassificationPipeline) RunPipeline(inputs []string) (*ImageClassificationOutput, error) {
	var runErrors []error
	batch := pipelineBackends.NewBatch(len(inputs))
	defer func(*pipelineBackends.PipelineBatch) {
		runErrors = append(runErrors, batch.Destroy())
	}(batch)

	images, err := util.LoadImagesFromPaths(inputs)
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
	batch := pipelineBackends.NewBatch(len(inputs))
	defer func(*pipelineBackends.PipelineBatch) {
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
