package pipelines

import (
	"errors"
	"fmt"
	"math"
	"sync/atomic"
	"time"

	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/pipelineBackends"
	"github.com/knights-analytics/hugot/util"
)

// types

type TextClassificationPipeline struct {
	*pipelineBackends.BasePipeline
	AggregationFunctionName string
	ProblemType             string
}

type ClassificationOutput struct {
	Label string
	Score float32
}

type TextClassificationOutput struct {
	ClassificationOutputs [][]ClassificationOutput
}

func (t *TextClassificationOutput) GetOutput() []any {
	out := make([]any, len(t.ClassificationOutputs))
	for i, classificationOutput := range t.ClassificationOutputs {
		out[i] = any(classificationOutput)
	}
	return out
}

// options

func WithSoftmax() pipelineBackends.PipelineOption[*TextClassificationPipeline] {
	return func(pipeline *TextClassificationPipeline) error {
		pipeline.AggregationFunctionName = "SOFTMAX"
		return nil
	}
}

func WithSigmoid() pipelineBackends.PipelineOption[*TextClassificationPipeline] {
	return func(pipeline *TextClassificationPipeline) error {
		pipeline.AggregationFunctionName = "SIGMOID"
		return nil
	}
}

func WithSingleLabel() pipelineBackends.PipelineOption[*TextClassificationPipeline] {
	return func(pipeline *TextClassificationPipeline) error {
		pipeline.ProblemType = "singleLabel"
		return nil
	}
}

func WithMultiLabel() pipelineBackends.PipelineOption[*TextClassificationPipeline] {
	return func(pipeline *TextClassificationPipeline) error {
		pipeline.ProblemType = "multiLabel"
		return nil
	}
}

// NewTextClassificationPipeline initializes a new text classification pipeline.
func NewTextClassificationPipeline(config pipelineBackends.PipelineConfig[*TextClassificationPipeline], s *options.Options, model *pipelineBackends.Model) (*TextClassificationPipeline, error) {

	defaultPipeline, err := pipelineBackends.NewBasePipeline(config, s, model)
	if err != nil {
		return nil, err
	}

	pipeline := &TextClassificationPipeline{BasePipeline: defaultPipeline}
	for _, o := range config.Options {
		err = o(pipeline)
		if err != nil {
			return nil, err
		}
	}

	if pipeline.ProblemType == "" {
		pipeline.ProblemType = "singleLabel"
	}
	if pipeline.AggregationFunctionName == "" {
		if pipeline.PipelineName == "singleLabel" {
			pipeline.AggregationFunctionName = "SOFTMAX"
		} else {
			pipeline.AggregationFunctionName = "SIGMOID"
		}
	}

	// validate
	err = pipeline.Validate()
	if err != nil {
		return nil, err
	}
	return pipeline, nil
}

// INTERFACE IMPLEMENTATION

func (p *TextClassificationPipeline) GetModel() *pipelineBackends.Model {
	return p.BasePipeline.Model
}

// GetMetadata returns metadata information about the pipeline, in particular:
// OutputInfo: names and dimensions of the output layer used for text classification.
func (p *TextClassificationPipeline) GetMetadata() pipelineBackends.PipelineMetadata {
	return pipelineBackends.PipelineMetadata{
		OutputsInfo: []pipelineBackends.OutputInfo{
			{
				Name:       p.Model.OutputsMeta[0].Name,
				Dimensions: p.Model.OutputsMeta[0].Dimensions,
			},
		},
	}
}

// GetStats returns the runtime statistics for the pipeline.
func (p *TextClassificationPipeline) GetStats() []string {
	return []string{
		fmt.Sprintf("Statistics for pipeline: %s", p.PipelineName),
		fmt.Sprintf("Tokenizer: Total time=%s, Execution count=%d, Average query time=%s",
			time.Duration(p.Model.Tokenizer.TokenizerTimings.TotalNS),
			p.Model.Tokenizer.TokenizerTimings.NumCalls,
			time.Duration(float64(p.Model.Tokenizer.TokenizerTimings.TotalNS)/math.Max(1, float64(p.Model.Tokenizer.TokenizerTimings.NumCalls)))),
		fmt.Sprintf("ONNX: Total time=%s, Execution count=%d, Average query time=%s",
			time.Duration(p.PipelineTimings.TotalNS),
			p.PipelineTimings.NumCalls,
			time.Duration(float64(p.PipelineTimings.TotalNS)/math.Max(1, float64(p.PipelineTimings.NumCalls)))),
	}
}

// Validate checks that the pipeline is valid.
func (p *TextClassificationPipeline) Validate() error {
	var validationErrors []error

	if p.Model.Tokenizer == nil {
		validationErrors = append(validationErrors, fmt.Errorf("feature extraction pipeline requires a tokenizer"))
	}

	if len(p.Model.IDLabelMap) <= 0 {
		validationErrors = append(validationErrors, fmt.Errorf("pipeline configuration invalid: length of id2label map for text classification pipeline must be greater than zero"))
	}

	outDims := p.Model.OutputsMeta[0].Dimensions
	if len(outDims) != 2 {
		validationErrors = append(validationErrors, fmt.Errorf("pipeline configuration invalid: text classification must have 2 dimensional output"))
	}
	dynamicBatch := false
	for _, d := range outDims {
		if d == -1 {
			if dynamicBatch {
				validationErrors = append(validationErrors, fmt.Errorf("pipeline configuration invalid: text classification must have max one dynamic dimensions (input)"))
				break
			}
			dynamicBatch = true
		}
	}
	nLogits := int(outDims[len(outDims)-1])
	if len(p.Model.IDLabelMap) != nLogits {
		validationErrors = append(validationErrors, fmt.Errorf("pipeline configuration invalid: length of id2label map does not match number of logits in output (%d)", nLogits))
	}
	return errors.Join(validationErrors...)
}

// Preprocess tokenizes the input strings.
func (p *TextClassificationPipeline) Preprocess(batch *pipelineBackends.PipelineBatch, inputs []string) error {
	start := time.Now()
	pipelineBackends.TokenizeInputs(batch, p.Model.Tokenizer, inputs)
	atomic.AddUint64(&p.Model.Tokenizer.TokenizerTimings.NumCalls, 1)
	atomic.AddUint64(&p.Model.Tokenizer.TokenizerTimings.TotalNS, uint64(time.Since(start)))
	err := pipelineBackends.CreateInputTensors(batch, p.Model, p.Runtime)
	return err
}

func (p *TextClassificationPipeline) Forward(batch *pipelineBackends.PipelineBatch) error {
	start := time.Now()
	err := pipelineBackends.RunSessionOnBatch(batch, p.BasePipeline)
	if err != nil {
		return err
	}
	atomic.AddUint64(&p.PipelineTimings.NumCalls, 1)
	atomic.AddUint64(&p.PipelineTimings.TotalNS, uint64(time.Since(start)))
	return nil
}

func (p *TextClassificationPipeline) Postprocess(batch *pipelineBackends.PipelineBatch) (*TextClassificationOutput, error) {
	var aggregationFunction func([]float32) []float32
	switch p.AggregationFunctionName {
	case "SIGMOID":
		aggregationFunction = util.Sigmoid
	case "SOFTMAX":
		aggregationFunction = util.SoftMax
	default:
		return nil, fmt.Errorf("aggregation function %s is not supported", p.AggregationFunctionName)
	}

	output := batch.OutputValues[0]
	var outputCast [][]float32
	switch v := output.(type) {
	case [][]float32:
		for i, logits := range v {
			v[i] = aggregationFunction(logits)
		}
		outputCast = v
	default:
		return nil, fmt.Errorf("output is not 2D, expected batch size x logits, got %T", output)
	}

	batchClassificationOutputs := TextClassificationOutput{
		ClassificationOutputs: make([][]ClassificationOutput, len(batch.Input)),
	}

	var err error

	for i := 0; i < len(batch.Input); i++ {
		switch p.ProblemType {
		case "singleLabel":
			inputClassificationOutputs := make([]ClassificationOutput, 1)
			index, value, errArgMax := util.ArgMax(outputCast[i])
			if errArgMax != nil {
				err = errArgMax
				continue
			}
			class, ok := p.Model.IDLabelMap[index]
			if !ok {
				err = fmt.Errorf("class with index number %d not found in id label map", index)
			}
			inputClassificationOutputs[0] = ClassificationOutput{
				Label: class,
				Score: value,
			}
			batchClassificationOutputs.ClassificationOutputs[i] = inputClassificationOutputs
		case "multiLabel":
			inputClassificationOutputs := make([]ClassificationOutput, len(p.Model.IDLabelMap))
			for j := range outputCast[i] {
				class, ok := p.Model.IDLabelMap[j]
				if !ok {
					err = fmt.Errorf("class with index number %d not found in id label map", j)
				}
				inputClassificationOutputs[j] = ClassificationOutput{
					Label: class,
					Score: outputCast[i][j],
				}
			}
			batchClassificationOutputs.ClassificationOutputs[i] = inputClassificationOutputs
		default:
			err = fmt.Errorf("problem type %s not recognized", p.ProblemType)
		}
	}
	return &batchClassificationOutputs, err
}

// Run the pipeline on a string batch.
func (p *TextClassificationPipeline) Run(inputs []string) (pipelineBackends.PipelineBatchOutput, error) {
	return p.RunPipeline(inputs)
}

func (p *TextClassificationPipeline) RunPipeline(inputs []string) (*TextClassificationOutput, error) {
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
