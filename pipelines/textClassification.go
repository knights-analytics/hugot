package pipelines

import (
	"errors"
	"fmt"
	"sync/atomic"
	"time"

	util "github.com/knights-analytics/hugot/utils"

	jsoniter "github.com/json-iterator/go"
	"github.com/knights-analytics/tokenizers"
	ort "github.com/yalue/onnxruntime_go"
)

// types

type TextClassificationPipeline struct {
	BasePipeline
	IdLabelMap              map[int]string
	AggregationFunctionName string
	ProblemType             string
}

type TextClassificationPipelineConfig struct {
	IdLabelMap map[int]string `json:"id2label"`
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

type TextClassificationOption func(eo *TextClassificationPipeline)

func WithSoftmax() PipelineOption[*TextClassificationPipeline] {
	return func(pipeline *TextClassificationPipeline) {
		pipeline.AggregationFunctionName = "SOFTMAX"
	}
}

func WithSigmoid() PipelineOption[*TextClassificationPipeline] {
	return func(pipeline *TextClassificationPipeline) {
		pipeline.AggregationFunctionName = "SIGMOID"
	}
}

func WithSingleLabel() PipelineOption[*TextClassificationPipeline] {
	return func(pipeline *TextClassificationPipeline) {
		pipeline.ProblemType = "singleLabel"
	}
}

func WithMultiLabel() PipelineOption[*TextClassificationPipeline] {
	return func(pipeline *TextClassificationPipeline) {
		pipeline.ProblemType = "multiLabel"
	}
}

// NewTextClassificationPipeline initializes a new text classification pipeline
func NewTextClassificationPipeline(config PipelineConfig[*TextClassificationPipeline], ortOptions *ort.SessionOptions) (*TextClassificationPipeline, error) {
	pipeline := &TextClassificationPipeline{}
	pipeline.ModelPath = config.ModelPath
	pipeline.PipelineName = config.Name
	pipeline.OrtOptions = ortOptions
	pipeline.OnnxFilename = config.OnnxFilename

	for _, o := range config.Options {
		o(pipeline)
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

	pipeline.TokenizerOptions = []tokenizers.EncodeOption{
		tokenizers.WithReturnAttentionMask(),
	}

	configPath := util.PathJoinSafe(pipeline.ModelPath, "config.json")
	pipelineInputConfig := TextClassificationPipelineConfig{}
	mapBytes, err := util.ReadFileBytes(configPath)
	if err != nil {
		return nil, err
	}
	err = jsoniter.Unmarshal(mapBytes, &pipelineInputConfig)
	if err != nil {
		return nil, err
	}

	pipeline.IdLabelMap = pipelineInputConfig.IdLabelMap
	pipeline.PipelineTimings = &Timings{}
	pipeline.TokenizerTimings = &Timings{}

	// load onnx model
	loadErr := pipeline.loadModel()
	if loadErr != nil {
		return nil, loadErr
	}

	pipeline.OutputDim = int(pipeline.OutputsMeta[0].Dimensions[1])

	// validate
	validationErrors := pipeline.Validate()
	if validationErrors != nil {
		return nil, validationErrors
	}

	return pipeline, nil
}

func (p *TextClassificationPipeline) Validate() error {
	var validationErrors []error

	if len(p.IdLabelMap) < 1 {
		validationErrors = append(validationErrors, fmt.Errorf("only single label classification models are currently supported and more than one label is required"))
	}
	if p.OutputDim <= 0 {
		validationErrors = append(validationErrors, fmt.Errorf("pipeline configuration invalid: outputDim parameter must be greater than zero"))
	}
	if len(p.IdLabelMap) <= 0 {
		validationErrors = append(validationErrors, fmt.Errorf("pipeline configuration invalid: length of id2label map for token classification pipeline must be greater than zero"))
	}
	if len(p.IdLabelMap) != p.OutputDim {
		validationErrors = append(validationErrors, fmt.Errorf("pipeline configuration invalid: length of id2label map does not match model output dimension"))
	}
	return errors.Join(validationErrors...)
}

func (p *TextClassificationPipeline) Forward(batch PipelineBatch) (PipelineBatch, error) {
	start := time.Now()

	actualBatchSize := int64(len(batch.Input))
	maxSequence := int64(batch.MaxSequence)
	inputTensors, err := p.getInputTensors(batch, actualBatchSize, maxSequence)
	if err != nil {
		return batch, err
	}

	defer func(inputTensors []ort.ArbitraryTensor) {
		for _, tensor := range inputTensors {
			err = errors.Join(err, tensor.Destroy())
		}
	}(inputTensors)

	outputTensor, errTensor := ort.NewEmptyTensor[float32](ort.NewShape(actualBatchSize, int64(p.OutputDim)))
	if errTensor != nil {
		return batch, errTensor
	}

	defer func(outputTensor *ort.Tensor[float32]) {
		err = errors.Join(err, outputTensor.Destroy())
	}(outputTensor)

	// Run Onnx model
	errOnnx := p.OrtSession.Run(inputTensors, []ort.ArbitraryTensor{outputTensor})
	if errOnnx != nil {
		return batch, errOnnx
	}
	batch.OutputTensor = outputTensor.GetData()

	atomic.AddUint64(&p.PipelineTimings.NumCalls, 1)
	atomic.AddUint64(&p.PipelineTimings.TotalNS, uint64(time.Since(start)))
	return batch, err
}

func (p *TextClassificationPipeline) Postprocess(batch PipelineBatch) (*TextClassificationOutput, error) {
	outputTensor := batch.OutputTensor
	output := make([][]float32, len(batch.Input))
	inputCounter := 0
	vectorCounter := 0
	inputVector := make([]float32, p.OutputDim)
	var aggregationFunction func([]float32) []float32
	switch p.AggregationFunctionName {
	case "SIGMOID":
		aggregationFunction = util.Sigmoid
	case "SOFTMAX":
		aggregationFunction = util.SoftMax
	default:
		return nil, fmt.Errorf("aggregation function %s is not supported", p.AggregationFunctionName)
	}

	for _, result := range outputTensor {
		inputVector[vectorCounter] = result
		if vectorCounter == p.OutputDim-1 {

			output[inputCounter] = aggregationFunction(inputVector)
			vectorCounter = 0
			inputVector = make([]float32, p.OutputDim)
			inputCounter++
		} else {
			vectorCounter++
		}
	}

	batchClassificationOutputs := TextClassificationOutput{
		ClassificationOutputs: make([][]ClassificationOutput, len(batch.Input)),
	}

	var err error

	for i := 0; i < len(batch.Input); i++ {
		switch p.ProblemType {
		case "singleLabel":
			inputClassificationOutputs := make([]ClassificationOutput, 1)
			index, value, errArgMax := util.ArgMax(output[i])
			if errArgMax != nil {
				err = errArgMax
				continue
			}
			class, ok := p.IdLabelMap[index]
			if !ok {
				err = fmt.Errorf("class with index number %d not found in id label map", index)
			}
			inputClassificationOutputs[0] = ClassificationOutput{
				Label: class,
				Score: value,
			}
			batchClassificationOutputs.ClassificationOutputs[i] = inputClassificationOutputs
		case "multiLabel":
			inputClassificationOutputs := make([]ClassificationOutput, len(p.IdLabelMap))
			for j := range output[i] {
				class, ok := p.IdLabelMap[j]
				if !ok {
					err = fmt.Errorf("class with index number %d not found in id label map", j)
				}
				inputClassificationOutputs[j] = ClassificationOutput{
					Label: class,
					Score: output[i][j],
				}
			}
			batchClassificationOutputs.ClassificationOutputs[i] = inputClassificationOutputs
		default:
			err = fmt.Errorf("problem type %s not recognized", p.ProblemType)
		}
	}
	return &batchClassificationOutputs, err
}

// Run the pipeline on a string batch
func (p *TextClassificationPipeline) Run(inputs []string) (PipelineBatchOutput, error) {
	return p.RunPipeline(inputs)
}

func (p *TextClassificationPipeline) RunPipeline(inputs []string) (*TextClassificationOutput, error) {
	batch := p.Preprocess(inputs)
	batch, err := p.Forward(batch)
	if err != nil {
		return nil, err
	}
	return p.Postprocess(batch)
}
