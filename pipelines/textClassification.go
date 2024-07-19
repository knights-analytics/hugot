package pipelines

import (
	"errors"
	"fmt"
	"math"
	"sync/atomic"
	"time"

	util "github.com/knights-analytics/hugot/utils"

	jsoniter "github.com/json-iterator/go"
	ort "github.com/yalue/onnxruntime_go"
)

// types

type TextClassificationPipeline struct {
	basePipeline
	IDLabelMap              map[int]string
	AggregationFunctionName string
	ProblemType             string
}

type TextClassificationPipelineConfig struct {
	IDLabelMap map[int]string `json:"id2label"`
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

// NewTextClassificationPipeline initializes a new text classification pipeline.
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

	// read id to label map
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

	pipeline.IDLabelMap = pipelineInputConfig.IDLabelMap

	// onnx model init
	model, err := loadOnnxModelBytes(pipeline.ModelPath, pipeline.OnnxFilename)
	if err != nil {
		return nil, err
	}

	// init of inputs and outputs
	inputs, outputs, err := loadInputOutputMeta(model)
	if err != nil {
		return nil, err
	}
	pipeline.InputsMeta = inputs
	pipeline.OutputsMeta = outputs

	// tokenizer init
	pipeline.TokenizerOptions, err = getTokenizerOptions(inputs)
	if err != nil {
		return nil, err
	}

	tk, tkErr := loadTokenizer(pipeline.ModelPath)
	if tkErr != nil {
		return nil, tkErr
	}
	pipeline.Tokenizer = tk

	// creation of the session
	session, err := createSession(model, inputs, pipeline.OutputsMeta, ortOptions)
	if err != nil {
		return nil, err
	}
	pipeline.OrtSession = session

	// initialize timings
	pipeline.PipelineTimings = &timings{}
	pipeline.TokenizerTimings = &timings{}

	// validate
	err = pipeline.Validate()
	if err != nil {
		errDestroy := pipeline.Destroy()
		return nil, errors.Join(err, errDestroy)
	}
	return pipeline, nil
}

// INTERFACE IMPLEMENTATION

// GetMetadata returns metadata information about the pipeline, in particular:
// OutputInfo: names and dimensions of the output layer used for text classification.
func (p *TextClassificationPipeline) GetMetadata() PipelineMetadata {
	return PipelineMetadata{
		OutputsInfo: []OutputInfo{
			{
				Name:       p.OutputsMeta[0].Name,
				Dimensions: p.OutputsMeta[0].Dimensions,
			},
		},
	}
}

// Destroy frees the text classification pipeline resources.
func (p *TextClassificationPipeline) Destroy() error {
	return destroySession(p.Tokenizer, p.OrtSession)
}

// GetStats returns the runtime statistics for the pipeline.
func (p *TextClassificationPipeline) GetStats() []string {
	return []string{
		fmt.Sprintf("Statistics for pipeline: %s", p.PipelineName),
		fmt.Sprintf("Tokenizer: Total time=%s, Execution count=%d, Average query time=%s",
			time.Duration(p.TokenizerTimings.TotalNS),
			p.TokenizerTimings.NumCalls,
			time.Duration(float64(p.TokenizerTimings.TotalNS)/math.Max(1, float64(p.TokenizerTimings.NumCalls)))),
		fmt.Sprintf("ONNX: Total time=%s, Execution count=%d, Average query time=%s",
			time.Duration(p.PipelineTimings.TotalNS),
			p.PipelineTimings.NumCalls,
			time.Duration(float64(p.PipelineTimings.TotalNS)/math.Max(1, float64(p.PipelineTimings.NumCalls)))),
	}
}

// Validate checks that the pipeline is valid.
func (p *TextClassificationPipeline) Validate() error {
	var validationErrors []error

	if len(p.IDLabelMap) <= 0 {
		validationErrors = append(validationErrors, fmt.Errorf("pipeline configuration invalid: length of id2label map for token classification pipeline must be greater than zero"))
	}

	outDims := p.OutputsMeta[0].Dimensions
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
	if len(p.IDLabelMap) != nLogits {
		validationErrors = append(validationErrors, fmt.Errorf("pipeline configuration invalid: length of id2label map does not match number of logits in output (%d)", nLogits))
	}
	return errors.Join(validationErrors...)
}

// Preprocess tokenizes the input strings.
func (p *TextClassificationPipeline) Preprocess(batch *PipelineBatch, inputs []string) error {
	start := time.Now()
	tokenizeInputs(batch, p.Tokenizer, inputs, p.TokenizerOptions)
	atomic.AddUint64(&p.TokenizerTimings.NumCalls, 1)
	atomic.AddUint64(&p.TokenizerTimings.TotalNS, uint64(time.Since(start)))
	err := createInputTensors(batch, p.InputsMeta)
	return err
}

func (p *TextClassificationPipeline) Forward(batch *PipelineBatch) error {
	start := time.Now()
	err := runSessionOnBatch(batch, p.OrtSession, p.OutputsMeta)
	if err != nil {
		return err
	}
	atomic.AddUint64(&p.PipelineTimings.NumCalls, 1)
	atomic.AddUint64(&p.PipelineTimings.TotalNS, uint64(time.Since(start)))
	return nil
}

func (p *TextClassificationPipeline) Postprocess(batch *PipelineBatch) (*TextClassificationOutput, error) {
	outputTensor := batch.OutputTensors[0]
	outputDims := p.OutputsMeta[0].Dimensions
	nLogit := outputDims[len(outputDims)-1]
	output := make([][]float32, len(batch.Input))
	inputCounter := 0
	vectorCounter := 0
	inputVector := make([]float32, nLogit)
	var aggregationFunction func([]float32) []float32
	switch p.AggregationFunctionName {
	case "SIGMOID":
		aggregationFunction = util.Sigmoid
	case "SOFTMAX":
		aggregationFunction = util.SoftMax
	default:
		return nil, fmt.Errorf("aggregation function %s is not supported", p.AggregationFunctionName)
	}

	for _, result := range outputTensor.GetData() {
		inputVector[vectorCounter] = result
		if vectorCounter == int(nLogit)-1 {
			output[inputCounter] = aggregationFunction(inputVector)
			vectorCounter = 0
			inputVector = make([]float32, nLogit)
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
			class, ok := p.IDLabelMap[index]
			if !ok {
				err = fmt.Errorf("class with index number %d not found in id label map", index)
			}
			inputClassificationOutputs[0] = ClassificationOutput{
				Label: class,
				Score: value,
			}
			batchClassificationOutputs.ClassificationOutputs[i] = inputClassificationOutputs
		case "multiLabel":
			inputClassificationOutputs := make([]ClassificationOutput, len(p.IDLabelMap))
			for j := range output[i] {
				class, ok := p.IDLabelMap[j]
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

// Run the pipeline on a string batch.
func (p *TextClassificationPipeline) Run(inputs []string) (PipelineBatchOutput, error) {
	return p.RunPipeline(inputs)
}

func (p *TextClassificationPipeline) RunPipeline(inputs []string) (*TextClassificationOutput, error) {
	var runErrors []error
	batch := NewBatch()
	defer func(*PipelineBatch) {
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
