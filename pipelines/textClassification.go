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
	IdLabelMap          map[int]string
	AggregationFunction func([]float32) []float32
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

func WithAggregationFunction(aggregationFunction func([]float32) []float32) TextClassificationOption {
	return func(pipeline *TextClassificationPipeline) {
		pipeline.AggregationFunction = aggregationFunction
	}
}

// NewTextClassificationPipeline initializes a new text classification pipeline
func NewTextClassificationPipeline(modelPath string, name string, opts ...TextClassificationOption) (*TextClassificationPipeline, error) {
	pipeline := &TextClassificationPipeline{}
	pipeline.ModelPath = modelPath
	pipeline.PipelineName = name
	for _, opt := range opts {
		opt(pipeline)
	}

	pipeline.TokenizerOptions = []tokenizers.EncodeOption{
		tokenizers.WithReturnAttentionMask(),
	}

	configPath := util.PathJoinSafe(modelPath, "config.json")
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

	// softmax by default

	pipeline.AggregationFunction = util.SoftMax

	// load onnx model
	loadErr := pipeline.loadModel()
	if loadErr != nil {
		return nil, loadErr
	}

	// we only support single label classification for now
	pipeline.OutputDim = int(pipeline.OutputsMeta[0].Dimensions[1])
	if len(pipeline.IdLabelMap) < 1 {
		return nil, fmt.Errorf("only single label classification models are currently supported and more than one label is required")
	}

	// output dimension
	if pipeline.OutputDim <= 0 {
		return nil, fmt.Errorf("pipeline configuration invalid: outputDim parameter must be greater than zero")
	}

	if len(pipeline.IdLabelMap) <= 0 {
		return nil, fmt.Errorf("pipeline configuration invalid: length of id2label map for token classification pipeline must be greater than zero")
	}
	if len(pipeline.IdLabelMap) != pipeline.OutputDim {
		return nil, fmt.Errorf("pipeline configuration invalid: length of id2label map does not match model output dimension")
	}
	return pipeline, nil
}

// TODO: perhaps this can be unified with the other pipelines

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

func (p *TextClassificationPipeline) Postprocess(batch PipelineBatch) (PipelineBatchOutput, error) {

	outputTensor := batch.OutputTensor
	output := make([][]float32, len(batch.Input))
	inputCounter := 0
	vectorCounter := 0
	inputVector := make([]float32, p.OutputDim)

	for _, result := range outputTensor {
		inputVector[vectorCounter] = result
		if vectorCounter == p.OutputDim-1 {

			output[inputCounter] = p.AggregationFunction(inputVector)
			vectorCounter = 0
			inputVector = make([]float32, p.OutputDim)
			inputCounter++
		} else {
			vectorCounter++
		}
	}

	// batchClassificationOutputs := make([][]ClassificationOutput, len(batch.Input))
	batchClassificationOutputs := TextClassificationOutput{
		ClassificationOutputs: make([][]ClassificationOutput, len(batch.Input)),
	}

	var err error

	for i := 0; i < len(batch.Input); i++ {
		// since we only support single label classification for now there's only one classification output in the slice
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
	}
	return &batchClassificationOutputs, err
}

// Run the pipeline on a string batch
func (p *TextClassificationPipeline) Run(inputs []string) (PipelineBatchOutput, error) {
	batch := p.Preprocess(inputs)
	batch, err := p.Forward(batch)
	if err != nil {
		return nil, err
	}
	return p.Postprocess(batch)
}
