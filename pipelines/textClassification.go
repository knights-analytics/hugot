package pipelines

import (
	"sync/atomic"
	"time"

	util "github.com/Knights-Analytics/HuGo/utils"

	"github.com/Knights-Analytics/HuGo/utils/checks"
	"github.com/Knights-Analytics/tokenizers"
	jsoniter "github.com/json-iterator/go"
	"github.com/phuslu/log"
	ort "github.com/yalue/onnxruntime_go"
)

// types

type TextClassificationPipeline struct {
	basePipeline
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

type TextClassificationOutput [][]ClassificationOutput

// options

type TextClassificationOption func(eo *TextClassificationPipeline)

func WithAggregationFunction(aggregationFunction func([]float32) []float32) TextClassificationOption {
	return func(pipeline *TextClassificationPipeline) {
		pipeline.AggregationFunction = aggregationFunction
	}
}

// Initializes a new text classification pipeline
func NewTextClassificationPipeline(modelPath string, name string, opts ...TextClassificationOption) *TextClassificationPipeline {
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
	if util.FileExists(configPath) {
		mapBytes := util.ReadFileBytes(configPath)
		checks.Check(jsoniter.Unmarshal(mapBytes, &pipelineInputConfig))
	} else {
		log.Info().Msgf("No config.json file found for %s in the model folder at %s", pipeline.PipelineName, pipeline.ModelPath)
	}
	pipeline.IdLabelMap = pipelineInputConfig.IdLabelMap
	pipeline.PipelineTimings = &Timings{}
	pipeline.TokenizerTimings = &Timings{}

	// softmax by default

	pipeline.AggregationFunction = util.SoftMax

	// load onnx model
	pipeline.loadModel()

	// we only support single label classification for now
	pipeline.OutputDim = int(pipeline.OutputsMeta[0].Dimensions[1])
	if len(pipeline.IdLabelMap) < 1 {
		log.Fatal().Msg("Only single label classification models are currently supported and more than one label is required.")
	}

	// output dimension
	if pipeline.OutputDim <= 0 {
		log.Fatal().Msg("Pipeline configuration invalid: outputDim parameter must be greater than zero.")
	}

	if len(pipeline.IdLabelMap) <= 0 {
		log.Fatal().Msg("Pipeline configuration invalid: length of id2label map for token classification pipeline must be greater than zero.")
	}
	if len(pipeline.IdLabelMap) != pipeline.OutputDim {
		log.Fatal().Msg("Pipeline configuration invalid: length of id2label map does not match model output dimension.")
	}
	return pipeline
}

// TODO: perhaps this can be unified with the other pipelines
func (p TextClassificationPipeline) Forward(batch PipelineBatch) PipelineBatch {
	start := time.Now()

	actualBatchSize := int64(len(batch.Input))
	maxSequence := int64(batch.MaxSequence)
	inputTensors := p.getInputTensors(batch, actualBatchSize, maxSequence)
	for _, tensor := range inputTensors {
		defer func(tensor ort.ArbitraryTensor) { checks.Check(tensor.Destroy()) }(tensor)
	}

	outputTensor, err4 := ort.NewEmptyTensor[float32](ort.NewShape(actualBatchSize, int64(p.OutputDim)))
	checks.Check(err4)
	defer func() { checks.Check(outputTensor.Destroy()) }()

	// Run Onnx model
	checks.Check(p.OrtSession.Run(inputTensors, []ort.ArbitraryTensor{outputTensor}))
	batch.OutputTensor = outputTensor.GetData()

	atomic.AddUint64(&p.PipelineTimings.NumCalls, 1)
	atomic.AddUint64(&p.PipelineTimings.TotalNS, uint64(time.Since(start)))
	return batch
}

func (p TextClassificationPipeline) Postprocess(batch PipelineBatch) TextClassificationOutput {

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

	batchClassificationOutputs := make([][]ClassificationOutput, len(batch.Input))

	for i := 0; i < len(batch.Input); i++ {
		// since we only support single label classification for now there's only one classification output in the slice
		inputClassificationOutputs := make([]ClassificationOutput, 1)
		index, value, err := util.ArgMax(output[i])
		checks.Check(err)
		class, ok := p.IdLabelMap[index]
		if !ok {
			log.Fatal().Msgf("class with index number %d not found in id label map", index)
		}
		inputClassificationOutputs[0] = ClassificationOutput{
			Label: class,
			Score: value,
		}
		batchClassificationOutputs[i] = inputClassificationOutputs
	}
	return batchClassificationOutputs
}

// Run the pipeline on a string batch
func (p TextClassificationPipeline) Run(inputs []string) TextClassificationOutput {
	batch := p.Preprocess(inputs)
	batch = p.Forward(batch)
	return p.Postprocess(batch)
}
