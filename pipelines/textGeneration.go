package pipelines

import (
	"errors"
	"fmt"
	"math"
	"sync/atomic"
	"time"

	"github.com/gomlx/gomlx/types/tensors"
	jsoniter "github.com/json-iterator/go"
	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/pipelineBackends"
	"github.com/knights-analytics/hugot/util"
)

type TextGenerationPipeline struct {
	*pipelineBackends.BasePipeline
	MaxNewTokens     int
	NumKeyValueHeads int
	HeadDim          int
	NumHiddenLayers  int
	EosTokenID       []int
	OutputName       string
	Output           pipelineBackends.InputOutputInfo
}

type TextGenerationOutput struct {
	TextGenerationOutputs []string
}

type configInput struct {
	NumKeyValueHeads int   `json:"num_key_value_heads"`
	HeadDim          int   `json:"head_dim"`
	NumHiddenLayers  int   `json:"num_hidden_layers"`
	EosTokenID       []int `json:"eos_token_id"`
}

func (t *TextGenerationOutput) GetOutput() []any {
	out := make([]any, len(t.TextGenerationOutputs))
	for i, textOutput := range t.TextGenerationOutputs {
		out[i] = any(textOutput)
	}
	return out
}

// options
func WithMaxTokens(maxToken int) pipelineBackends.PipelineOption[*TextGenerationPipeline] {
	return func(pipeline *TextGenerationPipeline) {
		pipeline.MaxNewTokens = maxToken
	}
}

// NewTextGenerationPipeline initializes a new text generation pipeline
func NewTextGenerationPipeline(config pipelineBackends.PipelineConfig[*TextGenerationPipeline], s *options.Options, model *pipelineBackends.Model) (*TextGenerationPipeline, error) {
	defaultPipeline, err := pipelineBackends.NewBasePipeline(config, s, model)
	if err != nil {
		return nil, err
	}

	pipeline := &TextGenerationPipeline{BasePipeline: defaultPipeline}
	for _, o := range config.Options {
		o(pipeline)
	}

	if pipeline.MaxNewTokens <= 0 {
		pipeline.MaxNewTokens = 500 // Default value if not set
	}

	configPath := util.PathJoinSafe(model.Path, "config.json")
	pipelineInputConfig := configInput{}
	mapBytes, err := util.ReadFileBytes(configPath)
	if err != nil {
		return nil, err
	}
	err = jsoniter.Unmarshal(mapBytes, &pipelineInputConfig)
	if err != nil {
		return nil, err
	}

	pipeline.NumKeyValueHeads = pipelineInputConfig.NumKeyValueHeads
	pipeline.HeadDim = pipelineInputConfig.HeadDim
	pipeline.NumHiddenLayers = pipelineInputConfig.NumHiddenLayers
	pipeline.EosTokenID = pipelineInputConfig.EosTokenID
	return pipeline, nil
}

// INTERFACE IMPLEMENTATION

func (p *TextGenerationPipeline) GetMetadata() pipelineBackends.PipelineMetadata {
	return pipelineBackends.PipelineMetadata{}
}

func (p *TextGenerationPipeline) GetModel() *pipelineBackends.Model {
	return p.BasePipeline.Model
}

func (p *TextGenerationPipeline) GetStats() []string {
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

func (p *TextGenerationPipeline) Validate() error {
	return nil
}

func CreateCache(batchSize, numLayers, numKeyValueHeads, seqLen, headDim int) []*tensors.Tensor {
	cache := make([]*tensors.Tensor, numLayers*2)

	for layer := range numLayers {
		keyTensor := tensors.FromScalarAndDimensions(float32(0), batchSize, numKeyValueHeads, seqLen, headDim)
		cache[layer*2] = keyTensor

		valueTensor := tensors.FromScalarAndDimensions(float32(0), batchSize, numKeyValueHeads, seqLen, headDim)
		cache[layer*2+1] = valueTensor
	}
	return cache
}

// Preprocess tokenizes the input strings.
func (p *TextGenerationPipeline) Preprocess(batch *pipelineBackends.PipelineBatch, inputs []string) error {
	start := time.Now()
	pipelineBackends.TokenizeInputs(batch, p.Model.Tokenizer, inputs)
	atomic.AddUint64(&p.Model.Tokenizer.TokenizerTimings.NumCalls, 1)
	atomic.AddUint64(&p.Model.Tokenizer.TokenizerTimings.TotalNS, uint64(time.Since(start)))
	err := pipelineBackends.CreateInputTensors(batch, p.Model.InputsMeta, p.Runtime)
	cache := CreateCache(len(batch.Input), p.NumHiddenLayers, p.NumKeyValueHeads, 0, p.HeadDim)
	batch.InputValues = append(batch.InputValues.([]*tensors.Tensor), cache...) // TODO this only works for gomlx
	return err
}

func (p *TextGenerationPipeline) Forward(batch *pipelineBackends.PipelineBatch) error {
	start := time.Now()

	// ++++++++++++++++++++++++++++++++ GENERATION LOOP ++++++++++++++++++++++++++++++++
	err := pipelineBackends.RunGenerativeGoMLXSessionOnBatch(batch, p.BasePipeline)
	if err != nil {
		return err
	}
	atomic.AddUint64(&p.PipelineTimings.NumCalls, 1)
	atomic.AddUint64(&p.PipelineTimings.TotalNS, uint64(time.Since(start)))
	return nil
}

func (p *TextGenerationPipeline) Postprocess(batch *pipelineBackends.PipelineBatch) (*TextGenerationOutput, error) {
	outputValues := batch.OutputValues
	output := TextGenerationOutput{
		TextGenerationOutputs: make([]string, len(batch.Input)),
	}

	for i, val := range outputValues {
		tokenIDs := val.([]int64)
		convertedTokens := make([]uint32, len(tokenIDs))
		for j, tok := range tokenIDs {
			convertedTokens[j] = uint32(tok)
		}

		decodedString, _ := pipelineBackends.Decode(convertedTokens, p.Model.Tokenizer, true)
		output.TextGenerationOutputs[i] = decodedString
	}

	return &output, nil
}

func (p *TextGenerationPipeline) Run(inputs []string) (pipelineBackends.PipelineBatchOutput, error) {
	return p.RunPipeline(inputs)
}

func (p *TextGenerationPipeline) RunPipeline(inputs []string) (*TextGenerationOutput, error) {
	var runErrors []error
	batch := pipelineBackends.NewBatch()
	batch.MaxNewTokens = p.MaxNewTokens
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
