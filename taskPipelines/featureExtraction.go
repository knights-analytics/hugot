package taskPipelines

import (
	"errors"
	"fmt"
	"math"
	"strings"
	"sync/atomic"
	"time"

	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/pipelines"

	"github.com/knights-analytics/hugot/util"
)

// FeatureExtractionPipeline A feature extraction pipeline is a go version of
// https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/feature_extraction.py
type FeatureExtractionPipeline struct {
	*pipelines.BasePipeline
	Normalization bool
	OutputName    string
	Output        pipelines.InputOutputInfo
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
func WithNormalization() pipelines.PipelineOption[*FeatureExtractionPipeline] {
	return func(pipeline *FeatureExtractionPipeline) {
		pipeline.Normalization = true
	}
}

// WithOutputName if there are multiple outputs from the underlying model, which output should
// be returned. If not passed, the first output from the feature pipeline is returned.
func WithOutputName(outputName string) pipelines.PipelineOption[*FeatureExtractionPipeline] {
	return func(pipeline *FeatureExtractionPipeline) {
		pipeline.OutputName = outputName
	}
}

// NewFeatureExtractionPipeline init a feature extraction pipeline.
func NewFeatureExtractionPipeline(config pipelines.PipelineConfig[*FeatureExtractionPipeline], s *options.Options) (*FeatureExtractionPipeline, error) {

	defaultPipeline, err := pipelines.NewBasePipeline(config, s)
	if err != nil {
		return nil, err
	}

	pipeline := &FeatureExtractionPipeline{BasePipeline: defaultPipeline}
	for _, o := range config.Options {
		o(pipeline)
	}

	// filter outputs
	if pipeline.OutputName != "" {
		for _, output := range pipeline.OutputsMeta {
			if output.Name == pipeline.OutputName {
				pipeline.Output = output
				break
			}
		}
		if pipeline.Output.Name == "" {
			return nil, fmt.Errorf("output %s is not available, outputs are: %s", pipeline.OutputName, strings.Join(pipelines.GetNames(pipeline.OutputsMeta), ", "))
		}
	} else {
		pipeline.Output = pipeline.OutputsMeta[0] // we take the first output otherwise, like transformers does
	}

	// validate pipeline
	err = pipeline.Validate()
	if err != nil {
		errDestroy := pipeline.Destroy()
		return nil, errors.Join(err, errDestroy)
	}
	return pipeline, nil
}

// INTERFACE IMPLEMENTATION

// GetMetadata returns metadata information about the pipeline, in particular:
// OutputInfo: names and dimensions of the output layer.
func (p *FeatureExtractionPipeline) GetMetadata() pipelines.PipelineMetadata {
	return pipelines.PipelineMetadata{
		OutputsInfo: []pipelines.OutputInfo{
			{
				Name:       p.OutputName,
				Dimensions: p.Output.Dimensions,
			},
		},
	}
}

// Destroy frees the pipeline resources.
func (p *FeatureExtractionPipeline) Destroy() error {
	return p.BasePipeline.Destroy()
}

// GetStats returns the runtime statistics for the pipeline.
func (p *FeatureExtractionPipeline) GetStats() []string {
	return []string{
		fmt.Sprintf("Statistics for pipeline: %s", p.PipelineName),
		fmt.Sprintf("Tokenizer: Total time=%s, Execution count=%d, Average query time=%s",
			time.Duration(p.Tokenizer.TokenizerTimings.TotalNS),
			p.Tokenizer.TokenizerTimings.NumCalls,
			time.Duration(float64(p.Tokenizer.TokenizerTimings.TotalNS)/math.Max(1, float64(p.Tokenizer.TokenizerTimings.NumCalls)))),
		fmt.Sprintf("ONNX: Total time=%s, Execution count=%d, Average query time=%s",
			time.Duration(p.PipelineTimings.TotalNS),
			p.PipelineTimings.NumCalls,
			time.Duration(float64(p.PipelineTimings.TotalNS)/math.Max(1, float64(p.PipelineTimings.NumCalls)))),
	}
}

// Validate checks that the pipeline is valid.
func (p *FeatureExtractionPipeline) Validate() error {
	var validationErrors []error

	for _, input := range p.InputsMeta {
		dims := []int64(input.Dimensions)
		if len(dims) > 3 {
			validationErrors = append(validationErrors, fmt.Errorf("inputs and outputs currently can have at most 3 dimensions"))
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
func (p *FeatureExtractionPipeline) Preprocess(batch *pipelines.PipelineBatch, inputs []string) error {
	start := time.Now()
	pipelines.TokenizeInputs(batch, p.Tokenizer, inputs)
	atomic.AddUint64(&p.Tokenizer.TokenizerTimings.NumCalls, 1)
	atomic.AddUint64(&p.Tokenizer.TokenizerTimings.TotalNS, uint64(time.Since(start)))
	err := pipelines.CreateInputTensors(batch, p.InputsMeta, p.Runtime)
	return err
}

// Forward performs the forward inference of the feature extraction pipeline.
func (p *FeatureExtractionPipeline) Forward(batch *pipelines.PipelineBatch) error {
	start := time.Now()
	err := pipelines.RunSessionOnBatch(batch, p.BasePipeline)
	if err != nil {
		return err
	}
	atomic.AddUint64(&p.PipelineTimings.NumCalls, 1)
	atomic.AddUint64(&p.PipelineTimings.TotalNS, uint64(time.Since(start)))
	return nil
}

// Postprocess parses the first output from the network similar to the transformers' implementation.
func (p *FeatureExtractionPipeline) Postprocess(batch *pipelines.PipelineBatch) (*FeatureExtractionOutput, error) {
	// TODO: this works if token embeddings are returned or sentence embeddings are returned.
	// in the former case embeddings are mean pooled. In the latter they are just returned.
	// to make this more general for other pipelines and to allow return of raw token embeddings,
	// we need an ndarray type that can be the return type of this pipeline. Need to think
	// about how to do this in a lightweight manner.

	batchEmbeddings := make([][]float32, len(batch.Input))
	outputDimensions := []int64(p.Output.Dimensions)
	embeddingDimension := outputDimensions[len(outputDimensions)-1]
	maxSequenceLength := batch.MaxSequenceLength

	// now take the output slice and gather the results as a "matrix"
	outputEmbedding := make([]float32, embeddingDimension)
	outputEmbeddingCounter := 0
	tokenEmbeddings := make([][]float32, maxSequenceLength)
	tokenEmbeddingsCounter := 0
	batchInputCounter := 0
	outputTensor := batch.OutputValues[0]

	for _, result := range outputTensor {
		outputEmbedding[outputEmbeddingCounter] = result
		if outputEmbeddingCounter == int(embeddingDimension)-1 {
			// we gathered one embedding
			if len(outputDimensions) <= 2 {
				// it is already a sentence embedding, just add it to batch outputs
				batchEmbeddings[batchInputCounter] = outputEmbedding
				outputEmbedding = make([]float32, embeddingDimension)
				batchInputCounter++
			} else {
				// output is embedding for a token, add to token embeddings
				tokenEmbeddings[tokenEmbeddingsCounter] = outputEmbedding
				outputEmbedding = make([]float32, embeddingDimension)
				if tokenEmbeddingsCounter == maxSequenceLength-1 {
					// computed all embeddings for the tokens, calculate sentence embedding, add to batch outputs, and reset token embeddings and counter
					batchEmbeddings[batchInputCounter] = meanPooling(tokenEmbeddings, batch.Input[batchInputCounter], maxSequenceLength, int(embeddingDimension))
					tokenEmbeddings = make([][]float32, maxSequenceLength)
					tokenEmbeddingsCounter = 0
					batchInputCounter++
				} else {
					// still more tokens to go
					tokenEmbeddingsCounter++
				}
			}
			outputEmbeddingCounter = 0
		} else {
			// still more elements of the embedding to go
			outputEmbeddingCounter++
		}
	}

	// Normalize embeddings (if asked), like in https://huggingface.co/sentence-transformers/all-mpnet-base-v2
	if p.Normalization {
		for i, output := range batchEmbeddings {
			batchEmbeddings[i] = util.Normalize(output, 2)
		}
	}

	return &FeatureExtractionOutput{Embeddings: batchEmbeddings}, nil
}

func meanPooling(tokens [][]float32, input pipelines.TokenizedInput, maxSequence int, dimensions int) []float32 {
	length := len(input.AttentionMask)
	vector := make([]float32, dimensions)
	for j := 0; j < maxSequence; j++ {
		if j+1 <= length && input.AttentionMask[j] != 0 {
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
func (p *FeatureExtractionPipeline) Run(inputs []string) (pipelines.PipelineBatchOutput, error) {
	return p.RunPipeline(inputs)
}

// RunPipeline is like Run, but returns the concrete feature extraction output type rather than the interface.
func (p *FeatureExtractionPipeline) RunPipeline(inputs []string) (*FeatureExtractionOutput, error) {
	var runErrors []error
	batch := pipelines.NewBatch()
	defer func(*pipelines.PipelineBatch) {
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