package pipelines

import (
	"errors"
	"fmt"
	"math"
	"strings"
	"sync/atomic"
	"time"

	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/pipelineBackends"
	"github.com/knights-analytics/hugot/util"
)

// FeatureExtractionPipeline A feature extraction pipeline is a go version of
// https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/feature_extraction.py
type FeatureExtractionPipeline struct {
	*pipelineBackends.BasePipeline
	Normalization bool
	OutputName    string
	Output        pipelineBackends.InputOutputInfo
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
func WithNormalization() pipelineBackends.PipelineOption[*FeatureExtractionPipeline] {
	return func(pipeline *FeatureExtractionPipeline) {
		pipeline.Normalization = true
	}
}

// WithOutputName if there are multiple outputs from the underlying model, which output should
// be returned. If not passed, the first output from the feature pipeline is returned.
func WithOutputName(outputName string) pipelineBackends.PipelineOption[*FeatureExtractionPipeline] {
	return func(pipeline *FeatureExtractionPipeline) {
		pipeline.OutputName = outputName
	}
}

// NewFeatureExtractionPipeline init a feature extraction pipeline.
func NewFeatureExtractionPipeline(config pipelineBackends.PipelineConfig[*FeatureExtractionPipeline], s *options.Options, model *pipelineBackends.Model) (*FeatureExtractionPipeline, error) {

	defaultPipeline, err := pipelineBackends.NewBasePipeline(config, s, model)
	if err != nil {
		return nil, err
	}

	pipeline := &FeatureExtractionPipeline{BasePipeline: defaultPipeline}
	for _, o := range config.Options {
		o(pipeline)
	}

	// filter outputs
	if pipeline.OutputName != "" {
		for _, output := range model.OutputsMeta {
			if output.Name == pipeline.OutputName {
				pipeline.Output = output
				break
			}
		}
		if pipeline.Output.Name == "" {
			return nil, fmt.Errorf("output %s is not available, outputs are: %s", pipeline.OutputName, strings.Join(pipelineBackends.GetNames(model.OutputsMeta), ", "))
		}
	} else {
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

func (p *FeatureExtractionPipeline) GetModel() *pipelineBackends.Model {
	return p.BasePipeline.Model
}

// GetMetadata returns metadata information about the pipeline, in particular:
// OutputInfo: names and dimensions of the output layer.
func (p *FeatureExtractionPipeline) GetMetadata() pipelineBackends.PipelineMetadata {
	return pipelineBackends.PipelineMetadata{
		OutputsInfo: []pipelineBackends.OutputInfo{
			{
				Name:       p.OutputName,
				Dimensions: p.Output.Dimensions,
			},
		},
	}
}

// GetStats returns the runtime statistics for the pipeline.
func (p *FeatureExtractionPipeline) GetStats() []string {
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
func (p *FeatureExtractionPipeline) Validate() error {
	var validationErrors []error

	for _, input := range p.Model.InputsMeta {
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
func (p *FeatureExtractionPipeline) Preprocess(batch *pipelineBackends.PipelineBatch, inputs []string) error {
	start := time.Now()
	pipelineBackends.TokenizeInputs(batch, p.Model.Tokenizer, inputs)
	atomic.AddUint64(&p.Model.Tokenizer.TokenizerTimings.NumCalls, 1)
	atomic.AddUint64(&p.Model.Tokenizer.TokenizerTimings.TotalNS, uint64(time.Since(start)))
	err := pipelineBackends.CreateInputTensors(batch, p.Model.InputsMeta, p.Runtime)
	return err
}

// Forward performs the forward inference of the feature extraction pipeline.
func (p *FeatureExtractionPipeline) Forward(batch *pipelineBackends.PipelineBatch) error {
	start := time.Now()
	err := pipelineBackends.RunSessionOnBatch(batch, p.BasePipeline)
	if err != nil {
		return err
	}
	atomic.AddUint64(&p.PipelineTimings.NumCalls, 1)
	atomic.AddUint64(&p.PipelineTimings.TotalNS, uint64(time.Since(start)))
	return nil
}

// Postprocess parses the first output from the network similar to the transformers' implementation.
func (p *FeatureExtractionPipeline) Postprocess(batch *pipelineBackends.PipelineBatch) (*FeatureExtractionOutput, error) {
	// TODO: this works if token embeddings are returned or sentence embeddings are returned.
	// in the former case embeddings are mean pooled. In the latter they are just returned.
	// to make this more general for other pipelines and to allow return of raw token embeddings,
	// we need an ndarray type that can be the return type of this pipeline. Need to think
	// about how to do this in a lightweight manner.

	output := batch.OutputValues[0]
	batchEmbeddings := make([][]float32, len(batch.Input))
	outputDimensions := []int64(p.Output.Dimensions)
	embeddingDimension := outputDimensions[len(outputDimensions)-1]

	if len(output.Result2D) > 0 {
		batchEmbeddings = output.Result2D
	} else if len(output.Result3D) > 0 {
		for batchIndex, tokens := range output.Result3D {
			batchEmbeddings[batchIndex] = meanPooling(tokens, batch.Input[batchIndex], batch.MaxSequenceLength, int(embeddingDimension))
		}
	} else {
		return nil, fmt.Errorf("2D output has empty result")
	}

	// Normalize embeddings (if asked), like in https://huggingface.co/sentence-transformers/all-mpnet-base-v2
	if p.Normalization {
		for i, output := range batchEmbeddings {
			batchEmbeddings[i] = util.Normalize(output, 2)
		}
	}

	return &FeatureExtractionOutput{Embeddings: batchEmbeddings}, nil
}

func meanPooling(tokens [][]float32, input pipelineBackends.TokenizedInput, maxSequence int, dimensions int) []float32 {
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
func (p *FeatureExtractionPipeline) Run(inputs []string) (pipelineBackends.PipelineBatchOutput, error) {
	return p.RunPipeline(inputs)
}

// RunPipeline is like Run, but returns the concrete feature extraction output type rather than the interface.
func (p *FeatureExtractionPipeline) RunPipeline(inputs []string) (*FeatureExtractionOutput, error) {
	var runErrors []error
	batch := pipelineBackends.NewBatch()
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
