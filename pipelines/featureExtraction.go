package pipelines

import (
	"errors"

	"github.com/knights-analytics/tokenizers"
)

// FeatureExtractionPipeline A feature extraction pipeline is a go version of
// https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/feature_extraction.py

type FeatureExtractionPipeline struct {
	BasePipeline
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

// NewFeatureExtractionPipeline Initialize a feature extraction pipeline
func NewFeatureExtractionPipeline(modelPath string, name string) (*FeatureExtractionPipeline, error) {
	pipeline := &FeatureExtractionPipeline{}
	pipeline.ModelPath = modelPath
	pipeline.PipelineName = name

	// tokenizer
	pipeline.TokenizerOptions = []tokenizers.EncodeOption{tokenizers.WithReturnTypeIDs(), tokenizers.WithReturnAttentionMask()}

	pipeline.PipelineTimings = &Timings{}
	pipeline.TokenizerTimings = &Timings{}

	// load onnx model
	err := pipeline.loadModel()
	if err != nil {
		return nil, err
	}

	// the dimension of the output is taken from the output meta. For the moment we assume that there is only one output
	pipeline.OutputDim = int(pipeline.OutputsMeta[0].Dimensions[2])

	// output dimension
	if pipeline.OutputDim <= 0 {
		return nil, errors.New("pipeline configuration invalid: outputDim parameter must be greater than zero")
	}

	return pipeline, nil
}

// Postprocess Parse the results of the forward pass into the output. Token embeddings are mean pooled.
func (p *FeatureExtractionPipeline) Postprocess(batch PipelineBatch) (PipelineBatchOutput, error) {

	maxSequence := batch.MaxSequence
	vectorCounter := 0
	tokenCounter := 0
	inputCounter := 0
	outputs := make([][]float32, len(batch.Input))
	tokens := make([][]float32, maxSequence)
	vectors := make([]float32, p.OutputDim)

	for _, result := range batch.OutputTensor {
		vectors[vectorCounter] = result
		if vectorCounter == p.OutputDim-1 {
			tokens[tokenCounter] = vectors
			vectorCounter = 0
			vectors = make([]float32, p.OutputDim)
			if tokenCounter == maxSequence-1 {
				outputs[inputCounter] = meanPooling(tokens, batch.Input[inputCounter], maxSequence, p.OutputDim)
				tokenCounter = 0
				tokens = make([][]float32, maxSequence)
				inputCounter++
			} else {
				tokenCounter++
			}
		} else {
			vectorCounter++
		}
	}
	return &FeatureExtractionOutput{Embeddings: outputs}, nil
}

func meanPooling(tokens [][]float32, input TokenizedInput, maxSequence int, dimensions int) []float32 {

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

// Run the pipeline on a string batch
func (p *FeatureExtractionPipeline) Run(inputs []string) (PipelineBatchOutput, error) {
	batch := p.Preprocess(inputs)
	batch, forwardError := p.Forward(batch)
	if forwardError != nil {
		return nil, forwardError
	}
	return p.Postprocess(batch)
}
