package pipelines

import (
	"github.com/Knights-Analytics/tokenizers"
	"github.com/phuslu/log"
)

// FeatureExtractionPipeline A feature extraction pipeline is a go version of
// https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/feature_extraction.py

// types

type FeatureExtractionPipeline struct {
	BasePipeline
}

func (p *FeatureExtractionPipeline) Destroy() {
	p.Destroy()
}

// NewFeatureExtractionPipeline Initialize a feature extraction pipeline
func NewFeatureExtractionPipeline(modelPath string, name string) *FeatureExtractionPipeline {
	pipeline := &FeatureExtractionPipeline{}
	pipeline.ModelPath = modelPath
	pipeline.PipelineName = name

	// tokenizer
	pipeline.TokenizerOptions = []tokenizers.EncodeOption{tokenizers.WithReturnTypeIDs(), tokenizers.WithReturnAttentionMask()}

	pipeline.PipelineTimings = &Timings{}
	pipeline.TokenizerTimings = &Timings{}

	// load onnx model
	pipeline.loadModel()

	// the dimension of the output is taken from the output meta. For the moment we assume that there is only one output
	pipeline.OutputDim = int(pipeline.OutputsMeta[0].Dimensions[2])

	// output dimension
	if pipeline.OutputDim <= 0 {
		log.Fatal().Msg("Pipeline configuration invalid: outputDim parameter must be greater than zero.")
	}

	return pipeline
}

// Postprocess Parse the results of the forward pass into the output. Token embeddings are mean pooled.
func (p *FeatureExtractionPipeline) Postprocess(batch PipelineBatch) [][]float32 {

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
	return outputs
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
func (p *FeatureExtractionPipeline) Run(inputs []string) [][]float32 {
	batch := p.Preprocess(inputs)
	batch = p.Forward(batch)
	return p.Postprocess(batch)
}
