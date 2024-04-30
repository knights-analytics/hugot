package pipelines

import (
	"errors"

	ort "github.com/yalue/onnxruntime_go"

	util "github.com/knights-analytics/hugot/utils"
	"github.com/knights-analytics/tokenizers"
)

// FeatureExtractionPipeline A feature extraction pipeline is a go version of
// https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/feature_extraction.py

// types

type FeatureExtractionPipeline struct {
	BasePipeline
	Normalization bool
}

type FeatureExtractionPipelineConfig struct {
	IdLabelMap map[int]string `json:"id2label"`
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

// options

func WithNormalization() PipelineOption[*FeatureExtractionPipeline] {
	return func(pipeline *FeatureExtractionPipeline) {
		pipeline.Normalization = true
	}
}

// NewFeatureExtractionPipeline Initialize a feature extraction pipeline
func NewFeatureExtractionPipeline(config PipelineConfig[*FeatureExtractionPipeline], ortOptions *ort.SessionOptions) (*FeatureExtractionPipeline, error) {
	pipeline := &FeatureExtractionPipeline{}
	pipeline.ModelPath = config.ModelPath
	pipeline.PipelineName = config.Name
	pipeline.OrtOptions = ortOptions
	pipeline.OnnxFilename = config.OnnxFilename

	for _, o := range config.Options {
		o(pipeline)
	}

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

	err = pipeline.Validate()
	if err != nil {
		return nil, err
	}

	return pipeline, nil
}

func (p *FeatureExtractionPipeline) Validate() error {
	var validationErrors []error

	if p.OutputDim <= 0 {
		validationErrors = append(validationErrors, errors.New("pipeline configuration invalid: outputDim parameter must be greater than zero"))
	}
	return errors.Join(validationErrors...)
}

// Postprocess Parse the results of the forward pass into the output. Token embeddings are mean pooled.
func (p *FeatureExtractionPipeline) Postprocess(batch PipelineBatch) (*FeatureExtractionOutput, error) {
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

	// Normalize embeddings (if asked), like in https://huggingface.co/sentence-transformers/all-mpnet-base-v2
	if p.Normalization {
		for i, output := range outputs {
			outputs[i] = util.Normalize(output, 2)
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
	return p.RunPipeline(inputs)
}

func (p *FeatureExtractionPipeline) RunPipeline(inputs []string) (*FeatureExtractionOutput, error) {
	batch := p.Preprocess(inputs)
	batch, forwardError := p.Forward(batch)
	if forwardError != nil {
		return nil, forwardError
	}
	return p.Postprocess(batch)
}
