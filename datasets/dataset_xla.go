//go:build XLA || ALL

package datasets

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"slices"

	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/types/tensors"

	"github.com/knights-analytics/hugot/pipelineBackends"
	"github.com/knights-analytics/hugot/pipelines"
)

type Dataset interface {
	train.Dataset
	Validate() error
	SetTokenizationPipeline(pipeline pipelineBackends.Pipeline) error
	SetVerbose(bool)
	Close() error
}

// SemanticSimilarityDataset is a dataset for fine-tuning a feature extraction pipeline for textual semantic similarity.
type SemanticSimilarityDataset struct {
	train.Dataset
	trainingPath     string
	trainingExamples []SemanticSimilarityExample
	preprocessFunc   ExamplePreprocessFunc
	batchSize        int
	pipeline         *pipelines.FeatureExtractionPipeline
	reader           *bufio.Reader
	sourceFile       io.ReadCloser
	batchN           int
	verbose          bool
}

// Yield returns the next batch of examples from the dataset. The examples are tokenized and converted to tensors for the training process.
func (s *SemanticSimilarityDataset) Yield() (spec any, inputs []*tensors.Tensor, labels []*tensors.Tensor, err error) {
	exampleBatch, rawErr := s.YieldRaw()
	if rawErr != nil && !errors.Is(rawErr, io.EOF) {
		return nil, nil, nil, err
	}

	var inputLhs []*tensors.Tensor
	var inputRhs []*tensors.Tensor
	var labelTensor []*tensors.Tensor

	if len(exampleBatch) > 0 {
		batchLhs := pipelineBackends.NewBatch()
		batchRhs := pipelineBackends.NewBatch()

		inputsLhs := make([]string, 0, len(exampleBatch))
		inputsRhs := make([]string, 0, len(exampleBatch))
		scores := make([]float32, 0, len(exampleBatch))
		for _, example := range exampleBatch {
			inputsLhs = append(inputsLhs, example.Sentence1)
			inputsRhs = append(inputsRhs, example.Sentence2)
			scores = append(scores, example.Score)
		}

		pipelineBackends.TokenizeInputs(batchLhs, s.pipeline.Model.Tokenizer, inputsLhs)
		pipelineBackends.TokenizeInputs(batchRhs, s.pipeline.Model.Tokenizer, inputsRhs)

		if err := pipelineBackends.CreateInputTensorsTraining(batchLhs, s.pipeline.Model.InputsMeta, s.pipeline.Runtime); err != nil {
			return nil, nil, nil, err
		}
		if err := pipelineBackends.CreateInputTensorsTraining(batchRhs, s.pipeline.Model.InputsMeta, s.pipeline.Runtime); err != nil {
			return nil, nil, nil, err
		}
		inputLhs = batchLhs.InputValues.([]*tensors.Tensor)
		inputRhs = batchRhs.InputValues.([]*tensors.Tensor)
		labelTensor = []*tensors.Tensor{tensors.FromFlatDataAndDimensions(scores, len(scores), 1)}

		if s.verbose {
			fmt.Printf("processing batch %d\n", s.batchN)
		}
	}

	return nil, slices.Concat(inputLhs, inputRhs), labelTensor, rawErr
}
