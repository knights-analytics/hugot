package datasets

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"slices"

	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/train"
	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/pipelines"
)

type Dataset interface {
	train.Dataset
	Validate() error
	SetTokenizationPipeline(pipeline backends.Pipeline) error
	SetVerbose(bool)
	Close() error
}

// SemanticSimilarityDataset is a dataset for fine-tuning a feature extraction pipeline for textual semantic similarity.
type SemanticSimilarityDataset struct {
	train.Dataset
	sourceFile       io.ReadCloser
	preprocessFunc   ExamplePreprocessFunc
	pipeline         *pipelines.FeatureExtractionPipeline
	reader           *bufio.Reader
	trainingPath     string
	trainingExamples []SemanticSimilarityExample
	batchSize        int
	batchN           int
	verbose          bool
}

// Yield returns the next batch of examples from the dataset. The examples are tokenized and converted to tensors for the training process.
func (s *SemanticSimilarityDataset) Yield() (spec any, inputs []*tensors.Tensor, labels []*tensors.Tensor, err error) {
	exampleBatch, rawErr := s.YieldRaw()
	if rawErr != nil && !errors.Is(rawErr, io.EOF) {
		return nil, nil, nil, err
	}
	var inputLHS []*tensors.Tensor
	var inputRHS []*tensors.Tensor
	var labelTensor []*tensors.Tensor
	if len(exampleBatch) > 0 {
		batchLHS := backends.NewBatch(len(exampleBatch))
		batchRHS := backends.NewBatch(len(exampleBatch))
		inputsLHS := make([]string, 0, len(exampleBatch))
		inputsRHS := make([]string, 0, len(exampleBatch))
		scores := make([]float32, 0, len(exampleBatch))
		for _, example := range exampleBatch {
			inputsLHS = append(inputsLHS, example.Sentence1)
			inputsRHS = append(inputsRHS, example.Sentence2)
			scores = append(scores, example.Score)
		}
		backends.TokenizeInputs(batchLHS, s.pipeline.Model.Tokenizer, inputsLHS)
		backends.TokenizeInputs(batchRHS, s.pipeline.Model.Tokenizer, inputsRHS)
		if err := backends.CreateInputTensorsTraining(batchLHS, s.pipeline.Model, s.pipeline.Runtime); err != nil {
			return nil, nil, nil, err
		}
		if err := backends.CreateInputTensorsTraining(batchRHS, s.pipeline.Model, s.pipeline.Runtime); err != nil {
			return nil, nil, nil, err
		}
		inputLHS = batchLHS.InputValues.([]*tensors.Tensor)
		inputRHS = batchRHS.InputValues.([]*tensors.Tensor)
		labelTensor = []*tensors.Tensor{tensors.FromFlatDataAndDimensions(scores, len(scores), 1)}
		if s.verbose {
			fmt.Printf("processing batch %d\n", s.batchN)
		}
	}
	return nil, slices.Concat(inputLHS, inputRHS), labelTensor, rawErr
}
