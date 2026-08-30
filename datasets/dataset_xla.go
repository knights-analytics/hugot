package datasets

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"iter"
	"slices"

	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/train"
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

func (s *SemanticSimilarityDataset) Name() string {
	return fmt.Sprintf("SemanticSimilarityDataset: %s", s.trainingPath)
}

// Iter returns an iterator over the dataset. The examples are tokenized and converted to tensors for the training process.
func (s *SemanticSimilarityDataset) Iter() iter.Seq2[train.Batch, error] {
	return func(yield func(train.Batch, error) bool) {
		s.Reset()
		for {
			exampleBatch, rawErr := s.YieldRaw()
			if rawErr != nil {
				if errors.Is(rawErr, io.EOF) {
					if len(exampleBatch) > 0 {
						batch, err := s.processBatch(exampleBatch)
						if err != nil {
							_ = yield(train.Batch{}, err)
							return
						}
						if !yield(batch, nil) {
							return
						}
					}
					return
				}
				_ = yield(train.Batch{}, rawErr)
				return
			}
			if len(exampleBatch) > 0 {
				batch, err := s.processBatch(exampleBatch)
				if err != nil {
					_ = yield(train.Batch{}, err)
					return
				}
				if !yield(batch, nil) {
					return
				}
			}
		}
	}
}

func (s *SemanticSimilarityDataset) processBatch(exampleBatch []SemanticSimilarityExample) (train.Batch, error) {
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
	if err := backends.CreateInputTensorsTraining(batchLHS, s.pipeline.Model); err != nil {
		return train.Batch{}, err
	}
	if err := backends.CreateInputTensorsTraining(batchRHS, s.pipeline.Model); err != nil {
		return train.Batch{}, err
	}
	inputLHS := batchLHS.InputValues.([]*tensors.Tensor)
	inputRHS := batchRHS.InputValues.([]*tensors.Tensor)
	labelTensor := tensors.FromFlatDataAndDimensions(scores, len(scores), 1)
	if s.verbose {
		fmt.Printf("processing batch %d\n", s.batchN)
	}
	inputs := slices.Concat(inputLHS, inputRHS)
	labels := []*tensors.Tensor{labelTensor}
	return train.Batch{
		Inputs: inputs,
		Labels: labels,
	}, nil
}
