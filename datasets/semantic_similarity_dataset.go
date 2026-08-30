package datasets

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"path/filepath"
	"slices"

	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/pipelines"
	"github.com/knights-analytics/hugot/util/fileutil"
)

// SemanticSimilarityDataset is a dataset for fine-tuning a feature extraction pipeline for textual semantic similarity.
type SemanticSimilarityDataset struct {
	train.Dataset
	sourceFile       io.ReadCloser
	preprocessFunc   ExamplePreprocessFunc
	pipeline         *pipelines.FeatureExtractionPipeline
	reader           *bufio.Reader
	fsContext        context.Context
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

func (s *SemanticSimilarityDataset) SetVerbose(v bool) {
	s.verbose = v
}

func (s *SemanticSimilarityDataset) SetTokenizationPipeline(pipeline backends.Pipeline) error {
	if pipeline == nil {
		return fmt.Errorf("pipeline is required")
	}
	if _, ok := pipeline.(*pipelines.FeatureExtractionPipeline); !ok {
		return fmt.Errorf("tokenization pipeline must be a FeatureExtractionPipeline")
	}
	s.pipeline = pipeline.(*pipelines.FeatureExtractionPipeline)
	return nil
}

func (s *SemanticSimilarityDataset) Validate() error {
	if len(s.trainingExamples) == 0 {
		if s.trainingPath == "" {
			return fmt.Errorf("training path is required")
		}
		if filepath.Ext(s.trainingPath) != ".jsonl" {
			return fmt.Errorf("training path must be a .jsonl file")
		}
	}
	return nil
}

// SemanticSimilarityExample is a single example for the semantic similarity dataset.
type SemanticSimilarityExample struct {
	Data      map[string]any // to store any additional data for the example. Not used by the dataset.
	Sentence1 string         `json:"sentence1"`
	Sentence2 string         `json:"sentence2"`
	Score     float32        `json:"score"`
}
type ExamplePreprocessFunc func([]SemanticSimilarityExample) ([]SemanticSimilarityExample, error)

// NewSemanticSimilarityDataset creates a new SemanticSimilarityDataset.
// The trainingPath must be a .jsonl file where each line has the following format:
// {"sentence1":"A plane is taking off.","sentence2":"An air plane is taking off.","score":1.0}
// The score is a float value between 0 and 1.
// preprocessFunc here must be a function that takes a slice of SemanticSimilarityExample and returns a slice of SemanticSimilarityExample.
// This function can be used to apply any custom preprocessing to the example batch before they are passed to the model.
func NewSemanticSimilarityDataset(ctx context.Context, trainingPath string, batchSize int, preprocessFunc ExamplePreprocessFunc, fs fileutil.FileSystem) (*SemanticSimilarityDataset, error) {
	d := &SemanticSimilarityDataset{
		trainingPath:   trainingPath,
		batchSize:      batchSize,
		preprocessFunc: preprocessFunc,
		fsContext:      fileutil.WithFileSystem(ctx, fs),
	}
	if err := d.Validate(); err != nil {
		return nil, err
	}
	sourceReadCloser, err := fileutil.OpenFile(d.fsContext, trainingPath)
	if err != nil {
		return nil, err
	}
	d.reader = bufio.NewReader(sourceReadCloser)
	d.sourceFile = sourceReadCloser
	return d, nil
}

// NewInMemorySemanticSimilarityDataset creates a new SemanticSimilarityDataset in memory from a slice of examples.
// preprocessFunc here must be a function that takes a slice of SemanticSimilarityExample and returns a slice of SemanticSimilarityExample.
// This function can be used to apply any custom preprocessing to the example batch before they are passed to the model.
func NewInMemorySemanticSimilarityDataset(examples []SemanticSimilarityExample, batchSize int, preprocessFunc ExamplePreprocessFunc) (*SemanticSimilarityDataset, error) {
	d := &SemanticSimilarityDataset{
		trainingExamples: examples,
		batchSize:        batchSize,
		preprocessFunc:   preprocessFunc,
	}
	if err := d.Validate(); err != nil {
		return nil, err
	}
	return d, nil
}

// Reset resets the dataset to the beginning of the training data (after the epoch is done).
func (s *SemanticSimilarityDataset) Reset() {
	if s.verbose {
		fmt.Printf("completed epoch in %d batches of %d examples, resetting dataset\n", s.batchN, s.batchSize)
	}
	s.batchN = 0
	if len(s.trainingExamples) == 0 {
		if err := s.sourceFile.Close(); err != nil {
			panic(err)
		}
		sourceReadCloser, err := fileutil.OpenFile(s.fsContext, s.trainingPath)
		if err != nil {
			panic(err) // note: these panics will be caught later with the TryExcept
		}
		s.sourceFile = sourceReadCloser
		// restart the reader
		s.reader = bufio.NewReader(sourceReadCloser)
	}
}

// YieldRaw returns the next raw batch of examples from the dataset. Note that if a preprocessing function has been
// provided at creation time, the examples will be preprocessed before being returned.
func (s *SemanticSimilarityDataset) YieldRaw() ([]SemanticSimilarityExample, error) {
	batchCounter := 0
	var lineBytes []byte
	var readErr error
	var lineData SemanticSimilarityExample
	examplesBatch := make([]SemanticSimilarityExample, 0, s.batchSize)
	var preprocessErr error
	for batchCounter < s.batchSize {
		if len(s.trainingExamples) > 0 {
			// in memory dataset
			start := s.batchN * s.batchSize
			if start >= len(s.trainingExamples) {
				return examplesBatch, io.EOF // return error for reset
			}
			end := start + s.batchSize
			for i := start; i < end && i < len(s.trainingExamples); i++ {
				examplesBatch = append(examplesBatch, s.trainingExamples[i])
			}
		} else {
			lineBytes, readErr = fileutil.ReadLine(s.reader)
			if readErr != nil {
				return examplesBatch, readErr
			}
			if err := json.Unmarshal(lineBytes, &lineData); err != nil {
				return nil, fmt.Errorf("failed to parse JSON line: %w", err)
			}
			examplesBatch = append(examplesBatch, lineData)
		}
		batchCounter++
	}
	s.batchN++
	if s.preprocessFunc != nil {
		examplesBatch, preprocessErr = s.preprocessFunc(examplesBatch)
		if preprocessErr != nil {
			return nil, preprocessErr
		}
	}
	return examplesBatch, nil
}

func (s *SemanticSimilarityDataset) Close() error {
	if s.sourceFile != nil {
		return s.sourceFile.Close()
	}
	return nil
}
