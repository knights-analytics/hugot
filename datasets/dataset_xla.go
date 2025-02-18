//go:build XLA || ALL

package datasets

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"path/filepath"
	"slices"

	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/types/tensors"

	"github.com/knights-analytics/hugot/pipelineBackends"
	"github.com/knights-analytics/hugot/pipelines"
	"github.com/knights-analytics/hugot/util"
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
	batchSize        int
	pipeline         *pipelines.FeatureExtractionPipeline
	reader           *bufio.Reader
	sourceFile       io.ReadCloser
	batchN           int
	verbose          bool
}

func (s *SemanticSimilarityDataset) SetVerbose(v bool) {
	s.verbose = v
}

func (s *SemanticSimilarityDataset) SetTokenizationPipeline(pipeline pipelineBackends.Pipeline) error {
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

type SemanticSimilarityExample struct {
	Sentence1 string  `json:"sentence1"`
	Sentence2 string  `json:"sentence2"`
	Score     float32 `json:"score"`
}

// NewSemanticSimilarityDataset creates a new SemanticSimilarityDataset.
// The trainingPath must be a .jsonl file where each line has the following format:
// {"sentence1":"A plane is taking off.","sentence2":"An air plane is taking off.","score":1.0}
// The score is a float value between 0 and 1.
func NewSemanticSimilarityDataset(trainingPath string, batchSize int) (*SemanticSimilarityDataset, error) {
	d := &SemanticSimilarityDataset{
		trainingPath: trainingPath,
		batchSize:    batchSize,
	}
	if err := d.Validate(); err != nil {
		return nil, err
	}

	sourceReadCloser, err := util.OpenFile(trainingPath)
	if err != nil {
		return nil, err
	}
	d.reader = bufio.NewReader(sourceReadCloser)
	d.sourceFile = sourceReadCloser
	return d, nil
}

func NewInMemorySemanticSimilarityDataset(examples []SemanticSimilarityExample, batchSize int) (*SemanticSimilarityDataset, error) {
	d := &SemanticSimilarityDataset{
		trainingExamples: examples,
		batchSize:        batchSize,
	}
	if err := d.Validate(); err != nil {
		return nil, err
	}
	return d, nil
}

func (s *SemanticSimilarityDataset) Reset() {
	if s.verbose {
		fmt.Printf("completed epoch in %d batches of %d examples, resetting dataset\n", s.batchN, s.batchSize)
	}
	s.batchN = 0

	if len(s.trainingExamples) == 0 {
		if err := s.sourceFile.Close(); err != nil {
			panic(err)
		}

		sourceReadCloser, err := util.OpenFile(s.trainingPath)
		if err != nil {
			panic(err) // note: these panics will be catched later with the TryExcept
		}
		s.sourceFile = sourceReadCloser

		// restart the reader
		s.reader = bufio.NewReader(sourceReadCloser)
	}
}

func (s *SemanticSimilarityDataset) YieldRaw() ([]SemanticSimilarityExample, error) {
	batchCounter := 0

	var lineBytes []byte
	var readErr error
	var lineData SemanticSimilarityExample

	examplesBatch := make([]SemanticSimilarityExample, 0, s.batchSize)

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
			lineBytes, readErr = util.ReadLine(s.reader)
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
	return examplesBatch, nil
}

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

func (s *SemanticSimilarityDataset) Close() error {
	if s.sourceFile != nil {
		return s.sourceFile.Close()
	}
	return nil
}
