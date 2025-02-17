//go:build XLA || ALL

package datasets

import (
	"bufio"
	"encoding/json"
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
	TrainingPath string
	BatchSize    int
	pipeline     *pipelines.FeatureExtractionPipeline
	reader       *bufio.Reader
	sourceFile   io.ReadCloser
	batchN       int
	verbose      bool
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
	if s.TrainingPath == "" {
		return fmt.Errorf("training path is required")
	}
	if filepath.Ext(s.TrainingPath) != ".jsonl" {
		return fmt.Errorf("training path must be a .jsonl file")
	}
	return nil
}

type SemanticSimilarityExample struct {
	Sentence1 *string  `json:"sentence1"`
	Sentence2 *string  `json:"sentence2"`
	Score     *float32 `json:"score"`
}

// NewSemanticSimilarityDataset creates a new SemanticSimilarityDataset.
// The trainingPath must be a .jsonl file where each line has the following format:
// {"sentence1":"A plane is taking off.","sentence2":"An air plane is taking off.","score":1.0}
// The score is a float value between 0 and 1.
func NewSemanticSimilarityDataset(trainingPath string, batchSize int) (*SemanticSimilarityDataset, error) {
	d := &SemanticSimilarityDataset{
		TrainingPath: trainingPath,
		BatchSize:    batchSize,
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

func (s *SemanticSimilarityDataset) Reset() {
	if s.verbose {
		fmt.Printf("completed epoch in %d batches of %d examples, resetting dataset\n", s.batchN, s.BatchSize)
	}
	s.batchN = 0
	if err := s.sourceFile.Close(); err != nil {
		panic(err)
	}

	sourceReadCloser, err := util.OpenFile(s.TrainingPath) // TODO how to handle errors here
	if err != nil {
		panic(err)
	}
	s.sourceFile = sourceReadCloser

	// restart the reader
	s.reader = bufio.NewReader(sourceReadCloser)
}

func (s *SemanticSimilarityDataset) Yield() (spec any, inputs []*tensors.Tensor, labels []*tensors.Tensor, err error) {

	var inputsLeft []string
	var inputsRight []string
	var scores []float32

	batchCounter := 0

	var lineBytes []byte
	var readErr error
	var lineData SemanticSimilarityExample

	for batchCounter < s.BatchSize {
		lineBytes, readErr = util.ReadLine(s.reader)
		if readErr == io.EOF {
			if batchCounter == 0 {
				return nil, nil, nil, io.EOF // return error for reset
			} else {
				break // batch was cut short but we still process what is left
			}
		}

		if e := json.Unmarshal(lineBytes, &lineData); e != nil {
			return nil, nil, nil, fmt.Errorf("failed to parse JSON line: %w", e)
		}
		if lineData.Sentence1 == nil || lineData.Sentence2 == nil || lineData.Score == nil {
			return nil, nil, nil, fmt.Errorf("missing required fields in JSON line")
		}
		inputsLeft = append(inputsLeft, *lineData.Sentence1)
		inputsRight = append(inputsRight, *lineData.Sentence2)
		scores = append(scores, *lineData.Score)
		batchCounter++
	}

	batchLeft := pipelineBackends.NewBatch()
	batchRight := pipelineBackends.NewBatch()

	pipelineBackends.TokenizeInputs(batchLeft, s.pipeline.Model.Tokenizer, inputsLeft)
	pipelineBackends.TokenizeInputs(batchRight, s.pipeline.Model.Tokenizer, inputsRight)

	if err := pipelineBackends.CreateInputTensorsTraining(batchLeft, s.pipeline.Model.InputsMeta, s.pipeline.Runtime); err != nil {
		return nil, nil, nil, err
	}
	if err := pipelineBackends.CreateInputTensorsTraining(batchRight, s.pipeline.Model.InputsMeta, s.pipeline.Runtime); err != nil {
		return nil, nil, nil, err
	}
	inputLeft := batchLeft.InputValues.([]*tensors.Tensor)
	inputRight := batchRight.InputValues.([]*tensors.Tensor)
	labelTensor := tensors.FromFlatDataAndDimensions(scores, len(scores), 1)

	if s.verbose {
		fmt.Printf("processing batch %d\n", s.batchN)
	}
	s.batchN++
	return nil, slices.Concat(inputLeft, inputRight), []*tensors.Tensor{labelTensor}, nil
}

func (s *SemanticSimilarityDataset) Close() error {
	if s.sourceFile != nil {
		return s.sourceFile.Close()
	}
	return nil
}
