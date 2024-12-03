//go:build XLA || ALL

package datasets

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
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
}

// SemanticSimilarityDataset is a dataset for fine-tuning a feature extraction pipeline for textual semantic similarity.
type SemanticSimilarityDataset struct {
	train.Dataset
	TrainingPath string
	pipeline     *pipelines.FeatureExtractionPipeline
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
	Sentence1 string  `json:"sentence1"`
	Sentence2 string  `json:"sentence2"`
	Score     float32 `json:"label"`
}

// NewSemanticSimilarityDataset creates a new SemanticSimilarityDataset.
// The trainingPath must be a .jsonl file where each line has the following format:
// {"sentence1":"A plane is taking off.","sentence2":"An air plane is taking off.","score":1.0}
// The score is a float value between 0 and 1.
func NewSemanticSimilarityDataset(trainingPath string) (*SemanticSimilarityDataset, error) {
	d := &SemanticSimilarityDataset{
		TrainingPath: trainingPath,
	}
	if err := d.Validate(); err != nil {
		return nil, err
	}
	return d, nil
}

func (s *SemanticSimilarityDataset) Yield() (spec any, inputs []*tensors.Tensor, labels []*tensors.Tensor, err error) {
	jsonBytes, err := util.ReadFileBytes(s.TrainingPath)
	if err != nil {
		return nil, nil, nil, err
	}

	scanner := bufio.NewScanner(bytes.NewReader(jsonBytes))

	inputsLeft := []string{}
	inputsRight := []string{}
	scores := []float32{}

	// nLines := 0 // just 20 examples

	for scanner.Scan() {
		// if nLines >= 20 {
		// 	break
		// }
		var lineData SemanticSimilarityExample
		if err := json.Unmarshal(scanner.Bytes(), &lineData); err != nil {
			return nil, nil, nil, fmt.Errorf("failed to parse JSON line: %w", err)
		} else {
			inputsLeft = append(inputsLeft, lineData.Sentence1)
			inputsRight = append(inputsRight, lineData.Sentence2)
			scores = append(scores, lineData.Score)
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, nil, nil, fmt.Errorf("error reading file: %w", err)
	}

	batchLeft := pipelineBackends.NewBatch()
	batchRight := pipelineBackends.NewBatch()

	pipelineBackends.TokenizeInputs(batchLeft, s.pipeline.Model.Tokenizer, inputsLeft)
	pipelineBackends.TokenizeInputs(batchRight, s.pipeline.Model.Tokenizer, inputsRight)

	if err := pipelineBackends.CreateInputTensors(batchLeft, s.pipeline.Model.InputsMeta, s.pipeline.Runtime); err != nil {
		return nil, nil, nil, err
	}
	if err := pipelineBackends.CreateInputTensors(batchRight, s.pipeline.Model.InputsMeta, s.pipeline.Runtime); err != nil {
		return nil, nil, nil, err
	}
	inputLeft := batchLeft.InputValues.([]*tensors.Tensor)
	inputRight := batchRight.InputValues.([]*tensors.Tensor)
	labelTensor := tensors.FromFlatDataAndDimensions(scores, len(scores), 1)
	return nil, slices.Concat(inputLeft, inputRight), []*tensors.Tensor{labelTensor}, nil
}
