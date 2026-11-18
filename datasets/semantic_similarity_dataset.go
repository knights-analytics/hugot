package datasets

import (
	"context"
	"errors"
	"fmt"
	"iter"
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
	source   exampleSource[SemanticSimilarityExample]
	pipeline *pipelines.FeatureExtractionPipeline
}

// SemanticSimilarityExample is a single example for the semantic similarity dataset.
type SemanticSimilarityExample struct {
	Data      map[string]any // to store any additional data for the example. Not used by the dataset.
	Sentence1 string         `json:"sentence1"`
	Sentence2 string         `json:"sentence2"`
	Score     float32        `json:"score"`
}

// ExamplePreprocessFunc can apply any custom preprocessing to a batch of examples before they are
// tokenized.
type ExamplePreprocessFunc func([]SemanticSimilarityExample) ([]SemanticSimilarityExample, error)

// NewSemanticSimilarityDataset creates a new SemanticSimilarityDataset.
// The trainingPath must be a .jsonl file where each line has the following format:
// {"sentence1":"A plane is taking off.","sentence2":"An air plane is taking off.","score":1.0}
// The score is a float value between 0 and 1.
// preprocessFunc here must be a function that takes a slice of SemanticSimilarityExample and returns a slice of SemanticSimilarityExample.
// This function can be used to apply any custom preprocessing to the example batch before they are passed to the model.
func NewSemanticSimilarityDataset(ctx context.Context, trainingPath string, batchSize int, preprocessFunc ExamplePreprocessFunc, fs fileutil.FileSystem) (*SemanticSimilarityDataset, error) {
	d := &SemanticSimilarityDataset{
		source: exampleSource[SemanticSimilarityExample]{path: trainingPath, batchSize: batchSize, preprocessFunc: preprocessFunc},
	}
	if err := d.Validate(); err != nil {
		return nil, err
	}
	if err := d.source.open(ctx, fs); err != nil {
		return nil, err
	}
	return d, nil
}

// NewInMemorySemanticSimilarityDataset creates a new SemanticSimilarityDataset in memory from a slice of examples.
// preprocessFunc here must be a function that takes a slice of SemanticSimilarityExample and returns a slice of SemanticSimilarityExample.
// This function can be used to apply any custom preprocessing to the example batch before they are passed to the model.
func NewInMemorySemanticSimilarityDataset(examples []SemanticSimilarityExample, batchSize int, preprocessFunc ExamplePreprocessFunc) (*SemanticSimilarityDataset, error) {
	d := &SemanticSimilarityDataset{
		source: exampleSource[SemanticSimilarityExample]{examples: examples, batchSize: batchSize, preprocessFunc: preprocessFunc},
	}
	if err := d.Validate(); err != nil {
		return nil, err
	}
	return d, nil
}

func (s *SemanticSimilarityDataset) Name() string {
	return fmt.Sprintf("SemanticSimilarityDataset: %s", s.source.path)
}

func (s *SemanticSimilarityDataset) SetVerbose(v bool) {
	s.source.verbose = v
}

// SetShuffle makes each pass over the dataset visit the examples in a new random order, drawn
// from seed. See [Shuffler].
func (s *SemanticSimilarityDataset) SetShuffle(seed uint64) {
	s.source.setShuffle(seed)
}

// NumBatches returns the number of batches in one pass over the dataset. See [BatchCounter].
func (s *SemanticSimilarityDataset) NumBatches() (int, error) {
	return s.source.numBatches()
}

func (s *SemanticSimilarityDataset) SetTokenizationPipeline(pipeline backends.Pipeline) error {
	if pipeline == nil {
		return fmt.Errorf("pipeline is required")
	}
	p, ok := pipeline.(*pipelines.FeatureExtractionPipeline)
	if !ok {
		return fmt.Errorf("tokenization pipeline must be a FeatureExtractionPipeline")
	}
	s.pipeline = p
	return nil
}

func (s *SemanticSimilarityDataset) Validate() error {
	return s.source.validate()
}

// Iter returns an iterator over the dataset. The examples are tokenized and converted to tensors for the training process.
func (s *SemanticSimilarityDataset) Iter() iter.Seq2[train.Batch, error] {
	return s.source.batches(s.processBatch)
}

func (s *SemanticSimilarityDataset) processBatch(exampleBatch []SemanticSimilarityExample) (train.Batch, error) {
	if s.pipeline == nil {
		return train.Batch{}, errors.New("no tokenization pipeline set; call SetTokenizationPipeline first")
	}
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
	if s.source.verbose {
		fmt.Printf("processing batch %d\n", s.source.batchN)
	}
	return train.Batch{
		Inputs: slices.Concat(inputLHS, inputRHS),
		Labels: []*tensors.Tensor{labelTensor},
	}, nil
}

// Examples reads the whole dataset and returns its examples, preprocessed if a preprocessing
// function was given. It rewinds the dataset first.
func (s *SemanticSimilarityDataset) Examples() ([]SemanticSimilarityExample, error) {
	return s.source.all()
}

// Reset resets the dataset to the beginning of the training data (after the epoch is done).
func (s *SemanticSimilarityDataset) Reset() {
	s.source.reset()
}

// YieldRaw returns the next raw batch of examples from the dataset, preprocessed if a preprocessing
// function was given. It returns io.EOF, possibly with a final partial batch, at the end of the data.
func (s *SemanticSimilarityDataset) YieldRaw() ([]SemanticSimilarityExample, error) {
	return s.source.next()
}

func (s *SemanticSimilarityDataset) Close() error {
	return s.source.close()
}
