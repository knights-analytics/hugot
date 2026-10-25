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

// ContrastiveExample is a single example for the contrastive dataset: an anchor (e.g. a query), a
// positive that matches it (e.g. the passage that answers it), and optionally a hard negative that
// looks related but does not match.
type ContrastiveExample struct {
	Data     map[string]any `json:"-"`                  // to store any additional data for the example. Not used by the dataset.
	Anchor   string         `json:"anchor"`             // e.g. a query
	Positive string         `json:"positive"`           // the text that matches the anchor
	Negative string         `json:"negative,omitempty"` // an optional hard negative
}

// ContrastivePreprocessFunc can apply any custom preprocessing to a batch of examples before they
// are tokenized.
type ContrastivePreprocessFunc func([]ContrastiveExample) ([]ContrastiveExample, error)

// ContrastiveDataset is a dataset for fine-tuning a feature extraction (embedding) pipeline with
// in-batch negatives, the multiple negatives ranking loss of sentence-transformers.
//
// Within a batch, each anchor must pick out its own positive among all the positives of the batch
// (and all its hard negatives, if the examples have them). Only matching pairs are needed, no
// similarity scores. Larger batches give more negatives and usually better embeddings. Two examples
// in one batch whose positives both fit the other's anchor are taught to differ, so avoid duplicate
// or near-duplicate pairs within a batch.
type ContrastiveDataset struct {
	train.Dataset
	source   exampleSource[ContrastiveExample]
	pipeline *pipelines.FeatureExtractionPipeline
}

// NewContrastiveDataset creates a ContrastiveDataset reading from a .jsonl file, where each line has
// the format:
//
//	{"anchor":"how do I reset my password?","positive":"To reset your password, open Settings…"}
//
// with an optional "negative". Either every example has a negative or none does. batchSize must be
// at least 2, so each anchor has at least one other positive to tell its own apart from.
func NewContrastiveDataset(ctx context.Context, trainingPath string, batchSize int, preprocessFunc ContrastivePreprocessFunc, fs fileutil.FileSystem) (*ContrastiveDataset, error) {
	d := &ContrastiveDataset{
		source: exampleSource[ContrastiveExample]{path: trainingPath, batchSize: batchSize, preprocessFunc: preprocessFunc},
	}
	if err := d.Validate(); err != nil {
		return nil, err
	}
	if err := d.source.open(ctx, fs); err != nil {
		return nil, err
	}
	return d, nil
}

// NewInMemoryContrastiveDataset creates a ContrastiveDataset from a slice of examples.
func NewInMemoryContrastiveDataset(examples []ContrastiveExample, batchSize int, preprocessFunc ContrastivePreprocessFunc) (*ContrastiveDataset, error) {
	d := &ContrastiveDataset{
		source: exampleSource[ContrastiveExample]{examples: examples, batchSize: batchSize, preprocessFunc: preprocessFunc},
	}
	if err := d.Validate(); err != nil {
		return nil, err
	}
	return d, nil
}

func (s *ContrastiveDataset) Name() string {
	return fmt.Sprintf("ContrastiveDataset: %s", s.source.path)
}

func (s *ContrastiveDataset) SetVerbose(v bool) {
	s.source.verbose = v
}

// SetTokenizationPipeline sets the pipeline whose tokenizer the dataset uses.
func (s *ContrastiveDataset) SetTokenizationPipeline(pipeline backends.Pipeline) error {
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

func (s *ContrastiveDataset) Validate() error {
	if s.source.batchSize < 2 {
		return fmt.Errorf("batch size must be at least 2 for in-batch negatives, got %d", s.source.batchSize)
	}
	return s.source.validate()
}

// Iter returns an iterator over the dataset. The examples are tokenized and converted to tensors for the training process.
func (s *ContrastiveDataset) Iter() iter.Seq2[train.Batch, error] {
	return s.source.batches(s.processBatch)
}

// processBatch tokenizes the anchors, the positives and any negatives, each as one group of model
// inputs, concatenated in that order. The label of anchor i is i, the index of its positive among
// the candidates (the positives, then the negatives).
func (s *ContrastiveDataset) processBatch(exampleBatch []ContrastiveExample) (train.Batch, error) {
	if s.pipeline == nil {
		return train.Batch{}, errors.New("no tokenization pipeline set; call SetTokenizationPipeline first")
	}
	withNegatives := exampleBatch[0].Negative != ""
	anchors := make([]string, len(exampleBatch))
	positives := make([]string, len(exampleBatch))
	var negatives []string
	for i, example := range exampleBatch {
		if example.Anchor == "" || example.Positive == "" {
			return train.Batch{}, fmt.Errorf("example %d has an empty anchor or positive", i)
		}
		if (example.Negative != "") != withNegatives {
			return train.Batch{}, errors.New("either every example has a negative or none does")
		}
		anchors[i] = example.Anchor
		positives[i] = example.Positive
		if withNegatives {
			negatives = append(negatives, example.Negative)
		}
	}
	groups := [][]string{anchors, positives}
	if withNegatives {
		groups = append(groups, negatives)
	}
	var inputs []*tensors.Tensor
	for _, texts := range groups {
		batch := backends.NewBatch(len(texts))
		backends.TokenizeInputs(batch, s.pipeline.Model.Tokenizer, texts)
		if err := backends.CreateInputTensorsTraining(batch, s.pipeline.Model); err != nil {
			return train.Batch{}, err
		}
		inputs = slices.Concat(inputs, batch.InputValues.([]*tensors.Tensor))
	}
	targets := make([]int32, len(exampleBatch))
	for i := range targets {
		targets[i] = int32(i) // #nosec G115 -- bounded by the batch size
	}
	if s.source.verbose {
		fmt.Printf("processing batch %d\n", s.source.batchN)
	}
	return train.Batch{
		Inputs: inputs,
		Labels: []*tensors.Tensor{tensors.FromFlatDataAndDimensions(targets, len(targets), 1)},
	}, nil
}

// Examples reads the whole dataset and returns its examples, preprocessed if a preprocessing
// function was given. It rewinds the dataset first.
func (s *ContrastiveDataset) Examples() ([]ContrastiveExample, error) {
	return s.source.all()
}

// Reset resets the dataset to the beginning of the training data (after the epoch is done).
func (s *ContrastiveDataset) Reset() {
	s.source.reset()
}

// YieldRaw returns the next raw batch of examples from the dataset, preprocessed if a preprocessing
// function was given. It returns io.EOF, possibly with a final partial batch, at the end of the data.
func (s *ContrastiveDataset) YieldRaw() ([]ContrastiveExample, error) {
	return s.source.next()
}

func (s *ContrastiveDataset) Close() error {
	return s.source.close()
}
