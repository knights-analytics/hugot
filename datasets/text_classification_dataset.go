package datasets

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"slices"
	"strings"

	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/pipelines"
	"github.com/knights-analytics/hugot/util/fileutil"
)

// ClassificationProblem says whether each example has exactly one label or any number of them.
type ClassificationProblem int

const (
	// SingleLabel examples have exactly one label each, set in TextClassificationExample.Label.
	// Training uses softmax cross-entropy.
	SingleLabel ClassificationProblem = iota
	// MultiLabel examples have any number of labels each, set in TextClassificationExample.Labels.
	// Training uses sigmoid (binary) cross-entropy per label.
	MultiLabel
)

func (p ClassificationProblem) String() string {
	switch p {
	case SingleLabel:
		return "single-label"
	case MultiLabel:
		return "multi-label"
	default:
		return fmt.Sprintf("ClassificationProblem(%d)", int(p))
	}
}

// TextClassificationExample is a single example for the text classification dataset. Labels are
// given by name, and must be names in the model's id2label map.
//
// Examples exported by Prodigy are read too: choice annotations give their labels in Accept, and
// binary annotations give one Label with an Answer. See [TextClassificationExample.IsChoiceFormat].
type TextClassificationExample struct {
	Data   map[string]any `json:"-"`                // to store any additional data for the example. Not used by the dataset.
	Text   string         `json:"text"`             // the input text
	Label  string         `json:"label,omitempty"`  // the label, for a SingleLabel dataset
	Labels []string       `json:"labels,omitempty"` // the labels, for a MultiLabel dataset; may be empty
	// Accept holds the labels chosen in a choice annotation (Prodigy textcat.manual). A non-nil
	// empty slice is a choice annotation with no label chosen.
	Accept []string `json:"accept,omitempty"`
	// Answer is the annotation answer: "accept", "reject" or "ignore". With a single Label it is a
	// binary annotation of whether that label applies.
	Answer string `json:"answer,omitempty"`
}

// TextClassificationPreprocessFunc can apply any custom preprocessing to a batch of examples before
// they are tokenized.
type TextClassificationPreprocessFunc func([]TextClassificationExample) ([]TextClassificationExample, error)

// TextClassificationDataset is a dataset for fine-tuning a text classification pipeline.
type TextClassificationDataset struct {
	train.Dataset
	source   exampleSource[TextClassificationExample]
	pipeline *pipelines.TextClassificationPipeline
	labelIDs map[string]int
	problem  ClassificationProblem
}

// NewTextClassificationDataset creates a TextClassificationDataset reading from a .jsonl file.
// For a SingleLabel dataset each line has the format:
//
//	{"text":"The film was a delight.","label":"POSITIVE"}
//
// and for a MultiLabel dataset:
//
//	{"text":"Late, and the wrong size.","labels":["delivery","sizing"]}
//
// Prodigy exports are read as well, and can be mixed with the lines above. A choice annotation
// (textcat.manual with several labels) gives the chosen labels, exactly one for a SingleLabel
// dataset:
//
//	{"text":"Late, and the wrong size.","accept":["delivery","sizing"],"answer":"accept"}
//
// A binary annotation (textcat.manual with one label, or textcat.teach) says whether one label
// applies:
//
//	{"text":"The film was a delight.","label":"NEGATIVE","answer":"reject"}
//
// Accepting a label makes it the class of a SingleLabel example. Rejecting one only says the text
// is not that class, which is a full label for a two-class model (the other class) and is skipped
// otherwise. A MultiLabel dataset can use binary annotations only when the model has that single
// label, since the other labels are unknown. Examples answered "ignore", and rejected choice
// annotations, are skipped.
//
// Label names must exist in the id2label map of the model being trained. preprocessFunc, if not
// nil, is applied to each batch of examples before tokenization.
func NewTextClassificationDataset(ctx context.Context, trainingPath string, batchSize int, problem ClassificationProblem, preprocessFunc TextClassificationPreprocessFunc, fs fileutil.FileSystem) (*TextClassificationDataset, error) {
	d := &TextClassificationDataset{
		source:  exampleSource[TextClassificationExample]{path: trainingPath, batchSize: batchSize, preprocessFunc: preprocessFunc},
		problem: problem,
	}
	d.source.keep = d.keepExample
	if err := d.Validate(); err != nil {
		return nil, err
	}
	if err := d.source.open(ctx, fs); err != nil {
		return nil, err
	}
	return d, nil
}

// NewInMemoryTextClassificationDataset creates a TextClassificationDataset from a slice of examples.
func NewInMemoryTextClassificationDataset(examples []TextClassificationExample, batchSize int, problem ClassificationProblem, preprocessFunc TextClassificationPreprocessFunc) (*TextClassificationDataset, error) {
	d := &TextClassificationDataset{
		source:  exampleSource[TextClassificationExample]{examples: examples, batchSize: batchSize, preprocessFunc: preprocessFunc},
		problem: problem,
	}
	if err := d.Validate(); err != nil {
		return nil, err
	}
	return d, nil
}

// Problem reports whether the dataset is single-label or multi-label.
func (s *TextClassificationDataset) Problem() ClassificationProblem {
	return s.problem
}

func (s *TextClassificationDataset) Name() string {
	return fmt.Sprintf("TextClassificationDataset: %s", s.source.path)
}

func (s *TextClassificationDataset) SetVerbose(v bool) {
	s.source.verbose = v
}

// SetTokenizationPipeline sets the pipeline whose tokenizer and id2label map the dataset uses.
func (s *TextClassificationDataset) SetTokenizationPipeline(pipeline backends.Pipeline) error {
	if pipeline == nil {
		return fmt.Errorf("pipeline is required")
	}
	p, ok := pipeline.(*pipelines.TextClassificationPipeline)
	if !ok {
		return fmt.Errorf("tokenization pipeline must be a TextClassificationPipeline")
	}
	if len(p.Model.IDLabelMap) == 0 {
		return fmt.Errorf("the model has no id2label map, so labels cannot be resolved")
	}
	labelIDs := make(map[string]int, len(p.Model.IDLabelMap))
	for id, label := range p.Model.IDLabelMap {
		labelIDs[label] = id
	}
	s.pipeline = p
	s.labelIDs = labelIDs
	if len(s.source.examples) > 0 {
		// The model's labels decide which binary rejections can be used, so in-memory examples
		// are filtered now that they are known.
		kept := s.source.examples[:0:0]
		for _, example := range s.source.examples {
			if s.keepExample(example) {
				kept = append(kept, example)
			}
		}
		if len(kept) == 0 {
			return errors.New("no example can be trained on: all were ignored or rejected")
		}
		s.source.examples = kept
	}
	return nil
}

// keepExample reports whether an example carries a usable label. Examples with an error are kept,
// so the error surfaces when the batch is built.
func (s *TextClassificationDataset) keepExample(example TextClassificationExample) bool {
	_, keep, err := example.resolve(s.problem, s.labelNames())
	return keep || err != nil
}

// labelNames returns the model's labels in id order, or nil before the pipeline is set.
func (s *TextClassificationDataset) labelNames() []string {
	if s.pipeline == nil {
		return nil
	}
	return LabelNames(s.pipeline.Model.IDLabelMap)
}

// LabelNames returns the labels of an id2label map in id order.
func LabelNames(idLabelMap map[int]string) []string {
	ids := make([]int, 0, len(idLabelMap))
	for id := range idLabelMap {
		ids = append(ids, id)
	}
	slices.Sort(ids)
	names := make([]string, len(ids))
	for i, id := range ids {
		names[i] = idLabelMap[id]
	}
	return names
}

// Resolve returns the example in the dataset's own form, Label or Labels set from a Prodigy
// annotation if it is one, or keep false if the example should be skipped. labels are the model's
// labels.
func (e TextClassificationExample) Resolve(problem ClassificationProblem, labels []string) (resolved TextClassificationExample, keep bool, err error) {
	return e.resolve(problem, labels)
}

func (s *TextClassificationDataset) Validate() error {
	if s.problem != SingleLabel && s.problem != MultiLabel {
		return fmt.Errorf("unknown classification problem %v", s.problem)
	}
	return s.source.validate()
}

// Iter returns an iterator over the dataset. The examples are tokenized and converted to tensors for the training process.
func (s *TextClassificationDataset) Iter() iter.Seq2[train.Batch, error] {
	return s.source.batches(s.processBatch)
}

func (s *TextClassificationDataset) processBatch(exampleBatch []TextClassificationExample) (train.Batch, error) {
	if s.pipeline == nil {
		return train.Batch{}, errors.New("no tokenization pipeline set; call SetTokenizationPipeline first")
	}
	texts := make([]string, len(exampleBatch))
	resolved := make([]TextClassificationExample, 0, len(exampleBatch))
	for _, example := range exampleBatch {
		example, keep, err := example.resolve(s.problem, s.labelNames())
		if err != nil {
			return train.Batch{}, err
		}
		if !keep {
			return train.Batch{}, fmt.Errorf("example %q should have been skipped before batching", example.Text)
		}
		resolved = append(resolved, example)
	}
	for i, example := range resolved {
		texts[i] = example.Text
	}
	labelTensor, err := s.labelTensor(resolved)
	if err != nil {
		return train.Batch{}, err
	}
	batch := backends.NewBatch(len(exampleBatch))
	backends.TokenizeInputs(batch, s.pipeline.Model.Tokenizer, texts)
	if err := backends.CreateInputTensorsTraining(batch, s.pipeline.Model); err != nil {
		return train.Batch{}, err
	}
	if s.source.verbose {
		fmt.Printf("processing batch %d\n", s.source.batchN)
	}
	return train.Batch{
		Inputs: batch.InputValues.([]*tensors.Tensor),
		Labels: []*tensors.Tensor{labelTensor},
	}, nil
}

// labelTensor returns the batch labels: class ids of shape [batch, 1] for a SingleLabel dataset,
// and multi-hot vectors of shape [batch, labels] for a MultiLabel one.
func (s *TextClassificationDataset) labelTensor(exampleBatch []TextClassificationExample) (*tensors.Tensor, error) {
	if s.problem == SingleLabel {
		ids := make([]int32, len(exampleBatch))
		for i, example := range exampleBatch {
			if len(example.Labels) > 0 {
				return nil, fmt.Errorf("example %q sets labels, but the dataset is single-label; use label", example.Text)
			}
			id, err := s.labelID(example.Label)
			if err != nil {
				return nil, err
			}
			ids[i] = int32(id) // #nosec G115 -- ids come from the model's id2label map
		}
		return tensors.FromFlatDataAndDimensions(ids, len(ids), 1), nil
	}
	nLabels := len(s.labelIDs)
	multiHot := make([]float32, len(exampleBatch)*nLabels)
	for i, example := range exampleBatch {
		if example.Label != "" {
			return nil, fmt.Errorf("example %q sets label, but the dataset is multi-label; use labels", example.Text)
		}
		for _, label := range example.Labels {
			id, err := s.labelID(label)
			if err != nil {
				return nil, err
			}
			multiHot[i*nLabels+id] = 1
		}
	}
	return tensors.FromFlatDataAndDimensions(multiHot, len(exampleBatch), nLabels), nil
}

func (s *TextClassificationDataset) labelID(label string) (int, error) {
	id, ok := s.labelIDs[label]
	if !ok {
		known := make([]string, 0, len(s.labelIDs))
		for name := range s.labelIDs {
			known = append(known, name)
		}
		slices.Sort(known)
		return 0, fmt.Errorf("label %q is not in the model's id2label map (%s)", label, strings.Join(known, ", "))
	}
	return id, nil
}

// Examples reads the whole dataset and returns its examples, preprocessed if a preprocessing
// function was given. It rewinds the dataset first.
func (s *TextClassificationDataset) Examples() ([]TextClassificationExample, error) {
	return s.source.all()
}

// Reset resets the dataset to the beginning of the training data (after the epoch is done).
func (s *TextClassificationDataset) Reset() {
	s.source.reset()
}

// YieldRaw returns the next raw batch of examples from the dataset, preprocessed if a preprocessing
// function was given. It returns io.EOF, possibly with a final partial batch, at the end of the data.
func (s *TextClassificationDataset) YieldRaw() ([]TextClassificationExample, error) {
	return s.source.next()
}

func (s *TextClassificationDataset) Close() error {
	return s.source.close()
}
