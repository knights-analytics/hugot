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

// TokenClassificationExample is a single example for the token classification dataset: a
// sentence split into words, and one tag per word. Tags are given by name, and must be names in
// the model's id2label map (for NER usually IOB tags such as "B-PER", "I-PER" and "O").
//
// The words are given either directly as Tokens, or as raw Text that the pipeline's word splitter
// (pipelines.WithWordSplitter) splits, which guarantees training splits text exactly as inference
// will. If both are given, the splitter's words must equal Tokens.
//
// Alternatively the entities are given as character Spans of Text, as span-based annotation tools
// such as Prodigy export them; see [TokenClassificationExample.Resolve] for how they become word
// tags.
type TokenClassificationExample struct {
	Data   map[string]any `json:"-"`              // to store any additional data for the example. Not used by the dataset.
	Text   string         `json:"text,omitempty"` // the raw sentence, split by the pipeline's word splitter
	Tokens []string       `json:"tokens"`         // the words of the sentence
	Tags   []string       `json:"tags"`           // one tag per word
	// Spans are the entities as character offsets into Text, instead of Tags. An empty, non-nil
	// slice is an annotated example with no entities.
	Spans []EntitySpan `json:"spans,omitempty"`
	// SourceTokens are the annotation tool's tokens of Text, used as the words when the pipeline
	// has no word splitter.
	SourceTokens []SourceToken `json:"-"`
	// Answer is the annotation answer; examples answered other than "accept" are skipped.
	Answer string `json:"answer,omitempty"`
}

// Words returns the example's words: Text split by the pipeline's word splitter when Text is set,
// otherwise Tokens. For a span-annotated example they are the words [TokenClassificationExample.Resolve]
// tags.
func (e TokenClassificationExample) Words(pipeline *pipelines.TokenClassificationPipeline) ([]string, error) {
	if e.IsSpanFormat() {
		words, _, err := e.Resolve(pipeline)
		return words, err
	}
	if e.Text == "" {
		if len(e.Tokens) == 0 {
			return nil, errors.New("example has no tokens")
		}
		return e.Tokens, nil
	}
	if pipeline == nil || pipeline.WordSplitter == nil {
		return nil, fmt.Errorf("example %q is given as text, but the pipeline has no word splitter to split it (set pipelines.WithWordSplitter, or give the words as tokens)", e.Text)
	}
	words := pipeline.WordSplitter(e.Text)
	if len(words) == 0 {
		return nil, fmt.Errorf("the word splitter returned no words for %q", e.Text)
	}
	if len(e.Tokens) > 0 && !slices.Equal(words, e.Tokens) {
		return nil, fmt.Errorf("example %q: the word splitter gives %q, which differs from its tokens %q", e.Text, words, e.Tokens)
	}
	return words, nil
}

// TokenClassificationPreprocessFunc can apply any custom preprocessing to a batch of examples before
// they are tokenized.
type TokenClassificationPreprocessFunc func([]TokenClassificationExample) ([]TokenClassificationExample, error)

// TokenClassificationDataset is a dataset for fine-tuning a token classification (e.g. NER)
// pipeline.
//
// Examples are word-level, but the model labels sub-word tokens. The words of each example are
// joined with single spaces and tokenized, and each word's tag is assigned to its first token. The
// remaining tokens of a word, the special tokens and the padding are masked out of the loss and the
// accuracy, which is the usual approach for fine-tuning NER models.
type TokenClassificationDataset struct {
	train.Dataset
	source   exampleSource[TokenClassificationExample]
	pipeline *pipelines.TokenClassificationPipeline
	labelIDs map[string]int
}

// NewTokenClassificationDataset creates a TokenClassificationDataset reading from a .jsonl file,
// where each line has the format:
//
//	{"tokens":["EU","rejects","German","call"],"tags":["B-ORG","O","B-MISC","O"]}
//
// or, when the pipeline has a word splitter, with the raw text in place of the tokens:
//
//	{"text":"EU rejects German call","tags":["B-ORG","O","B-MISC","O"]}
//
// or as character spans, in the format Prodigy exports (ner_manual); examples whose "answer" is
// not "accept" are skipped:
//
//	{"text":"EU rejects German call","spans":[{"start":0,"end":2,"label":"ORG"}],"answer":"accept"}
//
// preprocessFunc, if not nil, is applied to each batch of examples before tokenization.
func NewTokenClassificationDataset(ctx context.Context, trainingPath string, batchSize int, preprocessFunc TokenClassificationPreprocessFunc, fs fileutil.FileSystem) (*TokenClassificationDataset, error) {
	d := &TokenClassificationDataset{
		source: exampleSource[TokenClassificationExample]{path: trainingPath, batchSize: batchSize, preprocessFunc: preprocessFunc, keep: TokenClassificationExample.IsAccepted},
	}
	if err := d.Validate(); err != nil {
		return nil, err
	}
	if err := d.source.open(ctx, fs); err != nil {
		return nil, err
	}
	return d, nil
}

// NewInMemoryTokenClassificationDataset creates a TokenClassificationDataset from a slice of examples.
func NewInMemoryTokenClassificationDataset(examples []TokenClassificationExample, batchSize int, preprocessFunc TokenClassificationPreprocessFunc) (*TokenClassificationDataset, error) {
	d := &TokenClassificationDataset{
		source: exampleSource[TokenClassificationExample]{examples: acceptedExamples(examples), batchSize: batchSize, preprocessFunc: preprocessFunc},
	}
	if err := d.Validate(); err != nil {
		return nil, err
	}
	return d, nil
}

func (s *TokenClassificationDataset) Name() string {
	return fmt.Sprintf("TokenClassificationDataset: %s", s.source.path)
}

func (s *TokenClassificationDataset) SetVerbose(v bool) {
	s.source.verbose = v
}

// SetTokenizationPipeline sets the pipeline whose tokenizer and id2label map the dataset uses.
func (s *TokenClassificationDataset) SetTokenizationPipeline(pipeline backends.Pipeline) error {
	if pipeline == nil {
		return fmt.Errorf("pipeline is required")
	}
	p, ok := pipeline.(*pipelines.TokenClassificationPipeline)
	if !ok {
		return fmt.Errorf("tokenization pipeline must be a TokenClassificationPipeline")
	}
	if len(p.Model.IDLabelMap) == 0 {
		return fmt.Errorf("the model has no id2label map, so tags cannot be resolved")
	}
	labelIDs := make(map[string]int, len(p.Model.IDLabelMap))
	for id, label := range p.Model.IDLabelMap {
		labelIDs[label] = id
	}
	s.pipeline = p
	s.labelIDs = labelIDs
	return nil
}

func (s *TokenClassificationDataset) Validate() error {
	return s.source.validate()
}

// Iter returns an iterator over the dataset. The examples are tokenized and converted to tensors for the training process.
func (s *TokenClassificationDataset) Iter() iter.Seq2[train.Batch, error] {
	return s.source.batches(s.processBatch)
}

func (s *TokenClassificationDataset) processBatch(exampleBatch []TokenClassificationExample) (train.Batch, error) {
	if s.pipeline == nil {
		return train.Batch{}, errors.New("no tokenization pipeline set; call SetTokenizationPipeline first")
	}
	texts := make([]string, len(exampleBatch))
	words := make([][]string, len(exampleBatch))
	tags := make([][]string, len(exampleBatch))
	for i, example := range exampleBatch {
		exampleWords, exampleTags, err := example.Resolve(s.pipeline)
		if err != nil {
			return train.Batch{}, err
		}
		words[i], tags[i] = exampleWords, exampleTags
		texts[i] = strings.Join(exampleWords, " ")
	}
	batch := backends.NewBatch(len(exampleBatch))
	backends.TokenizeInputs(batch, s.pipeline.Model.Tokenizer, texts)
	if err := backends.CreateInputTensorsTraining(batch, s.pipeline.Model); err != nil {
		return train.Batch{}, err
	}
	inputs := batch.InputValues.([]*tensors.Tensor)
	// The label tensors must match the padded sequence length of the input tensors.
	sequenceLength := inputs[0].Shape().Dimensions[1]
	labels, mask, err := s.alignTags(words, tags, batch.Input, sequenceLength)
	if err != nil {
		return train.Batch{}, err
	}
	if s.source.verbose {
		fmt.Printf("processing batch %d\n", s.source.batchN)
	}
	return train.Batch{
		Inputs: inputs,
		Labels: []*tensors.Tensor{
			tensors.FromFlatDataAndDimensions(labels, len(exampleBatch), sequenceLength, 1),
			tensors.FromFlatDataAndDimensions(mask, len(exampleBatch), sequenceLength),
		},
	}, nil
}

// alignTags assigns each word's tag id to the first token of the word, and returns the flat
// [batch, sequenceLength] tag ids and the mask of the tokens that carry a tag.
func (s *TokenClassificationDataset) alignTags(words, tags [][]string, tokenized []backends.TokenizedInput, sequenceLength int) ([]int32, []bool, error) {
	labels := make([]int32, len(words)*sequenceLength)
	mask := make([]bool, len(words)*sequenceLength)
	for i := range words {
		firstTokens := WordStartTokens(words[i], tokenized[i], sequenceLength)
		for w, token := range firstTokens {
			if token < 0 {
				// The word produced no token (e.g. it was truncated away), so it cannot be trained on.
				continue
			}
			id, err := s.tagID(tags[i][w])
			if err != nil {
				return nil, nil, err
			}
			labels[i*sequenceLength+token] = int32(id) // #nosec G115 -- ids come from the model's id2label map
			mask[i*sequenceLength+token] = true
		}
	}
	return labels, mask, nil
}

// WordStartTokens returns, for each word, the index of its first non-special token among the first
// sequenceLength tokens, or -1 if it has none. input is the tokenization of the words joined with
// single spaces; its offsets are byte offsets into that string.
func WordStartTokens(words []string, input backends.TokenizedInput, sequenceLength int) []int {
	first := make([]int, len(words))
	for w := range first {
		first[w] = -1
	}
	starts := make([]uint, len(words))
	ends := make([]uint, len(words))
	var pos uint
	for w, word := range words {
		starts[w] = pos
		ends[w] = pos + uint(len(word))
		pos = ends[w] + 1
	}
	w := 0
	for t := 0; t < len(input.Offsets) && t < sequenceLength; t++ {
		if t < len(input.SpecialTokensMask) && input.SpecialTokensMask[t] != 0 {
			continue
		}
		start := input.Offsets[t][0]
		for w < len(words) && start >= ends[w] { // the token is past this word (or the word is empty)
			w++
		}
		if w == len(words) {
			break
		}
		if start >= starts[w] && start < ends[w] && first[w] < 0 {
			first[w] = t
		}
	}
	return first
}

func (s *TokenClassificationDataset) tagID(tag string) (int, error) {
	id, ok := s.labelIDs[tag]
	if !ok {
		known := make([]string, 0, len(s.labelIDs))
		for name := range s.labelIDs {
			known = append(known, name)
		}
		slices.Sort(known)
		return 0, fmt.Errorf("tag %q is not in the model's id2label map (%s)", tag, strings.Join(known, ", "))
	}
	return id, nil
}

// Examples reads the whole dataset and returns its examples, preprocessed if a preprocessing
// function was given. It rewinds the dataset first.
func (s *TokenClassificationDataset) Examples() ([]TokenClassificationExample, error) {
	return s.source.all()
}

// Reset resets the dataset to the beginning of the training data (after the epoch is done).
func (s *TokenClassificationDataset) Reset() {
	s.source.reset()
}

// YieldRaw returns the next raw batch of examples from the dataset, preprocessed if a preprocessing
// function was given. It returns io.EOF, possibly with a final partial batch, at the end of the data.
func (s *TokenClassificationDataset) YieldRaw() ([]TokenClassificationExample, error) {
	return s.source.next()
}

func (s *TokenClassificationDataset) Close() error {
	return s.source.close()
}

// acceptedExamples drops the examples whose annotation answer is not "accept".
func acceptedExamples(examples []TokenClassificationExample) []TokenClassificationExample {
	return slices.DeleteFunc(slices.Clone(examples), func(e TokenClassificationExample) bool { return !e.IsAccepted() })
}
