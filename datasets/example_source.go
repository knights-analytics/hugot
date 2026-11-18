package datasets

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"math/rand/v2"
	"path/filepath"

	"github.com/gomlx/gomlx/ml/train"
	"github.com/knights-analytics/hugot/util/fileutil"
)

// exampleSource yields batches of examples of type E, either from a .jsonl file (one example per
// line) or from a slice held in memory. It is the shared reading half of the datasets.
type exampleSource[E any] struct {
	sourceFile     io.ReadCloser
	reader         *bufio.Reader
	fsContext      context.Context
	preprocessFunc func([]E) ([]E, error)
	keep           func(E) bool // if set, file examples for which it returns false are skipped
	skipped        int          // examples skipped by keep in the current pass
	path           string
	examples       []E
	batchSize      int
	batchN         int
	verbose        bool

	// Shuffling, when enabled with setShuffle. An in-memory source is permuted at each reset; a file
	// source is streamed through a buffer of up to shuffleBufferSize examples, drawn at random.
	rng        *rand.Rand
	order      []int // the current pass's permutation of examples, in memory
	buffer     []E   // examples read from the file and not yet yielded
	sourceDone bool  // the file has been read to the end in the current pass
	ordered    bool  // set while all() reads, which never shuffles
}

// shuffleBufferSize is how many examples a shuffled file source holds in memory to draw from. A
// file of up to this many examples is shuffled uniformly, in the same order as the same examples
// held in memory with the same seed; a longer one is shuffled locally, each
// example landing at most this many places from where a uniform shuffle would put it on average.
const shuffleBufferSize = 10_000

// Shuffler is implemented by the datasets that can visit their examples in a random order. The
// trainer shuffles its training dataset through it; see hugot.WithSeed and hugot.WithoutShuffle.
type Shuffler interface {
	// SetShuffle makes each pass over the dataset visit the examples in a new random order, drawn
	// from seed: the same seed gives the same sequence of orders. Examples read with Examples are
	// always in their original order.
	SetShuffle(seed uint64)
}

// setShuffle enables shuffling with a random sequence drawn from seed.
func (s *exampleSource[E]) setShuffle(seed uint64) {
	s.rng = rand.New(rand.NewPCG(seed, seed)) // #nosec G404 -- reproducible data order, not security
}

func (s *exampleSource[E]) shuffling() bool {
	return s.rng != nil && !s.ordered
}

func (s *exampleSource[E]) validate() error {
	if s.batchSize <= 0 {
		return fmt.Errorf("batch size must be greater than 0, got %d", s.batchSize)
	}
	if len(s.examples) == 0 {
		if s.path == "" {
			return fmt.Errorf("training path is required")
		}
		if filepath.Ext(s.path) != ".jsonl" {
			return fmt.Errorf("training path must be a .jsonl file")
		}
	}
	return nil
}

// open opens the source file; an in-memory source needs no opening.
func (s *exampleSource[E]) open(ctx context.Context, fs fileutil.FileSystem) error {
	if len(s.examples) > 0 {
		return nil
	}
	s.fsContext = fileutil.WithFileSystem(ctx, fs)
	sourceReadCloser, err := fileutil.OpenFile(s.fsContext, s.path)
	if err != nil {
		return err
	}
	s.sourceFile = sourceReadCloser
	s.reader = bufio.NewReader(sourceReadCloser)
	return nil
}

// reset rewinds to the first example, reopening the file.
func (s *exampleSource[E]) reset() {
	if s.verbose {
		fmt.Printf("completed epoch in %d batches of %d examples, resetting dataset\n", s.batchN, s.batchSize)
		if s.skipped > 0 {
			fmt.Printf("skipped %d examples\n", s.skipped)
		}
	}
	s.batchN = 0
	s.skipped = 0
	s.order = nil
	s.buffer = s.buffer[:0]
	s.sourceDone = false
	if s.shuffling() && len(s.examples) > 0 {
		s.order = s.rng.Perm(len(s.examples))
	}
	if len(s.examples) == 0 {
		if err := s.sourceFile.Close(); err != nil {
			panic(err)
		}
		sourceReadCloser, err := fileutil.OpenFile(s.fsContext, s.path)
		if err != nil {
			panic(err) // note: these panics will be caught later with the TryExcept
		}
		s.sourceFile = sourceReadCloser
		s.reader = bufio.NewReader(sourceReadCloser)
	}
}

// next returns the next batch of examples, preprocessed if a preprocessing function was given. It
// returns io.EOF, possibly with a final partial batch, at the end of the data.
func (s *exampleSource[E]) next() ([]E, error) {
	batch := make([]E, 0, s.batchSize)
	var endErr error
	switch {
	case len(s.examples) > 0:
		start := s.batchN * s.batchSize
		if start >= len(s.examples) {
			return batch, io.EOF
		}
		end := min(start+s.batchSize, len(s.examples))
		if s.order != nil {
			for _, i := range s.order[start:end] {
				batch = append(batch, s.examples[i])
			}
		} else {
			batch = append(batch, s.examples[start:end]...)
		}
	case s.shuffling():
		if s.batchN == 0 {
			// A file that fits in the buffer is permuted whole, with the same draw as the same
			// examples held in memory, so the two train in the same order.
			if err := s.fillBuffer(); err != nil {
				return nil, err
			}
			if s.sourceDone {
				s.order = s.rng.Perm(len(s.buffer))
			}
		}
		if s.order != nil {
			start := s.batchN * s.batchSize
			if start >= len(s.order) {
				return batch, io.EOF
			}
			for _, i := range s.order[start:min(start+s.batchSize, len(s.order))] {
				batch = append(batch, s.buffer[i])
			}
			break
		}
		for len(batch) < s.batchSize {
			if err := s.fillBuffer(); err != nil {
				return nil, err
			}
			if len(s.buffer) == 0 {
				endErr = io.EOF
				break
			}
			i := s.rng.IntN(len(s.buffer))
			batch = append(batch, s.buffer[i])
			last := len(s.buffer) - 1
			s.buffer[i] = s.buffer[last]
			s.buffer = s.buffer[:last]
		}
	default:
		for len(batch) < s.batchSize {
			example, err := s.read()
			if errors.Is(err, io.EOF) {
				endErr = io.EOF
				break
			}
			if err != nil {
				return nil, err
			}
			batch = append(batch, example)
		}
	}
	s.batchN++
	if s.preprocessFunc != nil && len(batch) > 0 {
		var err error
		if batch, err = s.preprocessFunc(batch); err != nil {
			return nil, err
		}
	}
	return batch, endErr
}

// read returns the next example from the file that keep accepts, or io.EOF at the end of the file.
func (s *exampleSource[E]) read() (E, error) {
	var zero E
	for {
		lineBytes, err := fileutil.ReadLine(s.reader)
		if err != nil {
			return zero, err
		}
		var example E
		if err = json.Unmarshal(lineBytes, &example); err != nil {
			return zero, fmt.Errorf("failed to parse JSON line: %w", err)
		}
		if s.keep != nil && !s.keep(example) {
			s.skipped++
			continue
		}
		return example, nil
	}
}

// fillBuffer tops the shuffle buffer up from the file, until it is full or the file is exhausted.
func (s *exampleSource[E]) fillBuffer() error {
	for !s.sourceDone && len(s.buffer) < shuffleBufferSize {
		example, err := s.read()
		if errors.Is(err, io.EOF) {
			s.sourceDone = true
			return nil
		}
		if err != nil {
			return err
		}
		s.buffer = append(s.buffer, example)
	}
	return nil
}

// all rewinds and returns every example in its original order, preprocessed if a preprocessing
// function was given. It does not draw from the shuffle sequence, so reading the examples does not
// change the order of the next training pass.
func (s *exampleSource[E]) all() ([]E, error) {
	s.ordered = true
	defer func() { s.ordered = false }()
	s.reset()
	var examples []E
	for {
		batch, err := s.next()
		examples = append(examples, batch...)
		if errors.Is(err, io.EOF) {
			return examples, nil
		}
		if err != nil {
			return nil, err
		}
	}
}

func (s *exampleSource[E]) close() error {
	if s.sourceFile != nil {
		return s.sourceFile.Close()
	}
	return nil
}

// batches iterates over one epoch of the source, turning each batch of examples into a training
// batch with process.
func (s *exampleSource[E]) batches(process func([]E) (train.Batch, error)) iter.Seq2[train.Batch, error] {
	return func(yield func(train.Batch, error) bool) {
		s.reset()
		for {
			examples, rawErr := s.next()
			if rawErr != nil && !errors.Is(rawErr, io.EOF) {
				_ = yield(train.Batch{}, rawErr)
				return
			}
			if len(examples) > 0 {
				batch, err := process(examples)
				if err != nil {
					_ = yield(train.Batch{}, err)
					return
				}
				if !yield(batch, nil) {
					return
				}
			}
			if rawErr != nil { // io.EOF
				return
			}
		}
	}
}
