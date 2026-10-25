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
	if len(s.examples) > 0 {
		start := s.batchN * s.batchSize
		if start >= len(s.examples) {
			return batch, io.EOF
		}
		end := min(start+s.batchSize, len(s.examples))
		batch = append(batch, s.examples[start:end]...)
	} else {
		for len(batch) < s.batchSize {
			lineBytes, readErr := fileutil.ReadLine(s.reader)
			if readErr != nil {
				if !errors.Is(readErr, io.EOF) {
					return nil, readErr
				}
				endErr = io.EOF
				break
			}
			var example E
			if err := json.Unmarshal(lineBytes, &example); err != nil {
				return nil, fmt.Errorf("failed to parse JSON line: %w", err)
			}
			if s.keep != nil && !s.keep(example) {
				s.skipped++
				continue
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

// all rewinds and returns every example, preprocessed if a preprocessing function was given.
func (s *exampleSource[E]) all() ([]E, error) {
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
