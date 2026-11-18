package datasets

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

type numberExample struct {
	N int `json:"n"`
}

// pass reads one full pass over the source, batch by batch, as the training loop does.
func pass(t *testing.T, s *exampleSource[numberExample]) []int {
	t.Helper()
	s.reset()
	var seen []int
	for {
		batch, err := s.next()
		for _, e := range batch {
			seen = append(seen, e.N)
		}
		if errors.Is(err, io.EOF) {
			return seen
		}
		if err != nil {
			t.Fatal(err)
		}
		if len(batch) > s.batchSize {
			t.Fatalf("batch of %d examples, larger than the batch size %d", len(batch), s.batchSize)
		}
	}
}

func numbers(n int) []numberExample {
	examples := make([]numberExample, n)
	for i := range examples {
		examples[i] = numberExample{N: i}
	}
	return examples
}

func numbersFile(t *testing.T, n int) string {
	t.Helper()
	var b strings.Builder
	for i := range n {
		fmt.Fprintf(&b, "{\"n\":%d}\n", i)
	}
	path := filepath.Join(t.TempDir(), "numbers.jsonl")
	if err := os.WriteFile(path, []byte(b.String()), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func fileSource(t *testing.T, n, batchSize int) *exampleSource[numberExample] {
	t.Helper()
	s := &exampleSource[numberExample]{path: numbersFile(t, n), batchSize: batchSize}
	if err := s.open(context.Background(), nil); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = s.close() })
	return s
}

func sorted(values []int) []int {
	values = slices.Clone(values)
	slices.Sort(values)
	return values
}

// checkShuffled checks that every pass visits each example exactly once, that consecutive passes
// use different orders, and that none is the original order.
func checkShuffled(t *testing.T, s *exampleSource[numberExample], n int) [][]int {
	t.Helper()
	var passes [][]int
	for range 3 {
		seen := pass(t, s)
		if !slices.Equal(sorted(seen), sorted(numbersInOrder(n))) {
			t.Fatalf("a shuffled pass must visit every example exactly once, got %v", seen)
		}
		if slices.Equal(seen, numbersInOrder(n)) {
			t.Fatalf("a shuffled pass came out in the original order")
		}
		passes = append(passes, seen)
	}
	if slices.Equal(passes[0], passes[1]) || slices.Equal(passes[1], passes[2]) {
		t.Fatalf("each pass should have a new order, got %v", passes)
	}
	return passes
}

func numbersInOrder(n int) []int {
	values := make([]int, n)
	for i := range values {
		values[i] = i
	}
	return values
}

func TestExampleSourceUnshuffledKeepsOrder(t *testing.T) {
	inMemory := &exampleSource[numberExample]{examples: numbers(10), batchSize: 3}
	if got := pass(t, inMemory); !slices.Equal(got, numbersInOrder(10)) {
		t.Fatalf("in memory: got %v", got)
	}
	if got := pass(t, fileSource(t, 10, 3)); !slices.Equal(got, numbersInOrder(10)) {
		t.Fatalf("file: got %v", got)
	}
}

func TestExampleSourceShufflesInMemory(t *testing.T) {
	s := &exampleSource[numberExample]{examples: numbers(20), batchSize: 3}
	s.setShuffle(7)
	first := checkShuffled(t, s, 20)

	// The same seed gives the same sequence of orders.
	again := &exampleSource[numberExample]{examples: numbers(20), batchSize: 3}
	again.setShuffle(7)
	for i, want := range first {
		if got := pass(t, again); !slices.Equal(got, want) {
			t.Fatalf("pass %d with the same seed: got %v, want %v", i, got, want)
		}
	}
}

func TestExampleSourceShufflesFile(t *testing.T) {
	s := fileSource(t, 20, 3)
	s.setShuffle(7)
	first := checkShuffled(t, s, 20)

	again := fileSource(t, 20, 3)
	again.setShuffle(7)
	for i, want := range first {
		if got := pass(t, again); !slices.Equal(got, want) {
			t.Fatalf("pass %d with the same seed: got %v, want %v", i, got, want)
		}
	}
}

// A file longer than the shuffle buffer is still read whole, each example once.
func TestExampleSourceShufflesFileLongerThanBuffer(t *testing.T) {
	n := shuffleBufferSize + 1234
	s := fileSource(t, n, 64)
	s.setShuffle(1)
	seen := pass(t, s)
	if !slices.Equal(sorted(seen), numbersInOrder(n)) {
		t.Fatalf("got %d examples, want each of the %d once", len(seen), n)
	}
}

// Examples are read in their original order, and reading them does not disturb the shuffle
// sequence of the training passes.
func TestExampleSourceAllIsOrdered(t *testing.T) {
	for name, source := range map[string]func() *exampleSource[numberExample]{
		"in memory": func() *exampleSource[numberExample] {
			return &exampleSource[numberExample]{examples: numbers(12), batchSize: 5}
		},
		"file": func() *exampleSource[numberExample] { return fileSource(t, 12, 5) },
	} {
		t.Run(name, func(t *testing.T) {
			reference := source()
			reference.setShuffle(3)
			want := [][]int{pass(t, reference), pass(t, reference)}

			s := source()
			s.setShuffle(3)
			got := [][]int{pass(t, s)}
			all, err := s.all()
			if err != nil {
				t.Fatal(err)
			}
			var allN []int
			for _, e := range all {
				allN = append(allN, e.N)
			}
			if !slices.Equal(allN, numbersInOrder(12)) {
				t.Fatalf("all() must keep the original order, got %v", allN)
			}
			got = append(got, pass(t, s))
			for i := range want {
				if !slices.Equal(got[i], want[i]) {
					t.Fatalf("pass %d changed after reading all(): got %v, want %v", i, got[i], want[i])
				}
			}
		})
	}
}

// An in-memory semantic similarity dataset yields each example once per pass, in batches of the
// batch size. It used to repeat each batch batch-size times.
func TestSemanticSimilarityInMemoryBatches(t *testing.T) {
	examples := make([]SemanticSimilarityExample, 5)
	for i := range examples {
		examples[i] = SemanticSimilarityExample{Sentence1: fmt.Sprint(i), Sentence2: "x", Score: 1}
	}
	d, err := NewInMemorySemanticSimilarityDataset(examples, 2, nil)
	if err != nil {
		t.Fatal(err)
	}
	d.Reset()
	var sizes []int
	var seen []string
	for {
		batch, err := d.YieldRaw()
		if len(batch) > 0 {
			sizes = append(sizes, len(batch))
		}
		for _, e := range batch {
			seen = append(seen, e.Sentence1)
		}
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
	}
	if !slices.Equal(sizes, []int{2, 2, 1}) {
		t.Fatalf("batch sizes %v, want [2 2 1]", sizes)
	}
	if !slices.Equal(seen, []string{"0", "1", "2", "3", "4"}) {
		t.Fatalf("examples %v, want each once in order", seen)
	}
}

// Every dataset built on exampleSource can be shuffled by the trainer.
func TestDatasetsAreShufflers(t *testing.T) {
	for _, d := range []any{
		&SemanticSimilarityDataset{},
		&ContrastiveDataset{},
		&TextClassificationDataset{},
		&TokenClassificationDataset{},
	} {
		if _, ok := d.(Shuffler); !ok {
			t.Errorf("%T does not implement Shuffler", d)
		}
		if _, ok := d.(BatchCounter); !ok {
			t.Errorf("%T does not implement BatchCounter", d)
		}
	}
}

// numBatches must equal the number of batches a pass actually yields, in memory and from a file,
// with the examples that keep rejects left out.
func TestExampleSourceNumBatches(t *testing.T) {
	for _, tc := range []struct{ n, batchSize, want int }{
		{10, 3, 4}, {9, 3, 3}, {1, 8, 1},
	} {
		memory := &exampleSource[numberExample]{examples: numbers(tc.n), batchSize: tc.batchSize}
		file := fileSource(t, tc.n, tc.batchSize)
		for name, s := range map[string]*exampleSource[numberExample]{"memory": memory, "file": file} {
			got, err := s.numBatches()
			if err != nil {
				t.Fatal(err)
			}
			if got != tc.want {
				t.Errorf("%s, %d examples in batches of %d: %d batches, want %d", name, tc.n, tc.batchSize, got, tc.want)
			}
		}
	}

	even := fileSource(t, 10, 2)
	even.keep = func(e numberExample) bool { return e.N%2 == 0 }
	if got, err := even.numBatches(); err != nil || got != 3 {
		t.Errorf("5 kept examples in batches of 2: got %d batches (err %v), want 3", got, err)
	}
}

// Counting a file in the middle of a pass must not move the pass along.
func TestExampleSourceNumBatchesLeavesPassAlone(t *testing.T) {
	s := fileSource(t, 10, 3)
	s.reset()
	first, err := s.next()
	if err != nil {
		t.Fatal(err)
	}
	if _, err = s.numBatches(); err != nil {
		t.Fatal(err)
	}
	second, err := s.next()
	if err != nil {
		t.Fatal(err)
	}
	if first[0].N != 0 || second[0].N != 3 {
		t.Errorf("the pass moved: batches start at %d and %d, want 0 and 3", first[0].N, second[0].N)
	}
}

// With the same seed, a file that fits in the shuffle buffer is visited in the same orders as the
// same examples held in memory, so the two train identically.
func TestExampleSourceFileShufflesLikeMemory(t *testing.T) {
	for _, batchSize := range []int{1, 3} {
		memory := &exampleSource[numberExample]{examples: numbers(20), batchSize: batchSize}
		file := fileSource(t, 20, batchSize)
		memory.setShuffle(7)
		file.setShuffle(7)
		for epoch := range 3 {
			fromMemory, fromFile := pass(t, memory), pass(t, file)
			if !slices.Equal(fromMemory, fromFile) {
				t.Fatalf("batch size %d, epoch %d: memory %v, file %v", batchSize, epoch, fromMemory, fromFile)
			}
		}
	}
}
