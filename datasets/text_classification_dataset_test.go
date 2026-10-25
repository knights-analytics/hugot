package datasets

import (
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// withLabels gives a dataset the label map it would otherwise take from the pipeline's model.
func withLabels(d *TextClassificationDataset, labels ...string) *TextClassificationDataset {
	d.labelIDs = make(map[string]int, len(labels))
	for i, label := range labels {
		d.labelIDs[label] = i
	}
	return d
}

func TestTextClassificationSingleLabelTensor(t *testing.T) {
	d, err := NewInMemoryTextClassificationDataset([]TextClassificationExample{{Text: "x", Label: "POSITIVE"}}, 2, SingleLabel, nil)
	if err != nil {
		t.Fatal(err)
	}
	withLabels(d, "NEGATIVE", "POSITIVE")
	labels, err := d.labelTensor([]TextClassificationExample{{Text: "a", Label: "POSITIVE"}, {Text: "b", Label: "NEGATIVE"}})
	if err != nil {
		t.Fatal(err)
	}
	if dims := labels.Shape().Dimensions; !slices.Equal(dims, []int{2, 1}) {
		t.Fatalf("label shape %v, want [2 1]", dims)
	}
	if got := labels.Value().([][]int32); got[0][0] != 1 || got[1][0] != 0 {
		t.Fatalf("label ids %v, want [[1] [0]]", got)
	}
}

func TestTextClassificationMultiLabelTensor(t *testing.T) {
	d, err := NewInMemoryTextClassificationDataset([]TextClassificationExample{{Text: "x"}}, 2, MultiLabel, nil)
	if err != nil {
		t.Fatal(err)
	}
	withLabels(d, "delivery", "sizing", "price")
	labels, err := d.labelTensor([]TextClassificationExample{
		{Text: "a", Labels: []string{"price", "delivery"}},
		{Text: "b"}, // no labels at all is a valid multi-label example
	})
	if err != nil {
		t.Fatal(err)
	}
	want := [][]float32{{1, 0, 1}, {0, 0, 0}}
	got := labels.Value().([][]float32)
	for i := range want {
		if !slices.Equal(got[i], want[i]) {
			t.Fatalf("multi-hot labels %v, want %v", got, want)
		}
	}
}

func TestTextClassificationLabelErrors(t *testing.T) {
	single, _ := NewInMemoryTextClassificationDataset([]TextClassificationExample{{Text: "x"}}, 1, SingleLabel, nil)
	withLabels(single, "NEGATIVE", "POSITIVE")
	multi, _ := NewInMemoryTextClassificationDataset([]TextClassificationExample{{Text: "x"}}, 1, MultiLabel, nil)
	withLabels(multi, "NEGATIVE", "POSITIVE")
	for name, tc := range map[string]struct {
		d       *TextClassificationDataset
		example TextClassificationExample
		want    string
	}{
		"unknown label":           {single, TextClassificationExample{Text: "a", Label: "NEUTRAL"}, `"NEUTRAL" is not in the model's id2label map (NEGATIVE, POSITIVE)`},
		"missing label":           {single, TextClassificationExample{Text: "a"}, `label "" is not in`},
		"labels on single-label":  {single, TextClassificationExample{Text: "a", Labels: []string{"POSITIVE"}}, "use label"},
		"label on multi-label":    {multi, TextClassificationExample{Text: "a", Label: "POSITIVE"}, "use labels"},
		"unknown multi-label one": {multi, TextClassificationExample{Text: "a", Labels: []string{"NEUTRAL"}}, `"NEUTRAL" is not in`},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := tc.d.labelTensor([]TextClassificationExample{tc.example})
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("got error %v, want one containing %q", err, tc.want)
			}
		})
	}
}

func TestTextClassificationValidate(t *testing.T) {
	examples := []TextClassificationExample{{Text: "x", Label: "POSITIVE"}}
	if _, err := NewInMemoryTextClassificationDataset(examples, 0, SingleLabel, nil); err == nil {
		t.Fatal("expected an error for batch size 0")
	}
	if _, err := NewInMemoryTextClassificationDataset(examples, 1, ClassificationProblem(7), nil); err == nil {
		t.Fatal("expected an error for an unknown problem type")
	}
	if _, err := NewTextClassificationDataset(context.Background(), "train.csv", 1, SingleLabel, nil, nil); err == nil {
		t.Fatal("expected an error for a non-jsonl file")
	}
}

func readAllRaw(t *testing.T, d *TextClassificationDataset) [][]string {
	t.Helper()
	var batches [][]string
	for {
		batch, err := d.YieldRaw()
		texts := make([]string, len(batch))
		for i, example := range batch {
			texts[i] = example.Text
		}
		if len(texts) > 0 {
			batches = append(batches, texts)
		}
		if errors.Is(err, io.EOF) {
			return batches
		}
		if err != nil {
			t.Fatal(err)
		}
	}
}

func TestTextClassificationYieldRawFromFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "train.jsonl")
	content := `{"text":"a","label":"POSITIVE"}
{"text":"b","label":"NEGATIVE"}
{"text":"c","label":"POSITIVE"}
`
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	d, err := NewTextClassificationDataset(context.Background(), path, 2, SingleLabel, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = d.Close() }()
	// Two passes, so Reset must reopen the file for the second epoch.
	for epoch := range 2 {
		d.Reset()
		batches := readAllRaw(t, d)
		if len(batches) != 2 || !slices.Equal(batches[0], []string{"a", "b"}) || !slices.Equal(batches[1], []string{"c"}) {
			t.Fatalf("epoch %d: batches %v, want [[a b] [c]]", epoch, batches)
		}
	}
}

func TestTextClassificationYieldRawInMemoryWithPreprocessing(t *testing.T) {
	examples := []TextClassificationExample{{Text: "a"}, {Text: "b"}, {Text: "c"}}
	upper := func(batch []TextClassificationExample) ([]TextClassificationExample, error) {
		out := make([]TextClassificationExample, len(batch))
		for i, example := range batch {
			example.Text = strings.ToUpper(example.Text)
			out[i] = example
		}
		return out, nil
	}
	d, err := NewInMemoryTextClassificationDataset(examples, 2, SingleLabel, upper)
	if err != nil {
		t.Fatal(err)
	}
	batches := readAllRaw(t, d)
	if len(batches) != 2 || !slices.Equal(batches[0], []string{"A", "B"}) || !slices.Equal(batches[1], []string{"C"}) {
		t.Fatalf("batches %v, want [[A B] [C]]", batches)
	}
	if examples[0].Text != "a" {
		t.Fatal("preprocessing must not modify the caller's examples")
	}
}

func TestTextClassificationRequiresClassificationPipeline(t *testing.T) {
	d, _ := NewInMemoryTextClassificationDataset([]TextClassificationExample{{Text: "x"}}, 1, SingleLabel, nil)
	if err := d.SetTokenizationPipeline(nil); err == nil {
		t.Fatal("expected an error for a nil pipeline")
	}
}
