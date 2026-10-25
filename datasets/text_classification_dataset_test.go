package datasets

import (
	"context"
	"encoding/json"
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

func parseTextExample(t *testing.T, line string) TextClassificationExample {
	t.Helper()
	var example TextClassificationExample
	if err := json.Unmarshal([]byte(line), &example); err != nil {
		t.Fatal(err)
	}
	return example
}

func TestTextChoiceAnnotations(t *testing.T) {
	choice := parseTextExample(t, `{"text": "Late and the wrong size", "options": [{"id": "DELIVERY"}, {"id": "SIZING"}, {"id": "PRICE"}], "accept": ["DELIVERY", "SIZING"], "answer": "accept"}`)
	if !choice.IsChoiceFormat() {
		t.Fatal("want a choice annotation")
	}
	got, keep, err := choice.Resolve(MultiLabel, nil)
	if err != nil || !keep || !slices.Equal(got.Labels, []string{"DELIVERY", "SIZING"}) || got.Accept != nil {
		t.Fatalf("multi-label: got %+v, %v, %v", got, keep, err)
	}
	if _, _, err = choice.Resolve(SingleLabel, nil); err == nil || !strings.Contains(err.Error(), "exactly one") {
		t.Fatalf("single-label with two choices: got %v, want an error", err)
	}
	exclusive := parseTextExample(t, `{"text": "Arrived a week late", "accept": ["DELIVERY"], "answer": "accept"}`)
	if got, keep, err = exclusive.Resolve(SingleLabel, nil); err != nil || !keep || got.Label != "DELIVERY" {
		t.Fatalf("single-label: got %+v, %v, %v", got, keep, err)
	}
	none := parseTextExample(t, `{"text": "Fine, thanks", "accept": [], "answer": "accept"}`)
	if got, keep, err = none.Resolve(MultiLabel, nil); err != nil || !keep || got.Labels == nil || len(got.Labels) != 0 {
		t.Fatalf("no label chosen: got %+v, %v, %v; want an example with no labels", got, keep, err)
	}
	for _, answer := range []string{"reject", "ignore"} {
		skipped := parseTextExample(t, `{"text": "x", "accept": ["DELIVERY"], "answer": "`+answer+`"}`)
		if _, keep, err = skipped.Resolve(MultiLabel, nil); err != nil || keep {
			t.Fatalf("%s: got keep %v, %v; want skipped", answer, keep, err)
		}
	}
}

func TestTextBinaryAnnotations(t *testing.T) {
	accept := parseTextExample(t, `{"text": "You fool", "label": "INSULT", "answer": "accept"}`)
	reject := parseTextExample(t, `{"text": "Good morning", "label": "INSULT", "answer": "reject"}`)
	twoClass := []string{"INSULT", "POLITE"}
	threeClass := []string{"INSULT", "POLITE", "NEUTRAL"}

	if got, keep, err := accept.Resolve(SingleLabel, threeClass); err != nil || !keep || got.Label != "INSULT" {
		t.Fatalf("accept: got %+v, %v, %v", got, keep, err)
	}
	// Rejecting a label of a two-class model is the other class.
	if got, keep, err := reject.Resolve(SingleLabel, twoClass); err != nil || !keep || got.Label != "POLITE" {
		t.Fatalf("reject, two classes: got %+v, %v, %v; want POLITE", got, keep, err)
	}
	// With more classes a rejection does not say which class the text is, so it is skipped.
	if _, keep, err := reject.Resolve(SingleLabel, threeClass); err != nil || keep {
		t.Fatalf("reject, three classes: got keep %v, %v; want skipped", keep, err)
	}
	// A model with only the annotated label: binary multi-label.
	if got, keep, err := reject.Resolve(MultiLabel, []string{"INSULT"}); err != nil || !keep || got.Labels == nil || len(got.Labels) != 0 {
		t.Fatalf("reject, one-label multi-label: got %+v, %v, %v; want no labels", got, keep, err)
	}
	if got, keep, err := accept.Resolve(MultiLabel, []string{"INSULT"}); err != nil || !keep || !slices.Equal(got.Labels, []string{"INSULT"}) || got.Label != "" {
		t.Fatalf("accept, one-label multi-label: got %+v, %v, %v", got, keep, err)
	}
	// With several labels the other labels are unknown: an error rather than a silent negative.
	if _, _, err := accept.Resolve(MultiLabel, threeClass); err == nil || !strings.Contains(err.Error(), "other labels are unknown") {
		t.Fatalf("accept, several-label multi-label: got %v, want an error", err)
	}
	if _, _, err := parseTextExample(t, `{"text": "x", "answer": "accept", "accept": ["A"], "label": "A"}`).Resolve(SingleLabel, nil); err == nil {
		t.Fatal("want an error for accept and label together")
	}
	if _, _, err := parseTextExample(t, `{"text": "x", "label": "A", "answer": "maybe"}`).Resolve(SingleLabel, nil); err == nil {
		t.Fatal("want an error for an unknown answer")
	}
	// The dataset's own format is untouched.
	plain := TextClassificationExample{Text: "x", Label: "A"}
	if got, keep, err := plain.Resolve(SingleLabel, twoClass); err != nil || !keep || got.Label != "A" {
		t.Fatalf("plain: got %+v, %v, %v", got, keep, err)
	}
}

// Ignored examples are skipped on read; binary rejections wait until the model's labels are known.
func TestTextProdigySkipping(t *testing.T) {
	lines := []string{
		`{"text": "You fool", "label": "INSULT", "answer": "accept"}`,
		`{"text": "Good morning", "label": "INSULT", "answer": "reject"}`,
		`{"text": "Hmm", "label": "INSULT", "answer": "ignore"}`,
		`{"text": "Have a nice day", "label": "POLITE"}`,
	}
	path := filepath.Join(t.TempDir(), "binary.jsonl")
	if err := os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	fromFile, err := NewTextClassificationDataset(t.Context(), path, 2, SingleLabel, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	examples, err := fromFile.Examples()
	if err != nil || len(examples) != 3 {
		t.Fatalf("file: got %d examples, %v; want the ignored one skipped", len(examples), err)
	}

	var inMemoryExamples []TextClassificationExample
	for _, line := range lines {
		inMemoryExamples = append(inMemoryExamples, parseTextExample(t, line))
	}
	inMemory, err := NewInMemoryTextClassificationDataset(inMemoryExamples, 2, SingleLabel, nil)
	if err != nil {
		t.Fatal(err)
	}
	inMemory.labelIDs = map[string]int{"INSULT": 0, "POLITE": 1, "NEUTRAL": 2}
	kept := 0
	for _, example := range inMemory.source.examples {
		if inMemory.keepExample(example) {
			kept++
		}
	}
	// Without the model known, a rejection is kept for later; ignore is always skipped.
	if kept != 3 {
		t.Fatalf("kept %d before the model is known, want 3", kept)
	}
}
