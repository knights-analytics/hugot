package datasets

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

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
