package pipelines

import (
	"strings"
	"testing"

	"github.com/knights-analytics/hugot/backends"
)

func TestAutomaticSpeechRecognitionValidation(t *testing.T) {
	model := &backends.Model{InputsMeta: []backends.InputOutputInfo{{Name: "input_values", Dimensions: backends.NewShape(-1, -1)}}, OutputsMeta: []backends.InputOutputInfo{{Name: "logits", Dimensions: backends.NewShape(-1, -1, 3)}}, IDLabelMap: map[int]string{0: "_", 1: "a", 2: "b"}}
	pipeline := &AutomaticSpeechRecognitionPipeline{BasePipeline: &backends.BasePipeline{Model: model}, IDLabelMap: model.IDLabelMap}
	if err := pipeline.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestAutomaticSpeechRecognitionCTCDecode(t *testing.T) {
	pipeline := &AutomaticSpeechRecognitionPipeline{BasePipeline: &backends.BasePipeline{Model: &backends.Model{}}, IDLabelMap: map[int]string{0: "_", 1: "a", 2: "b"}}
	output, err := pipeline.postprocess(&backends.PipelineBatch{OutputValues: []any{[][][]float32{{{0, 4, 0}, {0, 3, 0}, {0, 1, 0}, {0, 0, 5}, {0, 2, 0}}}}})
	if err != nil {
		t.Fatal(err)
	}
	if output.Text[0] != "aba" {
		t.Fatalf("expected aba, got %q", output.Text[0])
	}
}

func TestAutomaticSpeechRecognitionCTCTextUsesWordDelimiter(t *testing.T) {
	labels := map[int]string{
		1: "M", 2: "I", 3: "S", 4: "T", 5: "E", 6: "R",
		7: "|", 8: "Q", 9: "U", 10: "I", 11: "L",
	}
	ids := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 4, 5, 6}
	frames := make([][]float32, len(ids))
	for i, id := range ids {
		frames[i] = make([]float32, 12)
		frames[i][id] = 1
	}
	pipeline := &AutomaticSpeechRecognitionPipeline{
		BasePipeline: &backends.BasePipeline{Model: &backends.Model{}},
		IDLabelMap:   labels,
	}
	output, err := pipeline.postprocess(&backends.PipelineBatch{OutputValues: []any{[][][]float32{frames}}})
	if err != nil {
		t.Fatal(err)
	}
	if output.Text[0] != "MISTER QUILTER" {
		t.Fatalf("expected MISTER QUILTER, got %q", output.Text[0])
	}
}

func TestAutomaticSpeechRecognitionCTCMixedLabelsKeepOrder(t *testing.T) {
	pipeline := &AutomaticSpeechRecognitionPipeline{BasePipeline: &backends.BasePipeline{Model: &backends.Model{}}, IDLabelMap: map[int]string{1: "a", 3: "c"}}
	output, err := pipeline.postprocess(&backends.PipelineBatch{OutputValues: []any{[][][]float32{{
		{0, 4, 0, 0}, {0, 3, 0, 0}, {0, 0, 0, 5}, {9, 0, 0, 0},
		{0, 0, 6, 0}, {0, 0, 5, 0}, {8, 0, 0, 0}, {0, 7, 0, 0},
	}}}})
	if err != nil {
		t.Fatal(err)
	}
	if output.Text[0] != "ac2a" || strings.Join(output.Words[0], ",") != "a,c,2,a" {
		t.Fatalf("unexpected mixed-label CTC result: %#v", output)
	}
}

func TestAutomaticSpeechRecognitionRejectsOutputLayout(t *testing.T) {
	pipeline := &AutomaticSpeechRecognitionPipeline{BasePipeline: &backends.BasePipeline{Model: &backends.Model{}}}
	if _, err := pipeline.postprocess(&backends.PipelineBatch{OutputValues: []any{"bad"}}); err == nil || !strings.Contains(err.Error(), "unsupported ASR output type") {
		t.Fatalf("unexpected error: %v", err)
	}
}
