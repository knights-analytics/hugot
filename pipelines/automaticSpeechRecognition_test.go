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

func TestAutomaticSpeechRecognitionRejectsOutputLayout(t *testing.T) {
	pipeline := &AutomaticSpeechRecognitionPipeline{BasePipeline: &backends.BasePipeline{Model: &backends.Model{}}}
	if _, err := pipeline.postprocess(&backends.PipelineBatch{OutputValues: []any{"bad"}}); err == nil || !strings.Contains(err.Error(), "unsupported ASR output type") {
		t.Fatalf("unexpected error: %v", err)
	}
}
