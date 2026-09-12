package pipelines

import (
	"strings"
	"testing"

	"github.com/knights-analytics/hugot/backends"
)

func TestZeroShotAudioClassificationValidateAndPostprocess(t *testing.T) {
	model := &backends.Model{
		InputsMeta:  []backends.InputOutputInfo{{Name: "input_values", Dimensions: backends.Shape{-1, -1}}},
		OutputsMeta: []backends.InputOutputInfo{{Name: "logits", Dimensions: backends.Shape{-1, 3}}}}
	p := &ZeroShotAudioClassificationPipeline{BasePipeline: &backends.BasePipeline{Model: model}, Labels: []string{"speech", "music", "noise"}, TopK: 2}
	if err := p.Validate(); err != nil {
		t.Fatalf("expected valid configuration: %v", err)
	}
	out, err := p.postprocess(&backends.PipelineBatch{OutputValues: []any{[][]float32{{0.1, 0.9, 0.4}}}})
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Predictions) != 1 || len(out.Predictions[0]) != 2 || out.Predictions[0][0].Label != "music" || out.Predictions[0][1].Label != "noise" {
		t.Fatalf("unexpected predictions: %#v", out.Predictions)
	}
}

func TestZeroShotAudioClassificationRejectsLabelMismatch(t *testing.T) {
	p := &ZeroShotAudioClassificationPipeline{BasePipeline: &backends.BasePipeline{Model: &backends.Model{
		InputsMeta: []backends.InputOutputInfo{{Dimensions: backends.Shape{-1, -1}}}, OutputsMeta: []backends.InputOutputInfo{{Dimensions: backends.Shape{-1, 2}}}}}, Labels: []string{"one", "two", "three"}, TopK: 1}
	if err := p.Validate(); err == nil || !strings.Contains(err.Error(), "candidate labels") {
		t.Fatalf("expected candidate label mismatch, got %v", err)
	}
}

func TestZeroShotAudioClassificationRejectsUnsupportedOutput(t *testing.T) {
	p := &ZeroShotAudioClassificationPipeline{TopK: 1}
	if _, err := p.postprocess(&backends.PipelineBatch{OutputValues: []any{"invalid"}}); err == nil {
		t.Fatal("expected unsupported output type to fail")
	}
}
