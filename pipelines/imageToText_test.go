package pipelines

import (
	"testing"

	"github.com/knights-analytics/hugot/backends"
)

func TestImageToTextRejectsMissingModel(t *testing.T) {
	if _, err := NewImageToTextPipeline(t.Context(), ImageToTextConfig{}, nil); err == nil {
		t.Fatal("captioning accepted a missing model")
	}
}

func TestImageToTextRejectsUnrelatedONNXModel(t *testing.T) {
	model := &backends.Model{InputsMeta: []backends.InputOutputInfo{{Name: "input_ids"}}, OutputsMeta: []backends.InputOutputInfo{{Name: "logits"}}}
	if _, err := NewImageToTextPipeline(t.Context(), ImageToTextConfig{}, model); err == nil {
		t.Fatal("captioning accepted an unrelated ONNX model")
	}
}