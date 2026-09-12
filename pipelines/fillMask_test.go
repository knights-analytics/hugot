package pipelines

import (
	"testing"

	"github.com/knights-analytics/hugot/backends"
)

func TestFillMaskPipelineValidate(t *testing.T) {
	model := &backends.Model{
		OutputsMeta: []backends.InputOutputInfo{{Name: "logits", Dimensions: backends.Shape{-1, -1, 32}}}}
	pipeline := &FillMaskPipeline{BasePipeline: &backends.BasePipeline{Model: model}, TopK: 0}
	if err := pipeline.Validate(); err == nil {
		t.Fatal("expected invalid fill-mask configuration to fail")
	}
}

func TestFillMaskPipelinePostprocessRejectsUnsupportedOutput(t *testing.T) {
	pipeline := &FillMaskPipeline{BasePipeline: &backends.BasePipeline{}, TopK: 1}
	_, err := pipeline.postprocess(&backends.PipelineBatch{OutputValues: []any{"invalid"}})
	if err == nil {
		t.Fatal("expected unsupported output type to fail")
	}
}
