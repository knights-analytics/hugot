package pipelines

import (
	"testing"

	"github.com/knights-analytics/hugot/backends"
)

func TestImageFeatureExtractionPipelineValidate(t *testing.T) {
	model := &backends.Model{
		InputsMeta: []backends.InputOutputInfo{{Name: "pixel_values", Dimensions: backends.Shape{-1, 3, 224, 224}}}}
	pipeline := &ImageFeatureExtractionPipeline{BasePipeline: &backends.BasePipeline{Model: model}}
	if err := pipeline.Validate(); err == nil {
		t.Fatal("expected missing image output metadata to fail")
	}
}

func TestImageFeatureExtractionPipelinePostprocessRejectsUnsupportedOutput(t *testing.T) {
	pipeline := &ImageFeatureExtractionPipeline{OutputIndex: 0}
	_, err := pipeline.postprocess(&backends.PipelineBatch{OutputValues: []any{"invalid"}})
	if err == nil {
		t.Fatal("expected unsupported output type to fail")
	}
}
