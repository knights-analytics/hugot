package pipelines

import (
	"image"
	"testing"

	"github.com/knights-analytics/hugot/backends"
)

func TestImageSegmentationPipelineValidate(t *testing.T) {
	model := &backends.Model{
		InputsMeta:  []backends.InputOutputInfo{{Name: "pixel_values", Dimensions: backends.Shape{-1, 3, 224, 224}}},
		OutputsMeta: []backends.InputOutputInfo{{Name: "other", Dimensions: backends.Shape{-1, 2, 14, 14}}}}
	pipeline := &ImageSegmentationPipeline{BasePipeline: &backends.BasePipeline{Model: model}, ScoreThreshold: 2}
	if err := pipeline.Validate(); err == nil {
		t.Fatal("expected invalid segmentation configuration to fail")
	}
}

func TestResizeMask(t *testing.T) {
	mask := [][]bool{{true, false}, {false, true}}
	resized := resizeMask(mask, 4, 2)
	if len(resized) != 2 || len(resized[0]) != 4 || !resized[0][0] || !resized[0][1] || resized[0][2] || resized[1][0] || !resized[1][2] {
		t.Fatalf("unexpected resized mask: %#v", resized)
	}
}

func TestImageSegmentationPostprocessPreservesSourceDimensions(t *testing.T) {
	pipeline := &ImageSegmentationPipeline{
		BasePipeline: &backends.BasePipeline{
			Model: &backends.Model{
				OutputsMeta: []backends.InputOutputInfo{{Name: "logits"}},
			},
		},
		LogitsOutput:   "logits",
		ScoreThreshold: 0.5,
	}

	batch := &backends.PipelineBatch{
		InputValues:   []any{"tensor"},
		InputMetadata: []image.Point{{X: 3, Y: 2}},
		OutputValues:  []any{[][][][]float32{{{{1, 0}, {0, 1}}, {{0, 1}, {1, 0}}}}},
	}
	output, err := pipeline.postprocess(batch)
	if err != nil {
		t.Fatal(err)
	}
	if len(output.Results) != 1 || output.Results[0].Width != 3 || output.Results[0].Height != 2 {
		t.Fatalf("unexpected source dimensions: %#v", output.Results)
	}
}
