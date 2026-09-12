package pipelines

import (
	"context"
	"image"
	"reflect"
	"testing"

	"github.com/knights-analytics/hugot/backends"
)

func TestDepthEstimationPipelineValidate(t *testing.T) {
	model := &backends.Model{
		InputsMeta:  []backends.InputOutputInfo{{Name: "pixel_values", Dimensions: backends.Shape{-1, 3, 224, 224}}},
		OutputsMeta: []backends.InputOutputInfo{{Name: "predicted_depth", Dimensions: backends.Shape{-1, 224, 224}}}}
	pipeline := &DepthEstimationPipeline{BasePipeline: &backends.BasePipeline{Model: model}}
	if err := pipeline.Validate(); err != nil {
		t.Fatalf("expected valid depth configuration: %v", err)
	}
	if pipeline.DepthOutput != "predicted_depth" {
		t.Fatalf("unexpected inferred output: %q", pipeline.DepthOutput)
	}
}

func TestDepthEstimationUsesConfiguredImageSizeForDynamicInputs(t *testing.T) {
	pipeline, err := NewDepthEstimationPipeline(context.Background(), DepthEstimationConfig{}, &backends.Model{
		ModelMetadata: backends.ModelMetadata{
			ImageSize:   384,
			InputsMeta:  []backends.InputOutputInfo{{Name: "pixel_values", Dimensions: backends.Shape{-1, 3, -1, -1}}},
			OutputsMeta: []backends.InputOutputInfo{{Name: "predicted_depth", Dimensions: backends.Shape{-1, -1, -1}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	img := image.NewRGBA(image.Rect(0, 0, 640, 480))
	processed := image.Image(img)
	for _, step := range pipeline.preprocessSteps {
		processed, err = step.Apply(processed)
		if err != nil {
			t.Fatal(err)
		}
	}
	if got := processed.Bounds().Size(); got != image.Pt(384, 384) {
		t.Fatalf("unexpected preprocessed image size: %v", got)
	}
}

func TestDepthEstimationPostprocessDecodesAndResizes(t *testing.T) {
	pipeline := &DepthEstimationPipeline{
		BasePipeline: &backends.BasePipeline{Model: &backends.Model{
			OutputsMeta: []backends.InputOutputInfo{{Name: "depth"}}}},
		DepthOutput: "depth",
	}
	batch := &backends.PipelineBatch{
		InputMetadata: []image.Point{{X: 4, Y: 2}},
		OutputValues:  []any{[][][]float32{{{1, 2}, {3, 4}}}},
	}
	output, err := pipeline.postprocess(batch)
	if err != nil {
		t.Fatal(err)
	}
	want := [][]float32{{1, 1, 2, 2}, {3, 3, 4, 4}}
	if !reflect.DeepEqual(output.Results[0].DepthMap, want) {
		t.Fatalf("unexpected depth map: %#v", output.Results[0].DepthMap)
	}
}

func TestDecodeDepthOutputAcceptsSingletonChannel(t *testing.T) {
	output, err := decodeDepthOutput([][][][]float32{{{{1, 2}, {3, 4}}}})
	if err != nil || !reflect.DeepEqual(output, [][][]float32{{{1, 2}, {3, 4}}}) {
		t.Fatalf("unexpected decoded output: %#v, %v", output, err)
	}
}

func TestDepthEstimationPostprocessRejectsUnsupportedOutput(t *testing.T) {
	pipeline := &DepthEstimationPipeline{
		BasePipeline: &backends.BasePipeline{Model: &backends.Model{
			OutputsMeta: []backends.InputOutputInfo{{Name: "depth"}}}},
		DepthOutput: "depth",
	}
	if _, err := pipeline.postprocess(&backends.PipelineBatch{OutputValues: []any{"invalid"}}); err == nil {
		t.Fatal("expected unsupported output type to fail")
	}
}
