package pipelines

import (
	"context"
	"image"
	"testing"

	"github.com/knights-analytics/hugot/backends"
)

func TestZeroShotObjectDetectionIsNotGenerative(t *testing.T) {
	pipeline := &ZeroShotObjectDetectionPipeline{}
	if pipeline.IsGenerative() {
		t.Fatal("zero-shot object detection pipeline must not be generative")
	}
}

func TestZeroShotObjectDetectionUsesConfiguredImageSize(t *testing.T) {
	model := &backends.Model{
		ImageSize: 960,
		InputsMeta: []backends.InputOutputInfo{
			{Name: "pixel_values", Dimensions: backends.Shape{-1, 3, -1, -1}},
			{Name: "input_ids", Dimensions: backends.Shape{-1, -1}},
		},
		OutputsMeta: []backends.InputOutputInfo{
			{Name: "pred_boxes"},
			{Name: "logits"},
		},
		Tokenizer: &backends.Tokenizer{},
	}
	pipeline, err := NewZeroShotObjectDetectionPipeline(context.Background(), backends.PipelineConfig[*ZeroShotObjectDetectionPipeline]{
		Options: []backends.PipelineOption[*ZeroShotObjectDetectionPipeline]{WithZeroShotObjectLabels([]string{"cat"})},
	}, model)
	if err != nil {
		t.Fatal(err)
	}

	img := image.NewRGBA(image.Rect(0, 0, 640, 480))
	processed, err := pipeline.preprocessImages([]image.Image{img})
	if err != nil {
		t.Fatal(err)
	}
	if len(processed) != 1 {
		t.Fatalf("unexpected preprocessed image batch size: got %d, want 1", len(processed))
	}
	if len(processed[0]) != 3 {
		t.Fatalf("unexpected preprocessed image channel count: got %d, want 3", len(processed[0]))
	}
	if len(processed[0][0]) != 960 {
		t.Fatalf("unexpected preprocessed image height: got %d, want 960", len(processed[0][0]))
	}
	if len(processed[0][0][0]) != 960 {
		t.Fatalf("unexpected preprocessed image width: got %d, want 960", len(processed[0][0][0]))
	}
}
