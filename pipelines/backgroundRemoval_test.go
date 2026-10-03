package pipelines

import (
	"image"
	"math"
	"testing"

	"github.com/knights-analytics/hugot/backends"
)

func TestBackgroundRemovalRejectsSemanticClassOutput(t *testing.T) {
	model := &backends.Model{
		InputsMeta:  []backends.InputOutputInfo{{Name: "input", Dimensions: backends.Shape{-1, 3, -1, -1}}},
		OutputsMeta: []backends.InputOutputInfo{{Name: "logits", Dimensions: backends.Shape{-1, 150, -1, -1}}},
	}
	pipeline := &BackgroundRemovalPipeline{BasePipeline: &backends.BasePipeline{Model: model}, OutputName: "logits"}
	if err := pipeline.Validate(); err == nil {
		t.Fatal("semantic class channels must not be interpreted as a foreground mask")
	}
	pipeline.OutputName = "missing"
	if err := pipeline.Validate(); err == nil {
		t.Fatal("expected an unknown output name to fail validation")
	}
}

func TestBackgroundRemovalPostprocess(t *testing.T) {
	model := &backends.Model{OutputsMeta: []backends.InputOutputInfo{{Name: "other"}, {Name: "alpha"}}}
	pipeline := &BackgroundRemovalPipeline{BasePipeline: &backends.BasePipeline{Model: model}, OutputName: "alpha"}
	mask := [][]float32{{0, 1}, {0.25, 0.75}}
	for _, raw := range []any{[][][]float32{mask}, [][][][]float32{{mask}}} {
		result, err := pipeline.postprocess(&backends.PipelineBatch{
			InputMetadata: []image.Point{{X: 4, Y: 4}}, OutputValues: []any{"unused", raw},
		})
		if err != nil {
			t.Fatal(err)
		}
		if len(result.Results) != 1 || result.Results[0].Width != 4 || result.Results[0].Height != 4 {
			t.Fatal("expected a source-sized result")
		}
		output := result.Results[0].Mask
		if len(output) != 4 || len(output[0]) != 4 || output[0][0] != 0 || output[3][3] != 0.75 {
			t.Fatal("resizing must preserve alpha probabilities")
		}
	}
}

func TestBackgroundRemovalRejectsInvalidMasks(t *testing.T) {
	model := &backends.Model{OutputsMeta: []backends.InputOutputInfo{{Name: "alpha"}}}
	pipeline := &BackgroundRemovalPipeline{BasePipeline: &backends.BasePipeline{Model: model}, OutputName: "alpha"}
	for name, raw := range map[string]any{
		"class channels": [][][][]float32{{{{0}}, {{1}}}},
		"empty":          [][][]float32{{}},
		"ragged":         [][][]float32{{{0, 1}, {1}}},
		"nan":            [][][]float32{{{float32(math.NaN())}}},
		"infinite":       [][][]float32{{{float32(math.Inf(1))}}},
		"logit":          [][][]float32{{{2}}},
		"negative":       [][][]float32{{{-1}}},
		"extra batch":    [][][]float32{{{0}}, {{1}}},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := pipeline.postprocess(&backends.PipelineBatch{
				InputMetadata: []image.Point{{X: 2, Y: 2}}, OutputValues: []any{raw},
			})
			if err == nil {
				t.Fatal("expected invalid foreground probabilities to fail")
			}
		})
	}
}