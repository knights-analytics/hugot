package backends

import (
	"encoding/json"
	"image"
	"image/color"
	"math"
	"testing"

	"github.com/knights-analytics/hugot/testcases/embedded"
	"github.com/knights-analytics/hugot/util/imageutil"
)

func TestPreprocessImagesPythonReference(t *testing.T) {
	var reference struct {
		Cases []struct {
			Name  string
			Input struct {
				Kind          string
				Width, Height int
				RGB           [3]uint8
			}
			OutputShape       []int   `json:"output_shape"`
			AbsoluteTolerance float64 `json:"absolute_tolerance"`
			Samples           []struct {
				X, Y   int
				Values [3]float64
			}
		}
	}
	if err := json.Unmarshal(embedded.PipelineReferenceByte, &reference); err != nil {
		t.Fatal(err)
	}
	if len(reference.Cases) != 5 {
		t.Fatalf("expected five Python reference cases, got %d", len(reference.Cases))
	}
	for _, tc := range reference.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			img := image.NewRGBA(image.Rect(0, 0, tc.Input.Width, tc.Input.Height))
			for y := range tc.Input.Height {
				for x := range tc.Input.Width {
					rgb := tc.Input.RGB
					if tc.Input.Kind != "constant" {
						for c := range rgb {
							rgb[c] = uint8((17*x + 29*y + 43*c + 7*((x*y)%19)) % 256)
						}
					}
					img.SetRGBA(x, y, color.RGBA{R: rgb[0], G: rgb[1], B: rgb[2], A: 255})
				}
			}
			var preprocess []imageutil.PreprocessStep
			if tc.Input.Kind != "constant" {
				preprocess = []imageutil.PreprocessStep{imageutil.ResizeBilinearStep(256), imageutil.CenterCropStep(224, 224)}
			}
			values, err := PreprocessImages("NCHW", []image.Image{img}, preprocess,
				[]imageutil.NormalizationStep{imageutil.RescaleStep(), imageutil.ImagenetPixelNormalizationStep()})
			if err != nil {
				t.Fatal(err)
			}
			if len(tc.OutputShape) != 4 || tc.OutputShape[0] != len(values) ||
				tc.OutputShape[1] != len(values[0]) || tc.OutputShape[2] != len(values[0][0]) ||
				tc.OutputShape[3] != len(values[0][0][0]) {
				t.Fatalf("processed tensor does not match reference shape %v", tc.OutputShape)
			}
			for _, sample := range tc.Samples {
				for c, want := range sample.Values {
					got := float64(values[0][c][sample.Y][sample.X])
					if math.IsNaN(got) || math.IsInf(got, 0) || math.Abs(got-want) > tc.AbsoluteTolerance {
						t.Errorf("channel %d pixel (%d,%d): got %g, want %g (tolerance %g)", c, sample.X, sample.Y, got, want, tc.AbsoluteTolerance)
					}
				}
			}
		})
	}
}

func TestPreprocessImagesRescaleBeforeNormalize(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 1, 1))
	img.SetRGBA(0, 0, color.RGBA{R: 255, G: 128, B: 0, A: 255})
	steps := []imageutil.NormalizationStep{imageutil.RescaleStep(), imageutil.ImagenetPixelNormalizationStep()}
	mean := [3]float64{0.485, 0.456, 0.406}
	std := [3]float64{0.229, 0.224, 0.225}
	pixels := [3]float64{255, 128, 0}
	for _, format := range []string{"NCHW", "NHWC"} {
		t.Run(format, func(t *testing.T) {
			values, err := PreprocessImages(format, []image.Image{img}, nil, steps)
			if err != nil {
				t.Fatal(err)
			}
			for c := range 3 {
				var got float32
				if format == "NCHW" {
					got = values[0][c][0][0]
				} else {
					got = values[0][0][0][c]
				}
				want := (pixels[c]/255 - mean[c]) / std[c]
				if math.Abs(float64(got)-want) > 1e-6 {
					t.Fatalf("channel %d: got %g, want %g", c, got, want)
				}
			}
		})
	}
	wrong, err := PreprocessImages("NCHW", []image.Image{img}, nil,
		[]imageutil.NormalizationStep{imageutil.ImagenetPixelNormalizationStep(), imageutil.RescaleStep()})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(float64(wrong[0][2][0][0])-(pixels[2]/255-mean[2])/std[2]) < 1 {
		t.Fatal("synthetic pixel does not distinguish the original incorrect ordering")
	}
}

func TestFlattenImageValuesPreservesNCHW(t *testing.T) {
	model := &Model{InputsMeta: []InputOutputInfo{{Name: "pixel_values", Dimensions: Shape{-1, 3, 2, 2}}}}
	values := [][][][]float32{{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}, {{9, 10}, {11, 12}}}}

	backing, dimensions, err := flattenImageValues(model, 1, values)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := dimensions, []int64{1, 3, 2, 2}; !equalInt64Slices(got, want) {
		t.Fatalf("unexpected dimensions: got %v, want %v", got, want)
	}
	if len(backing) != 12 || backing[0] != 1 || backing[11] != 12 {
		t.Fatalf("unexpected backing data: %v", backing)
	}
}

func TestFlattenImageValuesPreservesNHWC(t *testing.T) {
	model := &Model{InputsMeta: []InputOutputInfo{{Name: "image", Dimensions: Shape{-1, 2, 2, 3}}}}
	values := [][][][]float32{{{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}}}

	backing, dimensions, err := flattenImageValues(model, 1, values)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := dimensions, []int64{1, 2, 2, 3}; !equalInt64Slices(got, want) {
		t.Fatalf("unexpected dimensions: got %v, want %v", got, want)
	}
	if len(backing) != 12 || backing[0] != 1 || backing[11] != 12 {
		t.Fatalf("unexpected backing data: %v", backing)
	}
}

func TestReshapeOutputVisionUsesBatchSizeWithoutPaddingMask(t *testing.T) {
	meta := InputOutputInfo{Dimensions: Shape{2, 1, 2, 2}}
	values := make([]float32, 2*1*2*2)

	reshaped, ok := ReshapeOutput(values, meta, 2, nil).([][][][]float32)
	if !ok {
		t.Fatalf("unexpected output type %T", ReshapeOutput(values, meta, 2, nil))
	}
	if got, want := len(reshaped), 2; got != want {
		t.Fatalf("unexpected batch size: got %d, want %d", got, want)
	}
}

func TestReshapeOutputUsesRuntimeDimensionsForDynamicVisionOutput(t *testing.T) {
	values := []float32{1, 2, 3, 4, 5, 6, 7, 8}

	reshaped, ok := reshapeOutput(values, Shape{1, 2, 2, 2}, 1, nil).([][][][]float32)
	if !ok {
		t.Fatalf("unexpected output type %T", reshapeOutput(values, Shape{1, 2, 2, 2}, 1, nil))
	}
	if len(reshaped) != 1 || len(reshaped[0]) != 2 || len(reshaped[0][0]) != 2 || len(reshaped[0][0][0]) != 2 {
		t.Fatalf("unexpected runtime output dimensions: %#v", reshaped)
	}
}

func TestReshapeOutputIgnoresUnrelatedPaddingMask(t *testing.T) {
	values := make([]float32, 256*4)
	mask := [][]bool{make([]bool, 512)}
	for i := range mask[0] {
		mask[0][i] = true
	}

	reshaped, ok := reshapeOutput(values, Shape{1, 256, 4}, 1, mask).([][][]float32)
	if !ok {
		t.Fatalf("unexpected output type %T", reshapeOutput(values, Shape{1, 256, 4}, 1, mask))
	}
	if len(reshaped) != 1 || len(reshaped[0]) != 256 || len(reshaped[0][0]) != 4 {
		t.Fatalf("unexpected runtime output dimensions: %#v", reshaped)
	}
}

func TestReshapeOutputUsesMaskForPaddedSequenceDimension(t *testing.T) {
	values := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	mask := [][]bool{{true, false, true, false}}

	reshaped, ok := reshapeOutput(values, Shape{1, 4, 2}, 1, mask).([][][]float32)
	if !ok {
		t.Fatalf("unexpected output type %T", reshapeOutput(values, Shape{1, 4, 2}, 1, mask))
	}
	if len(reshaped) != 1 || len(reshaped[0]) != 2 || reshaped[0][0][0] != 1 || reshaped[0][0][1] != 2 || reshaped[0][1][0] != 5 || reshaped[0][1][1] != 6 {
		t.Fatalf("unexpected padded output: %#v", reshaped)
	}
}

func equalInt64Slices(a, b []int64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
