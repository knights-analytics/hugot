package backends

import "testing"

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
