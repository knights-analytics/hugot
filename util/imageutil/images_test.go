package imageutil

import (
	"image"
	"image/color"
	"testing"
)

func TestResizeBilinearAntialias(t *testing.T) {
	img := image.NewRGBA(image.Rect(7, 11, 15, 15))
	for y := img.Bounds().Min.Y; y < img.Bounds().Max.Y; y++ {
		for x := img.Bounds().Min.X; x < img.Bounds().Max.X; x++ {
			value := uint8((x - 7) % 2 * 255)
			img.SetRGBA(x, y, color.RGBA{R: value, G: value, B: value, A: 255})
		}
	}
	result, err := ResizeBilinearStep(2).Apply(img)
	if err != nil {
		t.Fatal(err)
	}
	if result.Bounds() != image.Rect(0, 0, 4, 2) {
		t.Fatalf("unexpected resized bounds: %v", result.Bounds())
	}
	for x := 1; x < 3; x++ {
		r, _, _, _ := result.At(x, 0).RGBA()
		if r>>8 < 127 || r>>8 > 128 {
			t.Fatalf("downsampling did not antialias: pixel %d = %d", x, r>>8)
		}
	}
	nearest, err := ResizeStep(2).Apply(img)
	if err != nil {
		t.Fatal(err)
	}
	r, _, _, _ := nearest.At(1, 0).RGBA()
	if r != 0 {
		t.Fatal("legacy nearest-neighbor resize changed")
	}
	bicubic, err := ResizeBicubicStep(2).Apply(img)
	if err != nil {
		t.Fatal(err)
	}
	if bicubic.Bounds() != result.Bounds() {
		t.Fatalf("bicubic bounds %v", bicubic.Bounds())
	}
	same := true
	for y := range 2 {
		for x := range 4 {
			br, _, _, _ := bicubic.At(x, y).RGBA()
			lr, _, _, _ := result.At(x, y).RGBA()
			if br != lr {
				same = false
			}
		}
	}
	if same {
		t.Fatal("bicubic resize matched bilinear")
	}
}

func TestResizeRejectsInvalidInput(t *testing.T) {
	for _, step := range []PreprocessStep{ResizeStep(0), ResizeBilinearStep(-1), ResizeBicubicStep(0)} {
		if _, err := step.Apply(image.NewRGBA(image.Rect(0, 0, 1, 1))); err == nil {
			t.Fatal("expected invalid size error")
		}
	}
	for _, img := range []image.Image{nil, image.NewRGBA(image.Rectangle{})} {
		if _, err := ResizeBilinearStep(256).Apply(img); err == nil {
			t.Fatal("expected invalid image error")
		}
	}
}
