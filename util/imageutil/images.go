package imageutil

import (
	"bytes"
	"image"

	"github.com/knights-analytics/hugot/util/fileutil"
)

func LoadImagesFromPaths(paths []string) ([]image.Image, error) {
	images := make([]image.Image, 0, len(paths))

	for _, path := range paths {
		b, err := fileutil.ReadFileBytes(path)
		if err != nil {
			return nil, err
		}
		img, _, err := image.Decode(bytes.NewReader(b))
		if err != nil {
			return nil, err
		}
		images = append(images, img)
	}
	return images, nil
}

type PreprocessStep interface {
	Apply(img image.Image) (image.Image, error)
}

type ResizePreprocessor struct {
	targetSize int
}

func ResizeStep(targetSize int) *ResizePreprocessor {
	return &ResizePreprocessor{targetSize: targetSize}
}

func (s *ResizePreprocessor) Apply(img image.Image) (image.Image, error) {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	var newW, newH int
	if w < h {
		newW = s.targetSize
		newH = int(float32(h) * float32(s.targetSize) / float32(w))
	} else {
		newH = s.targetSize
		newW = int(float32(w) * float32(s.targetSize) / float32(h))
	}
	return resizeImage(img, newW, newH), nil
}

func CenterCropStep(targetWidth, targetHeight int) *CenterCropPreprocessor {
	return &CenterCropPreprocessor{targetWidth: targetWidth, targetHeight: targetHeight}
}

type CenterCropPreprocessor struct {
	targetWidth  int
	targetHeight int
}

func (s *CenterCropPreprocessor) Apply(img image.Image) (image.Image, error) {
	bounds := img.Bounds()
	x0 := bounds.Min.X + (bounds.Dx()-s.targetWidth)/2
	y0 := bounds.Min.Y + (bounds.Dy()-s.targetHeight)/2
	rect := image.Rect(0, 0, s.targetWidth, s.targetHeight)
	dst := image.NewRGBA(rect)
	for y := 0; y < s.targetHeight; y++ {
		for x := 0; x < s.targetWidth; x++ {
			dst.Set(x, y, img.At(x0+x, y0+y))
		}
	}
	return dst, nil
}

type NormalizationStep interface {
	Apply(r, g, b float32) (float32, float32, float32)
}

type PixelNormalizationPreprocessor struct {
	mean [3]float32
	std  [3]float32
}

func (s *PixelNormalizationPreprocessor) Apply(r, g, b float32) (float32, float32, float32) {
	r = (r - s.mean[0]) / s.std[0]
	g = (g - s.mean[1]) / s.std[1]
	b = (b - s.mean[2]) / s.std[2]
	return r, g, b
}

func PixelNormalizationStep(mean, std [3]float32) *PixelNormalizationPreprocessor {
	return &PixelNormalizationPreprocessor{mean: mean, std: std}
}

func ImagenetPixelNormalizationStep() *PixelNormalizationPreprocessor {
	return &PixelNormalizationPreprocessor{
		mean: [3]float32{0.485, 0.456, 0.406},
		std:  [3]float32{0.229, 0.224, 0.225},
	}
}

type RescalePreprocessor struct{}

func (s *RescalePreprocessor) Apply(r, g, b float32) (float32, float32, float32) {
	scale := float32(1.0 / 255.0)
	return r * scale, g * scale, b * scale
}

func RescaleStep() *RescalePreprocessor {
	return &RescalePreprocessor{}
}

// resizeImage resizes an image to the given width and height using nearest neighbor (simple, replace with better if needed).
func resizeImage(img image.Image, newW, newH int) image.Image {
	dst := image.NewRGBA(image.Rect(0, 0, newW, newH))
	srcBounds := img.Bounds()
	for y := 0; y < newH; y++ {
		for x := 0; x < newW; x++ {
			srcX := srcBounds.Min.X + x*srcBounds.Dx()/newW
			srcY := srcBounds.Min.Y + y*srcBounds.Dy()/newH
			dst.Set(x, y, img.At(srcX, srcY))
		}
	}
	return dst
}
