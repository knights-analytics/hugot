package imageutil

import (
	"bytes"
	"context"
	"errors"
	"image"
	"math"

	_ "image/gif"  // adds gif support
	_ "image/jpeg" // adds jpeg support
	_ "image/png"  // adds png support

	"github.com/knights-analytics/hugot/util/fileutil"
	_ "golang.org/x/image/webp" // adds webp support
)

func LoadImagesFromPaths(ctx context.Context, paths []string) ([]image.Image, error) {
	ctx = fileutil.WithFileSystem(ctx, nil)
	images := make([]image.Image, 0, len(paths))

	for _, path := range paths {
		b, err := fileutil.ReadFileBytes(ctx, path)
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
	bilinear   bool
}

func ResizeStep(targetSize int) *ResizePreprocessor {
	return &ResizePreprocessor{targetSize: targetSize}
}

// ResizeBilinearStep resizes the shortest edge using antialiased bilinear
// interpolation. ResizeStep retains its original nearest-neighbor behavior.
func ResizeBilinearStep(targetSize int) *ResizePreprocessor {
	return &ResizePreprocessor{targetSize: targetSize, bilinear: true}
}

func (s *ResizePreprocessor) Apply(img image.Image) (image.Image, error) {
	if img == nil || s.targetSize <= 0 || img.Bounds().Empty() {
		return nil, errors.New("resize requires a nonempty image and positive target size")
	}
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	var newW, newH int
	if w < h {
		newW = s.targetSize
		newH = int(float64(h) * float64(s.targetSize) / float64(w))
		if !s.bilinear {
			newH = int(float32(h) * float32(s.targetSize) / float32(w))
		}
	} else {
		newH = s.targetSize
		newW = int(float64(w) * float64(s.targetSize) / float64(h))
		if !s.bilinear {
			newW = int(float32(w) * float32(s.targetSize) / float32(h))
		}
	}
	if s.bilinear {
		return resizeBilinear(img, newW, newH), nil
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

// CLIPPixelNormalizationStep returns CLIP's normalization values.
// Use after RescaleStep() to normalize to 0-1 range first.
func CLIPPixelNormalizationStep() *PixelNormalizationPreprocessor {
	return &PixelNormalizationPreprocessor{
		mean: [3]float32{0.48145466, 0.4578275, 0.40821073},
		std:  [3]float32{0.26862954, 0.26130258, 0.27577711},
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
	for y := range newH {
		for x := range newW {
			srcX := srcBounds.Min.X + x*srcBounds.Dx()/newW
			srcY := srcBounds.Min.Y + y*srcBounds.Dy()/newH
			dst.Set(x, y, img.At(srcX, srcY))
		}
	}
	return dst
}

type bilinearWeight struct {
	index int
	value float64
}

func resamplingWeights(source, target int, cubic bool) [][]bilinearWeight {
	weights := make([][]bilinearWeight, target)
	scale := float64(source) / float64(target)
	filterScale := math.Max(1, scale)
	support := filterScale
	if cubic {
		support *= 2
	}
	for i := range target {
		center := (float64(i) + 0.5) * scale
		start := max(0, int(math.Floor(center-support+0.5)))
		end := min(source, int(math.Floor(center+support+0.5)))
		var total float64
		for j := start; j < end; j++ {
			distance := math.Abs((float64(j) + 0.5 - center) / filterScale)
			value := math.Max(0, 1-distance)
			if cubic {
				switch {
				case distance < 1:
					value = ((1.5*distance-2.5)*distance)*distance + 1
				case distance < 2:
					value = ((-0.5*distance+2.5)*distance-4)*distance + 2
				default:
					value = 0
				}
			}
			weights[i] = append(weights[i], bilinearWeight{index: j, value: value})
			total += value
		}
		for j := range weights[i] {
			weights[i][j].value /= total
		}
	}
	return weights
}

func resizeBilinear(img image.Image, width, height int) image.Image {
	return resizeFiltered(img, width, height, false)
}

// ResizeRGB uses Pillow-compatible antialiased bilinear (2) or bicubic (3)
// interpolation with explicit output dimensions and opaque RGB output.
func ResizeRGB(img image.Image, width, height, resample int) (image.Image, error) {
	if img == nil || img.Bounds().Empty() || width <= 0 || height <= 0 {
		return nil, errors.New("resize requires a nonempty image and positive dimensions")
	}
	if resample != 2 && resample != 3 {
		return nil, errors.New("RGB resize supports bilinear (2) or bicubic (3) interpolation")
	}
	return resizeFiltered(img, width, height, resample == 3), nil
}

func resizeFiltered(img image.Image, width, height int, cubic bool) image.Image {
	bounds := img.Bounds()
	xWeights := resamplingWeights(bounds.Dx(), width, cubic)
	yWeights := resamplingWeights(bounds.Dy(), height, cubic)
	intermediate := image.NewRGBA(image.Rect(0, 0, width, bounds.Dy()))
	for y := range bounds.Dy() {
		for x, weights := range xWeights {
			var rgb [3]float64
			for _, weight := range weights {
				r, g, b, _ := img.At(bounds.Min.X+weight.index, bounds.Min.Y+y).RGBA()
				rgb[0] += float64(r>>8) * weight.value
				rgb[1] += float64(g>>8) * weight.value
				rgb[2] += float64(b>>8) * weight.value
			}
			offset := intermediate.PixOffset(x, y)
			for c, value := range rgb {
				intermediate.Pix[offset+c] = uint8(math.Round(math.Max(0, math.Min(255, value))))
			}
			intermediate.Pix[offset+3] = 255
		}
	}
	dst := image.NewRGBA(image.Rect(0, 0, width, height))
	for y, weights := range yWeights {
		for x := range width {
			var rgb [3]float64
			for _, weight := range weights {
				offset := intermediate.PixOffset(x, weight.index)
				for c := range rgb {
					rgb[c] += float64(intermediate.Pix[offset+c]) * weight.value
				}
			}
			offset := dst.PixOffset(x, y)
			for c, value := range rgb {
				dst.Pix[offset+c] = uint8(math.Round(math.Max(0, math.Min(255, value))))
			}
			dst.Pix[offset+3] = 255
		}
	}
	return dst
}
