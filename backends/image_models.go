package backends

import (
	"errors"
	"fmt"
	"image"
	"strings"

	"github.com/knights-analytics/hugot/util/imageutil"
)

func isImageInput(name string) bool {
	lower := strings.ToLower(name)
	return strings.Contains(lower, "pixel_values") || strings.Contains(lower, "image")
}

// DetectImageTensorFormat inspects the first image-like input and infers NHWC or NCHW.
func DetectImageTensorFormat(model *Model) (string, error) {
	inputs := model.InputsMeta
	if len(inputs) == 0 {
		return "", fmt.Errorf("no inputs found in model")
	}
	// Prefer a typical image input name
	var imgMeta InputOutputInfo
	for _, in := range inputs {
		lower := strings.ToLower(in.Name)
		if strings.Contains(lower, "pixel_values") || strings.Contains(lower, "image") {
			imgMeta = in
			break
		}
	}
	if imgMeta.Name == "" {
		imgMeta = inputs[0]
	}
	shape := imgMeta.Dimensions
	if len(shape) != 4 {
		// Unexpected for image tensors; default to NCHW
		return "NCHW", nil
	}
	// If we see channel=3 in second dim -> NCHW; if 3 in last dim -> NHWC.
	if shape[1] == 3 && shape[3] != 3 {
		return "NCHW", nil
	}
	if shape[3] == 3 {
		return "NHWC", nil
	}
	// Dynamic or unknown — default to NCHW
	return "NCHW", nil
}

func CreateImageTensors(batch *PipelineBatch, model *Model, preprocessed [][][][]float32) error {
	if model.Backend != nil {
		return model.Backend.CreateImageTensors(batch, model, preprocessed)
	}
	return fmt.Errorf("pipeline backend is not configured")
}

// PreprocessImages preprocesses images into a 4D tensor slice according to format and steps.
func PreprocessImages(format string, images []image.Image, preprocess []imageutil.PreprocessStep, normalize []imageutil.NormalizationStep) ([][][][]float32, error) {
	out := make([][][][]float32, len(images))

	for i, img := range images {
		processed := img
		for _, step := range preprocess {
			var err error
			processed, err = step.Apply(processed)
			if err != nil {
				return nil, err
			}
		}
		hh := processed.Bounds().Dy()
		ww := processed.Bounds().Dx()
		c := 3
		switch strings.ToUpper(format) {
		case "NHWC":
			tensor := make([][][]float32, hh)
			for y := range hh {
				tensor[y] = make([][]float32, ww)
				for x := range ww {
					tensor[y][x] = make([]float32, c)
				}
			}
			for y := range hh {
				for x := range ww {
					r, g, b, _ := processed.At(x, y).RGBA()
					rf, gf, bf := float32(r>>8), float32(g>>8), float32(b>>8)
					for _, step := range normalize {
						rf, gf, bf = step.Apply(rf, gf, bf)
					}
					tensor[y][x][0], tensor[y][x][1], tensor[y][x][2] = rf, gf, bf
				}
			}
			out[i] = tensor
		case "NCHW":
			tensor := make([][][]float32, c)
			for ch := range c {
				tensor[ch] = make([][]float32, hh)
				for y := range hh {
					tensor[ch][y] = make([]float32, ww)
				}
			}
			for y := range hh {
				for x := range ww {
					r, g, b, _ := processed.At(x, y).RGBA()
					rf, gf, bf := float32(r>>8), float32(g>>8), float32(b>>8)
					for _, step := range normalize {
						rf, gf, bf = step.Apply(rf, gf, bf)
					}
					tensor[0][y][x], tensor[1][y][x], tensor[2][y][x] = rf, gf, bf
				}
			}
			out[i] = tensor
		default:
			return nil, fmt.Errorf("unsupported format: %s", format)
		}
	}
	return out, nil
}

func flattenImageValues(model *Model, batchSize int, values [][][][]float32) ([]float32, []int64, error) {
	if len(values) != batchSize {
		return nil, nil, fmt.Errorf("image values do not match batch size")
	}
	if batchSize == 0 {
		return nil, nil, fmt.Errorf("image batch must not be empty")
	}
	format, err := DetectImageTensorFormat(model)
	if err != nil {
		return nil, nil, err
	}

	first := values[0]
	if len(first) == 0 || len(first[0]) == 0 || len(first[0][0]) == 0 {
		return nil, nil, errors.New("image tensor must not have empty dimensions")
	}
	var dimensions []int64
	switch format {
	case "NCHW":
		if len(first) != 3 {
			return nil, nil, errors.New("image tensor must be channel-first")
		}
		c, h, w := len(first), len(first[0]), len(first[0][0])
		dimensions = []int64{int64(batchSize), int64(c), int64(h), int64(w)}
	case "NHWC":
		if len(first[0][0]) != 3 {
			return nil, nil, errors.New("image tensor must be channel-last")
		}
		h, w, c := len(first), len(first[0]), len(first[0][0])
		dimensions = []int64{int64(batchSize), int64(h), int64(w), int64(c)}
	default:
		return nil, nil, fmt.Errorf("unsupported image tensor format %q", format)
	}

	backing := make([]float32, 0, batchSize*int(dimensions[1])*int(dimensions[2])*int(dimensions[3]))
	for i, imageValue := range values {
		if format == "NCHW" {
			if len(imageValue) != int(dimensions[1]) {
				return nil, nil, fmt.Errorf("image %d has inconsistent channel count", i)
			}
			for _, channel := range imageValue {
				if len(channel) != int(dimensions[2]) {
					return nil, nil, fmt.Errorf("image %d has inconsistent height", i)
				}
				for _, row := range channel {
					if len(row) != int(dimensions[3]) {
						return nil, nil, fmt.Errorf("image %d has inconsistent width", i)
					}
					backing = append(backing, row...)
				}
			}
			continue
		}
		if len(imageValue) != int(dimensions[1]) {
			return nil, nil, fmt.Errorf("image %d has inconsistent height", i)
		}
		for _, row := range imageValue {
			if len(row) != int(dimensions[2]) {
				return nil, nil, fmt.Errorf("image %d has inconsistent width", i)
			}
			for _, pixel := range row {
				if len(pixel) != int(dimensions[3]) {
					return nil, nil, fmt.Errorf("image %d has inconsistent channel count", i)
				}
				backing = append(backing, pixel...)
			}
		}
	}
	return backing, dimensions, nil
}
