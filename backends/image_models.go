package backends

import (
	"fmt"
	"image"
	"strings"

	"github.com/knights-analytics/hugot/util/imageutil"
)

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

func CreateImageTensors(batch *PipelineBatch, model *Model, preprocessed [][][][]float32, runtime string) error {
	switch runtime {
	case "ORT":
		return createImageTensorsORT(batch, model, preprocessed)
	case "GO", "XLA":
		return createImageTensorsGoXLA(batch, model, preprocessed)
	default:
		return fmt.Errorf("runtime %s is not supported for image tensors", runtime)
	}
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
