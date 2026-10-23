package pipelines

import (
	"errors"
	"image"
	"image/color"
	"math"

	"github.com/knights-analytics/hugot/util/imageutil"
)

// visionProcessor describes the FP32 RGB processors used by VideoMAE and
// SuperGlue. Unsupported transforms are rejected instead of guessed.
type visionProcessor struct {
	ProcessorType string `json:"image_processor_type"`
	Resize        bool   `json:"do_resize"`
	Size          struct {
		Height       int `json:"height"`
		Width        int `json:"width"`
		ShortestEdge int `json:"shortest_edge"`
	} `json:"size"`
	CenterCrop bool `json:"do_center_crop"`
	CropSize   struct {
		Height int `json:"height"`
		Width  int `json:"width"`
	} `json:"crop_size"`
	Resample  int       `json:"resample"`
	Rescale   bool      `json:"do_rescale"`
	Factor    float32   `json:"rescale_factor"`
	Normalize bool      `json:"do_normalize"`
	Mean      []float32 `json:"image_mean"`
	Std       []float32 `json:"image_std"`
	Grayscale bool      `json:"do_grayscale"`
}

func finite32(v float32) bool { return !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0) }

func (p visionProcessor) dimensions() (int, int, error) {
	w, h := p.Size.Width, p.Size.Height
	if p.CenterCrop {
		w, h = p.CropSize.Width, p.CropSize.Height
	}
	if !p.Resize || w <= 0 || h <= 0 || p.Size.ShortestEdge <= 0 && (p.Size.Width <= 0 || p.Size.Height <= 0) || p.Resample != 2 && p.Resample != 3 {
		return 0, 0, errors.New("processor requires resize, positive output dimensions and bilinear (2) or bicubic (3) resampling")
	}
	if p.Rescale && !finite32(p.Factor) {
		return 0, 0, errors.New("invalid processor rescale factor")
	}
	if p.Normalize {
		if len(p.Mean) != 3 || len(p.Std) != 3 {
			return 0, 0, errors.New("processor requires three-channel normalization")
		}
		for i := range p.Std {
			if !finite32(p.Mean[i]) || !finite32(p.Std[i]) || p.Std[i] <= 0 {
				return 0, 0, errors.New("invalid processor normalization")
			}
		}
	}
	return w, h, nil
}

func (p visionProcessor) pixels(img image.Image) ([]float32, error) {
	w, h, err := p.dimensions()
	if err != nil {
		return nil, err
	}
	if img == nil || img.Bounds().Empty() {
		return nil, errors.New("processor requires a nonempty image")
	}
	if p.Size.ShortestEdge > 0 {
		step := imageutil.ResizeBilinearStep(p.Size.ShortestEdge)
		if p.Resample == 3 {
			step = imageutil.ResizeBicubicStep(p.Size.ShortestEdge)
		}
		img, err = step.Apply(img)
	} else {
		img, err = imageutil.ResizeRGB(img, p.Size.Width, p.Size.Height, p.Resample)
	}
	if err != nil {
		return nil, err
	}
	if p.CenterCrop {
		if img.Bounds().Dx() < w || img.Bounds().Dy() < h {
			return nil, errors.New("processor crop exceeds resized image")
		}
		img, err = imageutil.CenterCropStep(w, h).Apply(img)
		if err != nil {
			return nil, err
		}
	}
	pixels := make([]float32, 3*w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			c := color.NRGBAModel.Convert(img.At(img.Bounds().Min.X+x, img.Bounds().Min.Y+y)).(color.NRGBA)
			v := [3]float32{float32(c.R), float32(c.G), float32(c.B)}
			if p.Rescale {
				for i := range v {
					v[i] *= p.Factor
				}
			}
			if p.Grayscale && (v[0] != v[1] || v[1] != v[2]) {
				gray := .2989*v[0] + .587*v[1] + .114*v[2]
				v = [3]float32{gray, gray, gray}
			}
			for i := range v {
				if p.Normalize {
					v[i] = (v[i] - p.Mean[i]) / p.Std[i]
				}
				pixels[i*w*h+y*w+x] = v[i]
			}
		}
	}
	return pixels, nil
}
