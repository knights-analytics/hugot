package pipelines

import (
	"image"
	"image/color"
	"testing"

	"github.com/knights-analytics/hugot/util/imageutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestImagePreprocessing(t *testing.T) {
	// Create a simple test image (10x10 red image)
	img := image.NewRGBA(image.Rect(0, 0, 10, 10))
	red := color.RGBA{255, 0, 0, 255}
	for y := 0; y < 10; y++ {
		for x := 0; x < 10; x++ {
			img.Set(x, y, red)
		}
	}

	// Create a pipeline with image mode (without a model, just for preprocessing test)
	pipeline := &FeatureExtractionPipeline{
		ImageMode:   true,
		imageFormat: "NCHW",
		preprocessSteps: []imageutil.PreprocessStep{
			imageutil.ResizeStep(5), // resize to 5x5
			imageutil.CenterCropStep(5, 5),
		},
		normalizationSteps: []imageutil.NormalizationStep{
			imageutil.RescaleStep(), // scale to 0-1
		},
	}

	// Test preprocessing
	images := []image.Image{img}
	tensors, err := pipeline.preprocessImages(images)
	require.NoError(t, err)
	require.Len(t, tensors, 1)

	// NCHW format: [Channels][Height][Width]
	tensor := tensors[0]
	assert.Len(t, tensor, 3, "should have 3 channels")
	assert.Len(t, tensor[0], 5, "height should be 5")
	assert.Len(t, tensor[0][0], 5, "width should be 5")

	// Red channel should be ~1.0 (255/255), green and blue ~0
	assert.InDelta(t, 1.0, tensor[0][0][0], 0.01, "red channel should be ~1.0")
	assert.InDelta(t, 0.0, tensor[1][0][0], 0.01, "green channel should be ~0")
	assert.InDelta(t, 0.0, tensor[2][0][0], 0.01, "blue channel should be ~0")
}

func TestImagePreprocessingNHWC(t *testing.T) {
	// Create a simple test image (10x10 green image)
	img := image.NewRGBA(image.Rect(0, 0, 10, 10))
	green := color.RGBA{0, 255, 0, 255}
	for y := 0; y < 10; y++ {
		for x := 0; x < 10; x++ {
			img.Set(x, y, green)
		}
	}

	pipeline := &FeatureExtractionPipeline{
		ImageMode:   true,
		imageFormat: "NHWC",
		preprocessSteps: []imageutil.PreprocessStep{
			imageutil.CenterCropStep(5, 5),
		},
		normalizationSteps: []imageutil.NormalizationStep{
			imageutil.RescaleStep(),
		},
	}

	images := []image.Image{img}
	tensors, err := pipeline.preprocessImages(images)
	require.NoError(t, err)
	require.Len(t, tensors, 1)

	// NHWC format: [Height][Width][Channels]
	tensor := tensors[0]
	assert.Len(t, tensor, 5, "height should be 5")
	assert.Len(t, tensor[0], 5, "width should be 5")
	assert.Len(t, tensor[0][0], 3, "should have 3 channels")

	// Green channel should be ~1.0, red and blue ~0
	assert.InDelta(t, 0.0, tensor[0][0][0], 0.01, "red channel should be ~0")
	assert.InDelta(t, 1.0, tensor[0][0][1], 0.01, "green channel should be ~1.0")
	assert.InDelta(t, 0.0, tensor[0][0][2], 0.01, "blue channel should be ~0")
}

func TestCLIPNormalization(t *testing.T) {
	// Create a simple test image (white pixel)
	img := image.NewRGBA(image.Rect(0, 0, 1, 1))
	white := color.RGBA{255, 255, 255, 255}
	img.Set(0, 0, white)

	pipeline := &FeatureExtractionPipeline{
		ImageMode:   true,
		imageFormat: "NCHW",
		normalizationSteps: []imageutil.NormalizationStep{
			imageutil.RescaleStep(),             // 255 -> 1.0
			imageutil.CLIPPixelNormalizationStep(), // apply CLIP normalization
		},
	}

	images := []image.Image{img}
	tensors, err := pipeline.preprocessImages(images)
	require.NoError(t, err)

	// After CLIP normalization of white (1.0, 1.0, 1.0):
	// R: (1.0 - 0.48145466) / 0.26862954 ≈ 1.93
	// G: (1.0 - 0.4578275) / 0.26130258 ≈ 2.07
	// B: (1.0 - 0.40821073) / 0.27577711 ≈ 2.15
	tensor := tensors[0]
	assert.InDelta(t, 1.93, tensor[0][0][0], 0.1, "CLIP normalized red")
	assert.InDelta(t, 2.07, tensor[1][0][0], 0.1, "CLIP normalized green")
	assert.InDelta(t, 2.15, tensor[2][0][0], 0.1, "CLIP normalized blue")
}

func TestImageModeValidation(t *testing.T) {
	// RunWithImages should fail if ImageMode is not enabled
	pipeline := &FeatureExtractionPipeline{
		ImageMode: false,
	}

	img := image.NewRGBA(image.Rect(0, 0, 10, 10))
	_, err := pipeline.RunWithImages([]image.Image{img})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "ImageMode")
}

func TestWithImageModeOption(t *testing.T) {
	pipeline := &FeatureExtractionPipeline{}

	err := WithImageMode()(pipeline)
	require.NoError(t, err)
	assert.True(t, pipeline.ImageMode)
}

func TestWithImageFormatOption(t *testing.T) {
	pipeline := &FeatureExtractionPipeline{}

	err := WithImageFormat("NHWC")(pipeline)
	require.NoError(t, err)
	assert.Equal(t, "NHWC", pipeline.imageFormat)
}

func TestWithImagePreprocessStepsOption(t *testing.T) {
	pipeline := &FeatureExtractionPipeline{}

	err := WithImagePreprocessSteps(
		imageutil.ResizeStep(224),
		imageutil.CenterCropStep(224, 224),
	)(pipeline)
	require.NoError(t, err)
	assert.Len(t, pipeline.preprocessSteps, 2)
}

func TestWithImageNormalizationStepsOption(t *testing.T) {
	pipeline := &FeatureExtractionPipeline{}

	err := WithImageNormalizationSteps(
		imageutil.RescaleStep(),
		imageutil.CLIPPixelNormalizationStep(),
	)(pipeline)
	require.NoError(t, err)
	assert.Len(t, pipeline.normalizationSteps, 2)
}
