package pipelines

import (
	"context"
	"image"
	"image/color"
	"testing"

	"github.com/knights-analytics/hugot/backends"
	"github.com/stretchr/testify/require"
)

func TestMaskGenerationRejectsSemanticSegmentation(t *testing.T) {
	model := &backends.Model{
		InputsMeta:  []backends.InputOutputInfo{{Name: "pixel_values", Dimensions: backends.Shape{1, 3, 512, 512}}},
		OutputsMeta: []backends.InputOutputInfo{{Name: "logits", Dimensions: backends.Shape{1, 150, 128, 128}}},
	}
	_, err := NewMaskGenerationPipeline(context.Background(), backends.PipelineConfig[*MaskGenerationPipeline]{}, model)
	require.Error(t, err)
}

func TestSAMGridAndPadding(t *testing.T) {
	require.Equal(t, []float32{2, 1, 6, 1, 2, 3, 6, 3}, samPointGrid(2, image.Pt(8, 4)))
	img := image.NewRGBA(image.Rect(10, 20, 18, 24))
	for y := 20; y < 24; y++ {
		for x := 10; x < 18; x++ {
			img.Set(x, y, color.White)
		}
	}
	pixels, size := samPixels(img, 8)
	require.Equal(t, image.Pt(8, 4), size)
	require.Equal(t, []int64{1, 3, 8, 8}, pixels.Shape)
	data := pixels.Data.([]float32)
	require.InDelta(t, (1-0.485)/0.229, data[0], 0.0001)
	require.Zero(t, data[4*8])
}

func TestSAM2EncoderReachesDecoderLoad(t *testing.T) {
	model := &backends.Model{
		InputsMeta: []backends.InputOutputInfo{{Name: "pixel_values"}},
		OutputsMeta: []backends.InputOutputInfo{
			{Name: "image_embeddings.0"},
			{Name: "image_embeddings.1"},
			{Name: "image_embeddings.2"},
		},
	}
	_, err := NewMaskGenerationPipeline(context.Background(), backends.PipelineConfig[*MaskGenerationPipeline]{}, model)
	require.ErrorContains(t, err, "model loading options are not configured")
}

func TestSAM2Pixels(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 2, 2))
	for i := range img.Pix {
		img.Pix[i] = 255
	}
	pixels := sam2Pixels(img)
	data := pixels.Data.([]float32)
	require.Equal(t, []int64{1, 3, samImageSize, samImageSize}, pixels.Shape)
	require.InDelta(t, (1-0.485)/0.229, float64(data[0]), 0.0001)
	require.InDelta(t, (1-0.456)/0.224, float64(data[samImageSize*samImageSize]), 0.0001)
	require.InDelta(t, (1-0.406)/0.225, float64(data[2*samImageSize*samImageSize]), 0.0001)
}

func TestSAM2CanvasBox(t *testing.T) {
	got := sam2CanvasBox([4]float32{64, 48, 576, 432}, image.Pt(640, 480))
	require.InDelta(t, float32(102.4), float64(got[0]), 0.01)
	require.InDelta(t, float32(102.4), float64(got[1]), 0.01)
	require.InDelta(t, float32(921.6), float64(got[2]), 0.01)
	require.InDelta(t, float32(921.6), float64(got[3]), 0.01)
}

func TestSAMPromptMasksKeepLowScores(t *testing.T) {
	p := &MaskGenerationPipeline{PredictedIOUThreshold: 0.88, StabilityThreshold: 0.95, StabilityOffset: 1}
	logits := []float32{4, 4, 4, 4, 0.5, 0.5, 0.5, 0.5, -4, -4, -4, -4, 4, 4, 4, 4}
	masks := backends.Tensor{Shape: []int64{1, 4, 1, 2, 2}, Data: logits}
	scores := backends.Tensor{Shape: []int64{1, 4, 1}, Data: []float32{0.99, 0.5, 0.99, 0.99}}
	result, err := p.samPromptMasks(masks, scores, image.Pt(8, 4), image.Pt(1024, 1024))
	require.NoError(t, err)
	require.Len(t, result, 4)
	require.Equal(t, 0, result[2].Area)
	require.Equal(t, float32(0.5), result[1].PredictedIOU)
}

func TestSAMProposalFilteringAndNMS(t *testing.T) {
	p := &MaskGenerationPipeline{PredictedIOUThreshold: 0.88, StabilityThreshold: 0.95, StabilityOffset: 1}
	logits := []float32{4, 4, 4, 4, 4, 4, 4, 4, 0.5, 0.5, 0.5, 0.5, -4, -4, -4, -4}
	masks := backends.Tensor{Shape: []int64{1, 4, 1, 2, 2}, Data: logits}
	scores := backends.Tensor{Shape: []int64{1, 4, 1}, Data: []float32{0.99, 0.5, 0.99, 0.99}}
	result, err := p.samProposals(masks, scores, image.Pt(8, 4), image.Pt(1024, 512))
	require.NoError(t, err)
	require.Len(t, result, 1)
	require.Equal(t, 32, result[0].Area)
	require.Len(t, result[0].Mask, 4)
	require.Len(t, result[0].Mask[0], 8)
	require.Equal(t, float32(1), result[0].StabilityScore)
	duplicate := result[0]
	duplicate.Score = 0.9
	separate := MaskGenerationMask{Score: 0.95, Box: [4]int{20, 20, 30, 30}}
	kept := samNMS([]MaskGenerationMask{duplicate, separate, result[0]}, 0.7)
	require.Len(t, kept, 2)
	require.Equal(t, float32(0.99), kept[0].Score)
	_, err = p.samProposals(backends.Tensor{Shape: []int64{1, 150, 2, 2}, Data: logits}, scores, image.Pt(8, 4), image.Pt(1024, 512))
	require.Error(t, err)
}
