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
