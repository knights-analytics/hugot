package pipelines

import (
	"context"
	"image"
	"image/color"
	"math"
	"testing"
	"time"

	"github.com/knights-analytics/hugot/backends"
	"github.com/stretchr/testify/require"
)

func videoTestPipeline(t *testing.T) *VideoClassificationPipeline {
	t.Helper()
	model := &backends.Model{InputsMeta: []backends.InputOutputInfo{{Name: "pixel_values", Dimensions: backends.Shape{1, 3, 3, 1, 1}}}, OutputsMeta: []backends.InputOutputInfo{{Name: "logits", Dimensions: backends.Shape{1, 2}}}, IDLabelMap: map[int]string{0: "first", 1: "second"}}
	p := &VideoClassificationPipeline{BasePipeline: backends.NewBasePipeline(t.Context(), VideoClassificationConfig{}, model), NumFrames: 3, FrameSamplingRate: 2, TopK: 2, Layout: "BTCHW", FunctionToApply: "softmax"}
	p.processor = visionProcessor{Resize: true, Resample: 2}
	p.processor.Size.Width, p.processor.Size.Height = 1, 1
	return p
}

func TestVideoSamplingAndLayouts(t *testing.T) {
	p := videoTestPipeline(t)
	input := VideoInput{}
	for i := 0; i < 9; i++ {
		img := image.NewNRGBA(image.Rect(7, 9, 8, 10))
		img.SetNRGBA(7, 9, color.NRGBA{R: uint8(i), G: uint8(i + 10), B: uint8(i + 20), A: 255})
		input.Frames = append(input.Frames, img)
		input.Timestamps = append(input.Timestamps, time.Duration(i*i)*time.Millisecond)
	}
	require.NoError(t, p.Validate())
	pixels, err := p.preprocessVideo(input)
	require.NoError(t, err)
	// linspace(0,5,3,dtype=int64) -> 0,2,5, regardless of timestamps.
	require.Equal(t, []float32{0, 10, 20, 2, 12, 22, 5, 15, 25}, pixels.Data)
	require.Equal(t, []int64{1, 3, 3, 1, 1}, pixels.Shape)
	p.Layout = "BCTHW"
	pixels, err = p.preprocessVideo(input)
	require.NoError(t, err)
	require.Equal(t, []float32{0, 2, 5, 10, 12, 15, 20, 22, 25}, pixels.Data)
	input.Frames = input.Frames[:5]
	input.Timestamps = input.Timestamps[:5]
	_, err = p.preprocessVideo(input)
	require.ErrorContains(t, err, "short video")
}

func TestVideoContractsAndScores(t *testing.T) {
	p := videoTestPipeline(t)
	result, err := p.classify(backends.Tensor{Shape: []int64{1, 2}, Data: []float32{1000, 1001}})
	require.NoError(t, err)
	require.Equal(t, "second", result[0].Label)
	require.InDelta(t, .7310586, result[0].Score, 1e-6)
	p.FunctionToApply = "sigmoid"
	result, err = p.classify(backends.Tensor{Shape: []int64{1, 2}, Data: []float32{-1, 0}})
	require.NoError(t, err)
	require.Equal(t, float32(.5), result[0].Score)
	p.FunctionToApply = "none"
	result, err = p.classify(backends.Tensor{Shape: []int64{1, 2}, Data: []float32{-1, 0}})
	require.NoError(t, err)
	require.Equal(t, float32(0), result[0].Score)
	_, err = p.classify(backends.Tensor{Shape: []int64{1, 2}, Data: []float32{0, float32(math.NaN())}})
	require.Error(t, err)
	_, err = p.classify(backends.Tensor{Shape: []int64{1, 2}, Data: []float32{0}})
	require.Error(t, err)
	p.Model.InputsMeta[0].Dimensions = backends.Shape{1, 3, 1, 1}
	require.Error(t, p.Validate())
	p = videoTestPipeline(t)
	p.TopK = 0
	require.Error(t, p.Validate())
	p = videoTestPipeline(t)
	p.FunctionToApply = "unknown"
	require.Error(t, p.Validate())
}

func TestVideoCancellationAndInvalidInputs(t *testing.T) {
	p := videoTestPipeline(t)
	_, err := p.RunFrames(t.Context(), nil)
	require.Error(t, err)
	_, err = p.RunFiles(t.Context(), []string{"missing.mp4"})
	require.ErrorContains(t, err, "WithVideoDecoder")
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	_, err = p.RunFrames(ctx, []VideoInput{{}})
	require.ErrorIs(t, err, context.Canceled)
	_, err = (FFmpegVideoDecoder{}).Decode(ctx, "missing", 1)
	require.ErrorIs(t, err, context.Canceled)
	input := VideoInput{Frames: make([]image.Image, 6), Timestamps: []time.Duration{0}}
	_, err = p.preprocessVideo(input)
	require.ErrorContains(t, err, "timestamps")
	input.Timestamps = []time.Duration{0, 2, 1, 3, 4, 5}
	_, err = p.preprocessVideo(input)
	require.ErrorContains(t, err, "ordered")
}
