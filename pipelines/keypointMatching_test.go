package pipelines

import (
	"image"
	"math"
	"testing"

	"github.com/knights-analytics/hugot/backends"
	"github.com/stretchr/testify/require"
)

func keypointTestOutputs() map[string]backends.Tensor {
	return map[string]backends.Tensor{
		"keypoints":       {Shape: []int64{1, 2, 3, 2}, Data: []float32{.5, .25, .1, .2, .9, .8, .2, .3, 0, 0, .8, .6}},
		"matches":         {Shape: []int64{1, 2, 3}, Data: []int64{1, -1, 0, -1, -1, -1}},
		"matching_scores": {Shape: []int64{1, 2, 3}, Data: []float32{.7, 0, .9, 0, 0, 0}},
		"mask":            {Shape: []int64{1, 2, 3}, Data: []int64{1, 1, 1, 1, 0, 1}},
	}
}

func TestKeypointCoordinateScalingMasksAndScores(t *testing.T) {
	p := &KeypointMatchingPipeline{}
	result, err := p.decodeMatches(keypointTestOutputs(), [2]image.Point{{X: 200, Y: 100}, {X: 400, Y: 300}})
	require.NoError(t, err)
	require.Len(t, result, 2)
	require.Equal(t, Keypoint{X: 180, Y: 80}, result[0].KeypointImage0)
	require.Equal(t, Keypoint{X: 80, Y: 90}, result[0].KeypointImage1)
	require.Equal(t, Keypoint{X: 100, Y: 25}, result[1].KeypointImage0)
	require.Equal(t, Keypoint{X: 320, Y: 180}, result[1].KeypointImage1)
	require.Greater(t, result[0].Score, result[1].Score)
	p.Threshold = .9
	result, err = p.decodeMatches(keypointTestOutputs(), [2]image.Point{{X: 200, Y: 100}, {X: 400, Y: 300}})
	require.NoError(t, err)
	require.Empty(t, result) // upstream uses a strict > threshold.
}

func TestKeypointEmptyAndMalformedOutputs(t *testing.T) {
	p := &KeypointMatchingPipeline{}
	for _, count := range []int{0, 1, 4} {
		outputs := map[string]backends.Tensor{
			"keypoints":       {Shape: []int64{1, 2, int64(count), 2}, Data: make([]float32, 4*count)},
			"matches":         {Shape: []int64{1, 2, int64(count)}, Data: make([]int64, 2*count)},
			"matching_scores": {Shape: []int64{1, 2, int64(count)}, Data: make([]float32, 2*count)},
			"mask":            {Shape: []int64{1, 2, int64(count)}, Data: make([]bool, 2*count)},
		}
		result, err := p.decodeMatches(outputs, [2]image.Point{{X: 10, Y: 10}, {X: 20, Y: 20}})
		require.NoError(t, err)
		require.NotNil(t, result)
		require.Empty(t, result)
	}
	outputs := keypointTestOutputs()
	outputs["matching_scores"].Data.([]float32)[0] = float32(math.NaN())
	_, err := p.decodeMatches(outputs, [2]image.Point{{X: 10, Y: 10}, {X: 20, Y: 20}})
	require.Error(t, err)
	outputs = keypointTestOutputs()
	delete(outputs, "mask")
	_, err = p.decodeMatches(outputs, [2]image.Point{{X: 10, Y: 10}, {X: 20, Y: 20}})
	require.Error(t, err)
}
