package testutil

import (
	"encoding/json"
	"math"
	"testing"

	"github.com/knights-analytics/hugot/pipelines"
	"github.com/knights-analytics/hugot/testcases/embedded"
	"github.com/stretchr/testify/require"
)

type hiddenStateReference struct {
	Shape   []int64 `json:"shape"`
	Samples []struct {
		Index [3]int  `json:"index"`
		Value float64 `json:"value"`
	} `json:"samples"`
}

type featureReference struct {
	SchemaVersion int     `json:"schema_version"`
	Tolerance     float64 `json:"absolute_tolerance"`
	Text          struct {
		hiddenStateReference
		Inputs []string `json:"inputs"`
	} `json:"text"`
	Image struct {
		hiddenStateReference
		RGB    [3]uint8 `json:"rgb"`
		Width  int      `json:"width"`
		Height int      `json:"height"`
	} `json:"image"`
}

func loadFeatureReference(t *testing.T) featureReference {
	t.Helper()
	var reference featureReference
	require.NoError(t, json.Unmarshal(embedded.FeatureReferenceByte, &reference))
	require.Equal(t, 1, reference.SchemaVersion)
	require.False(t, math.IsNaN(reference.Tolerance) || math.IsInf(reference.Tolerance, 0))
	require.Greater(t, reference.Tolerance, 0.0)
	require.LessOrEqual(t, reference.Tolerance, 0.1)
	return reference
}

func checkHiddenStateReference(t *testing.T, raw *pipelines.RawFeatureOutput, reference hiddenStateReference, tolerance float64) {
	t.Helper()
	require.NotNil(t, raw)
	require.Equal(t, reference.Shape, []int64(raw.Dimensions))
	require.NotEmpty(t, reference.Samples)
	for _, sample := range reference.Samples {
		b, p, c := sample.Index[0], sample.Index[1], sample.Index[2]
		require.GreaterOrEqual(t, b, 0)
		require.GreaterOrEqual(t, p, 0)
		require.GreaterOrEqual(t, c, 0)
		require.Less(t, b, len(raw.HiddenStates))
		require.Less(t, p, len(raw.HiddenStates[b]))
		require.Less(t, c, len(raw.HiddenStates[b][p]))
		value := float64(raw.HiddenStates[b][p][c])
		require.False(t, math.IsNaN(value) || math.IsInf(value, 0))
		require.InDelta(t, sample.Value, value, tolerance, "Python ONNX hidden state at %v", sample.Index)
	}
}
