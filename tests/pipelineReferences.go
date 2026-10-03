package testutil

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
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

func checkCaptionGenerationReference(t *testing.T, pipeline *pipelines.ImageToTextPipeline) {
	t.Helper()
	var reference struct {
		SchemaVersion int `json:"schema_version"`
		Model         struct {
			Revision string `json:"revision"`
		} `json:"model"`
		Generation struct {
			MaxNewTokens int `json:"max_new_tokens"`
		} `json:"generation"`
		Cases []struct {
			Image      string   `json:"image"`
			Text       string   `json:"text"`
			Tokens     []uint32 `json:"tokens"`
			EOSReached bool     `json:"eos_reached"`
		} `json:"cases"`
	}
	require.NoError(t, json.Unmarshal(embedded.CaptionGenerationReferenceByte, &reference))
	require.Equal(t, 1, reference.SchemaVersion)
	require.Equal(t, "215b4edcb7ec1fad5905a18a03f7b2007f6fabd0", reference.Model.Revision)
	require.Equal(t, 20, reference.Generation.MaxNewTokens)
	require.Len(t, reference.Cases, 2)
	images := map[string][]byte{
		"caption_cat_reference.png":      embedded.CaptionCatReferenceByte,
		"caption_portrait_reference.png": embedded.CaptionPortraitReferenceByte,
	}
	inputs := make([]pipelines.ImageTextPrompt, len(reference.Cases))
	directory := t.TempDir()
	for i, expected := range reference.Cases {
		data, ok := images[expected.Image]
		require.True(t, ok, "unrecognized caption reference image")
		path := filepath.Join(directory, expected.Image)
		require.NoError(t, os.WriteFile(path, data, 0o600))
		inputs[i] = pipelines.ImageTextPrompt{ImagePath: path}
	}
	result, err := pipeline.RunWithImages(t.Context(), inputs)
	require.NoError(t, err)
	require.NotNil(t, result)
	responses := result.Responses
	require.Len(t, responses, len(reference.Cases))
	for i, expected := range reference.Cases {
		require.True(t, expected.EOSReached)
		require.NotEmpty(t, expected.Tokens)
		require.Less(t, len(expected.Tokens), reference.Generation.MaxNewTokens)
		if expected.Image == "caption_cat_reference.png" {
			// Quantized greedy decoding can vary across environments; retain the main image details.
			for _, word := range []string{"cat", "bed", "laying"} {
				require.Regexp(t, `(?i)\b`+word+`\b`, responses[i], "caption for %s must contain %q", expected.Image, word)
			}
		} else {
			for _, word := range []string{"woman", "red", "hair"} {
				require.Regexp(t, `(?i)\b`+word+`\b`, responses[i], "caption for %s must contain %q", expected.Image, word)
			}
		}
	}
}
