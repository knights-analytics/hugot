//go:build cgo && (ORT || ALL) && !TRAINING

package ort_test

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/pipelines"
)

type pplxManifest struct {
	ORTVersion     string `json:"ort_version"`
	SelectedOutput string `json:"selected_output"`
	Dimension      int    `json:"dimension"`
	Graph          struct {
		Path string `json:"path"`
	} `json:"graph"`
}

type pplxFixture struct {
	SelectedOutput string     `json:"selected_output"`
	OutputDtype    string     `json:"output_dtype"`
	Dimension      int        `json:"dimension"`
	ORTVersion     string     `json:"ort_version"`
	Cases          []pplxCase `json:"cases"`
}

type pplxCase struct {
	Name   string   `json:"name"`
	Texts  []string `json:"texts"`
	Inputs []struct {
		TokenIDs       []uint32 `json:"token_ids"`
		AttentionMask  []uint32 `json:"attention_mask"`
		SequenceLength int      `json:"sequence_length"`
	} `json:"inputs"`
	ExpectedBytesB64 string `json:"expected_bytes_b64"`
	ExpectedSHA256   string `json:"expected_sha256"`
	BatchSize        int    `json:"batch_size"`
}

// TestPPLXNativeInt8Parity is the opt-in PPLX Q4 native INT8 qualification harness.
//
// It skips unless HUGOT_PPLX_MODEL_DIR points at a verified filesystem model directory
// containing the pinned Q4 graph, external data, and tokenizer files. Optional overrides:
// HUGOT_PPLX_MANIFEST, HUGOT_PPLX_FIXTURE, HUGOT_ORT_LIBRARY_DIR.
func TestPPLXNativeInt8Parity(t *testing.T) {
	modelDir := os.Getenv("HUGOT_PPLX_MODEL_DIR")
	if modelDir == "" {
		t.Skip("HUGOT_PPLX_MODEL_DIR not set; skipping opt-in PPLX INT8 parity test")
	}

	manifestPath := getenvDefault("HUGOT_PPLX_MANIFEST", filepath.Join("..", "..", "testdata", "pplx", "manifest.json"))
	fixturePath := getenvDefault("HUGOT_PPLX_FIXTURE", filepath.Join("..", "..", "testdata", "pplx", "golden_fixture.json"))

	manifest := loadPPLXManifest(t, manifestPath)
	fixture := loadPPLXFixture(t, fixturePath)
	require.Equal(t, "pooler_output_int8", manifest.SelectedOutput)
	require.Equal(t, "pooler_output_int8", fixture.SelectedOutput)
	require.Equal(t, "INT8", fixture.OutputDtype)
	require.Equal(t, 1024, fixture.Dimension)

	var opts []options.WithOption
	if libDir := os.Getenv("HUGOT_ORT_LIBRARY_DIR"); libDir != "" {
		opts = append(opts, options.WithOnnxLibraryPath(libDir))
	}
	session, err := hugot.NewORTSession(t.Context(), opts...)
	require.NoError(t, err)
	defer func() { require.NoError(t, session.Destroy()) }()

	pipeline, err := hugot.NewPipeline(session, hugot.NativeInt8FeatureExtractionConfig{
		ModelPath:       modelDir,
		Name:            "pplxNativeInt8",
		OnnxFilename:    manifest.Graph.Path,
		OnnxOutputNames: []string{"pooler_output_int8"},
	})
	require.NoError(t, err)
	require.Len(t, pipeline.Model.OutputsMeta, 1)
	assert.Equal(t, "pooler_output_int8", pipeline.Model.OutputsMeta[0].Name)
	assert.Equal(t, []string{"pooler_output_int8"}, pipeline.Model.OnnxOutputNames)
	assert.NotContains(t, backends.GetNames(pipeline.Model.OutputsMeta), "last_hidden_state")

	for _, testCase := range fixture.Cases {
		t.Run(testCase.Name, func(t *testing.T) {
			assertTokenParity(t, pipeline, testCase)
			result, runErr := pipeline.RunPipeline(t.Context(), testCase.Texts)
			require.NoError(t, runErr)
			require.Len(t, result.Embeddings, testCase.BatchSize)
			for _, vector := range result.Embeddings {
				require.Len(t, vector, fixture.Dimension)
			}

			actual := flattenInt8(result.Embeddings)
			expected, decErr := base64.StdEncoding.DecodeString(testCase.ExpectedBytesB64)
			require.NoError(t, decErr)
			assert.Equal(t, expected, actual)
			digest := sha256.Sum256(actual)
			assert.Equal(t, testCase.ExpectedSHA256, hex.EncodeToString(digest[:]))
		})
	}

	// Repeated inference on the same long-lived session.
	first := fixture.Cases[0]
	again, err := pipeline.RunPipeline(t.Context(), first.Texts)
	require.NoError(t, err)
	expected, err := base64.StdEncoding.DecodeString(first.ExpectedBytesB64)
	require.NoError(t, err)
	assert.Equal(t, expected, flattenInt8(again.Embeddings))

	require.NoError(t, hugot.ClosePipeline[*pipelines.NativeInt8FeatureExtractionPipeline](session, "pplxNativeInt8"))

	// Fresh session after close must load and run again.
	reopened, err := hugot.NewPipeline(session, hugot.NativeInt8FeatureExtractionConfig{
		ModelPath:       modelDir,
		Name:            "pplxNativeInt8Reopen",
		OnnxFilename:    manifest.Graph.Path,
		OnnxOutputNames: []string{"pooler_output_int8"},
	})
	require.NoError(t, err)
	reopenResult, err := reopened.RunPipeline(t.Context(), first.Texts)
	require.NoError(t, err)
	assert.Equal(t, expected, flattenInt8(reopenResult.Embeddings))
}

func assertTokenParity(t *testing.T, pipeline *pipelines.NativeInt8FeatureExtractionPipeline, testCase pplxCase) {
	t.Helper()
	batch := backends.NewBatch(len(testCase.Texts))
	defer func() { require.NoError(t, batch.Destroy()) }()
	backends.TokenizeInputs(batch, pipeline.Model.Tokenizer, testCase.Texts)
	require.Len(t, batch.Input, len(testCase.Inputs))
	for i, expected := range testCase.Inputs {
		assert.Equal(t, expected.TokenIDs, batch.Input[i].TokenIDs, "token id mismatch for case %s index %d", testCase.Name, i)
		assert.Equal(t, expected.AttentionMask, batch.Input[i].AttentionMask, "attention mask mismatch for case %s index %d", testCase.Name, i)
	}
}

func flattenInt8(vectors [][]int8) []byte {
	if len(vectors) == 0 {
		return nil
	}
	out := make([]byte, 0, len(vectors)*len(vectors[0]))
	for _, vector := range vectors {
		for _, value := range vector {
			out = append(out, byte(value))
		}
	}
	return out
}

func loadPPLXManifest(t *testing.T, path string) pplxManifest {
	t.Helper()
	raw, err := os.ReadFile(path) // #nosec G304 -- test reads explicit fixture path
	require.NoError(t, err)
	var manifest pplxManifest
	require.NoError(t, json.Unmarshal(raw, &manifest))
	return manifest
}

func loadPPLXFixture(t *testing.T, path string) pplxFixture {
	t.Helper()
	raw, err := os.ReadFile(path) // #nosec G304 -- test reads explicit fixture path
	require.NoError(t, err)
	var fixture pplxFixture
	require.NoError(t, json.Unmarshal(raw, &fixture))
	return fixture
}

func getenvDefault(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}
