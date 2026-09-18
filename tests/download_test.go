package testutil

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/gomlx/go-huggingface/hub"
	"github.com/knights-analytics/hugot"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// test download validation

func TestDownloadValidation(t *testing.T) {
	downloadOptions := hugot.NewDownloadOptions()

	// a model with the required files in a subfolder should not error
	_, err := hugot.ValidateDownloadedHFModel(hub.New("KnightsAnalytics/distilbert-base-uncased-finetuned-sst-2-english"), downloadOptions)
	assert.NoError(t, err)
	// a model without tokenizer.json or .onnx model should error
	_, err = hugot.ValidateDownloadedHFModel(hub.New("ByteDance/SDXL-Lightning"), downloadOptions)
	assert.Error(t, err)
}

func TestDownloadModelDestinationNotDirectory(t *testing.T) {
	destination := filepath.Join(t.TempDir(), "model.onnx")
	require.NoError(t, os.WriteFile(destination, nil, 0o600))

	// Fails before contacting the hub, so no network access is needed.
	_, err := hugot.DownloadModel(context.Background(), "KnightsAnalytics/distilbert-base-uncased-finetuned-sst-2-english", destination, hugot.NewDownloadOptions())
	assert.ErrorContains(t, err, "is not a directory")
}
