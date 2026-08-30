package testutil

import (
	"testing"

	"github.com/gomlx/go-huggingface/hub"
	"github.com/knights-analytics/hugot"
	"github.com/stretchr/testify/assert"
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
