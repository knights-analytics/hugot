package backends

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/knights-analytics/hugot/options"
)

func TestLoadModelWithOutputsRejectsUnsupportedBackends(t *testing.T) {
	t.Parallel()

	opts := options.Defaults()
	opts.Backend = "GO"
	_, err := LoadModelWithOutputs(t.Context(), "/tmp/model", "model.onnx", []string{"last_hidden_state"}, opts, false)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "only supported for non-generative ORT models")

	opts.Backend = "ORT"
	_, err = LoadModelWithOutputs(t.Context(), "/tmp/model", "", []string{"last_hidden_state"}, opts, true)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "only supported for non-generative ORT models")
}
