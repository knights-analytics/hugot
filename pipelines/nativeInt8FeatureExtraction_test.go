package pipelines

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/options"
)

func TestNewNativeInt8FeatureExtractionPipelineRequiresSingleSelectedOutput(t *testing.T) {
	t.Parallel()

	opts := options.Defaults()
	opts.Backend = "ORT"
	model := &backends.Model{
		OnnxOutputNames: []string{"pooler_output_int8"},
		OutputsMeta:     []backends.InputOutputInfo{{Name: "pooler_output_int8"}},
		Tokenizer:       &backends.Tokenizer{},
	}

	_, err := NewNativeInt8FeatureExtractionPipeline(t.Context(), backends.PipelineConfig[*NativeInt8FeatureExtractionPipeline]{
		Name:            "zero",
		OnnxOutputNames: nil,
	}, opts, model)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "exactly one")

	_, err = NewNativeInt8FeatureExtractionPipeline(t.Context(), backends.PipelineConfig[*NativeInt8FeatureExtractionPipeline]{
		Name:            "multi",
		OnnxOutputNames: []string{"a", "b"},
	}, opts, model)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "exactly one")
}

func TestNewNativeInt8FeatureExtractionPipelineRequiresORT(t *testing.T) {
	t.Parallel()

	opts := options.Defaults()
	opts.Backend = "GO"
	_, err := NewNativeInt8FeatureExtractionPipeline(t.Context(), backends.PipelineConfig[*NativeInt8FeatureExtractionPipeline]{
		Name:            "go",
		OnnxOutputNames: []string{"pooler_output_int8"},
	}, opts, &backends.Model{})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "ORT")
}

func TestInt8FeatureExtractionOutputGetOutputPreservesVectors(t *testing.T) {
	t.Parallel()

	out := &Int8FeatureExtractionOutput{Embeddings: [][]int8{{-128, 0, 127}, {1, 2, 3}}}
	got := out.GetOutput()
	require.Len(t, got, 2)
	assert.Equal(t, []int8{-128, 0, 127}, got[0])
	assert.Equal(t, []int8{1, 2, 3}, got[1])
}
