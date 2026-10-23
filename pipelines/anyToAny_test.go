package pipelines

import (
	"testing"

	"github.com/knights-analytics/hugot/backends"
	"github.com/stretchr/testify/require"
)

func TestAnyToAnyMixedMediaConversations(t *testing.T) {
	input := [][]backends.Message{
		{{Role: "system", Content: "answer briefly"}, {Content: "compare", ImageURLs: []string{"one.png"}, AudioURLs: []string{"one.wav"}}, {Content: "and this", AudioURLs: []string{"two.wav"}}},
		{{Content: "text only"}},
	}
	prepared, err := anyToAnyMessages(input)
	require.NoError(t, err)
	require.Equal(t, "<|audio_1|>\ncompare", prepared[0][1].Content)
	require.Equal(t, "<|audio_2|>\nand this", prepared[0][2].Content)
	require.Equal(t, "user", prepared[0][1].Role)
	require.Equal(t, "system", prepared[0][0].Role)
	require.Equal(t, "text only", prepared[1][0].Content)
	prepared[0][1].ImageURLs[0] = "changed"
	require.Equal(t, "one.png", input[0][1].ImageURLs[0])
	require.Equal(t, "compare", input[0][1].Content)
}

func TestAnyToAnyRejectsMissingCapabilitiesAndEmptyInputs(t *testing.T) {
	for _, input := range [][][]backends.Message{nil, {{}}, {{{Content: "<|audio_1|>"}}}, {{{ImageURLs: []string{""}}}}} {
		_, err := anyToAnyMessages(input)
		require.Error(t, err)
	}
	_, err := NewAnyToAnyPipeline(t.Context(), AnyToAnyConfig{}, nil)
	require.Error(t, err)
	_, err = NewAnyToAnyPipeline(t.Context(), AnyToAnyConfig{}, &backends.Model{})
	require.Error(t, err)
}
