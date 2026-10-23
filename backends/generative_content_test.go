package backends

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestOrderedPhiContentPreservesMediaPositionsAndIndices(t *testing.T) {
	input := [][]Message{{
		{Role: "user", ContentBlocks: []ContentBlock{{Type: "text", Text: "Before "}, {Type: "image", URL: "one.png"}, {Type: "text", Text: " between "}, {Type: "audio", URL: "one.wav"}, {Type: "text", Text: " after"}}},
		{Role: "user", ContentBlocks: []ContentBlock{{Type: "image", URL: "two.png"}, {Type: "audio", URL: "two.wav"}}},
	}, {{Role: "user", ContentBlocks: []ContentBlock{{Type: "image", URL: "three.png"}}}}}
	prepared, err := preparePhiContentBlocks(input)
	require.NoError(t, err)
	require.Equal(t, "Before <|image_1|> between <|audio_1|> after", prepared[0][0].Content)
	require.Equal(t, "<|image_2|><|audio_2|>", prepared[0][1].Content)
	require.Equal(t, "<|image_1|>", prepared[1][0].Content)
	require.Equal(t, []string{"one.png"}, prepared[0][0].ImageURLs)
	require.Equal(t, []string{"one.wav"}, prepared[0][0].AudioURLs)
	require.Empty(t, input[0][0].Content)
	for _, message := range []Message{
		{Content: "ambiguous", ContentBlocks: []ContentBlock{{Type: "text", Text: "test"}}},
		{ContentBlocks: []ContentBlock{{Type: "unknown"}}},
		{ContentBlocks: []ContentBlock{{Type: "image"}}},
		{ContentBlocks: []ContentBlock{{Type: "text", Text: "<|image_1|>"}}},
	} {
		_, err = preparePhiContentBlocks([][]Message{{message}})
		require.Error(t, err)
	}
}
