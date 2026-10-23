package pipelines

import (
	"context"
	"image"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestAnyToAnyVideoFramesPreserveOrderAndTimestamps(t *testing.T) {
	p := &AnyToAnyPipeline{VideoFrameLimit: 2}
	video := &VideoInput{Frames: []image.Image{image.NewNRGBA(image.Rect(0, 0, 2, 2)), image.NewNRGBA(image.Rect(0, 0, 2, 2))}, Timestamps: []time.Duration{0, 1700 * time.Millisecond}}
	inputs := [][]MultimodalMessage{{{Content: []MultimodalContent{{Type: "text", Text: "Compare "}, {Type: "video", Video: video}, {Type: "text", Text: " these frames."}}}}}
	prepared, err := p.prepareContent(t.Context(), inputs)
	require.NoError(t, err)
	blocks := prepared[0][0].ContentBlocks
	require.Len(t, blocks, 6)
	require.Equal(t, "Compare ", blocks[0].Text)
	require.Contains(t, blocks[3].Text, "1.700000 seconds")
	require.Contains(t, blocks[2].URL, "data:image/png;base64,")
	require.Equal(t, " these frames.", blocks[5].Text)
	video.Timestamps[0] = -1
	_, err = p.prepareContent(t.Context(), inputs)
	require.ErrorContains(t, err, "timestamps")
	video.Timestamps = nil
	p.VideoFrameLimit = 1
	_, err = p.prepareContent(t.Context(), inputs)
	require.ErrorContains(t, err, "sample longer")
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	_, err = p.prepareContent(ctx, inputs)
	require.ErrorIs(t, err, context.Canceled)
}
