package pipelines

import (
	"bytes"
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"image/png"
	"time"

	"github.com/knights-analytics/hugot/backends"
)

// MultimodalContent is one ordered text, image, audio or video item. Media URLs
// are paths (images also accept data URIs). Decoded videos use Video instead of
// URL. Phi-4 receives video as ordered images annotated with their timestamps.
type MultimodalContent struct {
	Type  string      `json:"type"`
	Text  string      `json:"text,omitempty"`
	URL   string      `json:"url,omitempty"`
	Video *VideoInput `json:"-"`
}

type MultimodalMessage struct {
	Role    string              `json:"role"`
	Content []MultimodalContent `json:"content"`
}

// WithAnyToAnyVideoDecoder sets the optional file decoder and the explicit
// maximum prefix frame count. Decoded inputs exceeding that limit are rejected.
func WithAnyToAnyVideoDecoder(decoder VideoDecoder, maxFrames int) AnyToAnyOption {
	return func(p *AnyToAnyPipeline) error {
		p.VideoDecoder, p.VideoFrameLimit = decoder, maxFrames
		return nil
	}
}

func (p *AnyToAnyPipeline) RunConversations(ctx context.Context, inputs [][]MultimodalMessage) (*AnyToAnyOutput, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	if err := validatePipelineContext(ctx, p.BasePipeline); err != nil {
		return nil, err
	}
	callerCtx := ctx
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	if p.SessionContext != nil {
		stop := context.AfterFunc(p.SessionContext, cancel)
		defer stop()
	}
	prepared, err := p.prepareContent(ctx, inputs)
	if err != nil {
		return nil, err
	}
	// RunMessages owns generation cancellation; do not cancel its context when a
	// streaming call returns. The temporary context above only covers decoding.
	return p.RunMessages(callerCtx, prepared)
}

func (p *AnyToAnyPipeline) prepareContent(ctx context.Context, inputs [][]MultimodalMessage) ([][]backends.Message, error) {
	result := make([][]backends.Message, len(inputs))
	for i, conversation := range inputs {
		result[i] = make([]backends.Message, len(conversation))
		for j, message := range conversation {
			out := backends.Message{Role: message.Role}
			for _, block := range message.Content {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
				if block.Type != "video" {
					if block.Video != nil {
						return nil, errors.New("decoded video requires type video")
					}
					out.ContentBlocks = append(out.ContentBlocks, backends.ContentBlock{Type: block.Type, Text: block.Text, URL: block.URL})
					continue
				}
				if block.Text != "" || (block.Video == nil) == (block.URL == "") {
					return nil, errors.New("video content requires exactly one decoded video or URL and no text")
				}
				video := block.Video
				if video == nil {
					if p.VideoDecoder == nil {
						return nil, errors.New("video paths require WithAnyToAnyVideoDecoder")
					}
					decoded, err := p.VideoDecoder.Decode(ctx, block.URL, p.VideoFrameLimit)
					if err != nil {
						return nil, err
					}
					video = &decoded
				}
				if len(video.Frames) == 0 || len(video.Frames) > p.VideoFrameLimit {
					return nil, fmt.Errorf("video requires 1 to %d frames; sample longer decoded videos explicitly", p.VideoFrameLimit)
				}
				if len(video.Timestamps) != 0 && len(video.Timestamps) != len(video.Frames) {
					return nil, errors.New("video timestamps must match frames")
				}
				var previous time.Duration
				for k, frame := range video.Frames {
					if frame == nil || frame.Bounds().Empty() {
						return nil, errors.New("video contains an empty frame")
					}
					label := fmt.Sprintf("\nFrame %d: ", k)
					if len(video.Timestamps) != 0 {
						t := video.Timestamps[k]
						if t < previous {
							return nil, errors.New("video timestamps must be nonnegative and ordered")
						}
						previous = t
						label = fmt.Sprintf("\nFrame %d at %.6f seconds: ", k, t.Seconds())
					}
					var data bytes.Buffer
					if err := png.Encode(&data, frame); err != nil {
						return nil, err
					}
					out.ContentBlocks = append(out.ContentBlocks, backends.ContentBlock{Type: "text", Text: label}, backends.ContentBlock{Type: "image", URL: "data:image/png;base64," + base64.StdEncoding.EncodeToString(data.Bytes())})
				}
			}
			result[i][j] = out
		}
	}
	return result, nil
}
