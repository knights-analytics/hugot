package backends

import (
	"context"
	"errors"
	"fmt"
	"strings"
)

// GuidanceType specifies the constrained-generation strategy.
type GuidanceType string

const (
	GuidanceTypeJSONSchema  GuidanceType = "json_schema"
	GuidanceTypeRegex       GuidanceType = "regex"
	GuidanceTypeLarkGrammar GuidanceType = "lark_grammar"
)

// Guidance configures constrained (guided) generation.
// EnableFFTokens speeds up generation by force-forwarding tokens that satisfy the grammar
// without calling the model. Requires BatchSize=1.
type Guidance struct {
	Type           GuidanceType
	Data           string
	EnableFFTokens bool
}

type SequenceDelta struct {
	Token    string
	Sequence int
}

// Message represents a single message in a conversation.
// Images and audio can be included via ImageURLs and AudioURLs for multimodal models.
type Message struct {
	Role      string   `json:"role"`
	Content   string   `json:"content"`
	ImageURLs []string `json:"image_urls,omitempty"` // File paths or data URIs for multimodal support
	AudioURLs []string `json:"audio_urls,omitempty"` // File paths for multimodal support
	// ContentBlocks preserves text/image/audio interleaving. It is mutually
	// exclusive with Content, ImageURLs and AudioURLs.
	ContentBlocks []ContentBlock `json:"content_blocks,omitempty"`
	mediaInline   bool
}

// ContentBlock is one ordered text or media item. URL is a local path or, for
// images, a data URI. Model-specific placeholders are supplied by the backend.
type ContentBlock struct {
	Type string `json:"type"` // text, image or audio
	Text string `json:"text,omitempty"`
	URL  string `json:"url,omitempty"`
}

func preparePhiContentBlocks(inputs [][]Message) ([][]Message, error) {
	result := make([][]Message, len(inputs))
	for i, conversation := range inputs {
		result[i] = make([]Message, len(conversation))
		imageIndex, audioIndex := 1, 1
		for j, message := range conversation {
			if len(message.ContentBlocks) == 0 {
				imageIndex += len(message.ImageURLs)
				audioIndex += len(message.AudioURLs)
				result[i][j] = message
				continue
			}
			if message.Content != "" || len(message.ImageURLs) != 0 || len(message.AudioURLs) != 0 {
				return nil, errors.New("content blocks cannot be combined with content or attachments")
			}
			var text strings.Builder
			for _, block := range message.ContentBlocks {
				switch block.Type {
				case "text":
					if block.URL != "" || strings.Contains(block.Text, "<|image_") || strings.Contains(block.Text, "<|audio_") {
						return nil, errors.New("text blocks cannot contain media URLs or Phi placeholders")
					}
					text.WriteString(block.Text)
				case "image", "audio":
					if strings.TrimSpace(block.URL) == "" || block.Text != "" {
						return nil, errors.New("media blocks require a URL and no text")
					}
					if block.Type == "image" {
						fmt.Fprintf(&text, "<|image_%d|>", imageIndex)
						imageIndex++
						message.ImageURLs = append(message.ImageURLs, block.URL)
					} else {
						fmt.Fprintf(&text, "<|audio_%d|>", audioIndex)
						audioIndex++
						message.AudioURLs = append(message.AudioURLs, block.URL)
					}
				default:
					return nil, fmt.Errorf("unsupported content block type %q", block.Type)
				}
			}
			message.Content, message.ContentBlocks, message.mediaInline = text.String(), nil, true
			if strings.TrimSpace(message.Content) == "" && len(message.ImageURLs) == 0 && len(message.AudioURLs) == 0 {
				return nil, errors.New("content blocks are empty")
			}
			result[i][j] = message
		}
	}
	return result, nil
}

// GenerativeModel abstracts either a generative session or engine.
type GenerativeModel interface {
	Generate(ctx context.Context, inputs [][]Message, tools []string, options *GenerativeOptions) (chan SequenceDelta, chan error, error)
	GetStatistics() PipelineStatistics
	Destroy() error
}

// GenerativeOptions contains settings for text generation.
type GenerativeOptions struct {
	Temperature *float64
	TopP        *float64
	Seed        *int
	Guidance    *Guidance
	MaxLength   int
}
