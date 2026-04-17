package backends

import "context"

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
// Images can be included via ImageURLs for multimodal models.
type Message struct {
	Role      string   `json:"role"`
	Content   string   `json:"content"`
	ImageURLs []string `json:"image_urls,omitempty"` // File paths or data URIs for multimodal support
}

// GenerativeModel abstracts either a generative session or engine.
type GenerativeModel interface {
	Generate(ctx context.Context, inputs [][]Message, tools []string, options *GenerativeOptions) (chan SequenceDelta, chan error, error)
	GetStatistics() PipelineStatistics
	Destroy() error
}

// GenerativeOptions contains settings for text generation.
type GenerativeOptions struct {
	MaxLength   int
	Temperature *float64
	TopP        *float64
	Seed        *int
	Guidance    *Guidance
}
