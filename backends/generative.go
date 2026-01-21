package backends

type SequenceDelta struct {
	Token string
	Index int
}

// Message represents a single message in a conversation.
// Images can be included via ImageURLs for multimodal models.
type Message struct {
	Role      string   `json:"role"`
	Content   string   `json:"content"`
	ImageURLs []string `json:"image_urls,omitempty"` // File paths or data URIs for multimodal support
}
