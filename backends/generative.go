package backends

type SequenceDelta struct {
	Token string
	Index int
}

type Message struct {
	Role      string   `json:"role"`
	Content   string   `json:"content"`
	ImageURLs []string `json:"image_urls,omitempty"` // File paths or data URIs for multimodal support
}
