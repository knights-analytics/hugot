package backends

type SequenceDelta struct {
	Token string
	Index int
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}
