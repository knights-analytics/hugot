package modelconfig

import (
	"encoding/json"
	"fmt"
)

// Config contains the model metadata used by Hugot when loading a model.
type Config struct {
	MaxPositionEmbeddings int               `json:"max_position_embeddings"`
	ID2Label              map[string]string `json:"id2label"`
	TextConfig            *Config           `json:"text_config"`
}

type TokenValue struct {
	Content string
}

func (v *TokenValue) UnmarshalJSON(data []byte) error {
	var text string
	if err := json.Unmarshal(data, &text); err == nil {
		v.Content = text
		return nil
	}
	var object struct {
		Content string `json:"content"`
	}
	if err := json.Unmarshal(data, &object); err != nil {
		return fmt.Errorf("token value must be a string or object: %w", err)
	}
	v.Content = object.Content
	return nil
}

type SpecialTokensConfig struct {
	SepToken TokenValue `json:"sep_token"`
}

type TokenizerConfig struct {
	SepToken     TokenValue `json:"sep_token"`
	UnknownToken TokenValue `json:"unk_token"`
}

type TokenizerJSONConfig struct {
	PostProcessor struct {
		SpecialTokens map[string]json.RawMessage `json:"special_tokens"`
	} `json:"post_processor"`
}
