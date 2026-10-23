package pipelines

import (
	"context"
	"errors"
	"fmt"
	"image"
	"image/color"
	"math"
	"strings"

	"github.com/knights-analytics/hugot/backends"
)

const janusSystem = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."

func (n *nativeMultimodal) janusPrompt(messages []backends.Message) (string, error) {
	if len(messages) == 0 {
		return "", errors.New("empty conversation")
	}
	var text strings.Builder
	text.WriteString("<｜begin▁of▁sentence｜>")
	if n.p.GenerationMode == "text" {
		system := n.p.SystemPrompt
		if system == "" {
			system = janusSystem
		}
		if messages[0].Role == "system" {
			blocks, err := nativeMessageBlocks(messages[0])
			if err != nil {
				return "", err
			}
			var content strings.Builder
			for _, block := range blocks {
				if block.Type != "text" || block.URL != "" {
					return "", errors.New("Janus system messages require text content")
				}
				content.WriteString(block.Text)
			}
			system = content.String()
			if strings.TrimSpace(system) == "" {
				return "", errors.New("empty Janus system message")
			}
			messages = messages[1:]
		}
		text.WriteString(system)
	}
	for _, message := range messages {
		role, err := nativeRole(message.Role)
		if err != nil {
			return "", err
		}
		if role == "system" {
			return "", errors.New("Janus system content must be the first message")
		}
		blocks, err := nativeMessageBlocks(message)
		if err != nil {
			return "", err
		}
		var content strings.Builder
		for _, block := range blocks {
			if block.Type != "text" || block.URL != "" {
				return "", errors.New("this Janus runner accepts text input; use Phi-4 for image/audio/video understanding")
			}
			content.WriteString(block.Text)
		}
		if strings.TrimSpace(content.String()) == "" {
			return "", errors.New("empty Janus message")
		}
		label := "<|User|>"
		if role == "assistant" {
			label = "<|Assistant|>"
		}
		if n.p.GenerationMode == "text" {
			text.WriteString("\n\n")
		}
		text.WriteString(label + ": " + strings.TrimSpace(content.String()))
		if n.p.GenerationMode == "image" {
			text.WriteString("\n\n")
		}
	}
	if n.p.GenerationMode == "image" {
		text.WriteString("<|Assistant|>:<begin_of_image>")
	} else {
		text.WriteString("\n\n<|Assistant|>:")
	}
	return text.String(), nil
}

func (n *nativeMultimodal) generateJanus(ctx context.Context, messages []backends.Message, emit func(AnyToAnyEvent) error) error {
	prompt, err := n.janusPrompt(messages)
	if err != nil {
		return err
	}
	ids := n.encode(prompt)
	if len(ids) == 0 || len(ids) >= n.p.MaxLength || n.p.GenerationMode == "image" && len(ids)+576 > n.p.MaxLength {
		return errors.New("Janus prompt exceeds generation length budget")
	}
	imageMode := n.p.GenerationMode == "image"
	batch := int64(1)
	if imageMode {
		batch = 2
	}
	inputs := append([]int64(nil), ids...)
	if imageMode {
		unconditional := make([]int64, len(ids))
		for i, id := range ids {
			if id == 100000 || id == 100003 {
				unconditional[i] = id
			} else {
				unconditional[i] = 100015
			}
		}
		inputs = append(inputs, unconditional...)
	}
	embed, err := n.run(ctx, "embed_tokens.onnx", map[string]backends.Tensor{"input_ids": {Shape: []int64{batch, int64(len(ids))}, Data: inputs}})
	if err != nil {
		return err
	}
	language, err := n.graph(ctx, "language_model_q4.onnx")
	if err != nil {
		return err
	}
	cache, err := initialCaches(language, batch)
	if err != nil {
		return err
	}
	embeddings := embed["inputs_embeds"]
	length, previousLength := len(ids), 0
	tokens := []int64{}
	limit := n.p.MaxNewTokens
	if imageMode {
		limit = 576
	}
	for step := 0; step < limit; step++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		if length > n.p.MaxLength {
			return errors.New("Janus generation exceeds maximum length")
		}
		cache["inputs_embeds"] = embeddings
		mask := make([]int64, int(batch)*length)
		for i := range mask {
			mask[i] = 1
		}
		cache["attention_mask"] = backends.Tensor{Shape: []int64{batch, int64(length)}, Data: mask}
		if tensorPresent(language.InputsMeta, "position_ids") {
			positions := make([]int64, int(batch)*(length-previousLength))
			for b := 0; b < int(batch); b++ {
				for t := previousLength; t < length; t++ {
					positions[b*(length-previousLength)+t-previousLength] = int64(t)
				}
			}
			cache["position_ids"] = backends.Tensor{Shape: []int64{batch, int64(length - previousLength)}, Data: positions}
		}
		outputs, err := runNativeTensors(ctx, n.p.BasePipeline, language, cache)
		if err != nil {
			return err
		}
		updateCaches(cache, outputs)
		hidden, ok := outputs["last_hidden_state"]
		if !ok {
			hidden = outputs["hidden_states"]
		}
		hidden, err = lastHidden(hidden)
		if err != nil {
			return err
		}
		head := "lm_head.onnx"
		if imageMode {
			head = "gen_head.onnx"
		}
		logits, err := n.run(ctx, head, map[string]backends.Tensor{"hidden_states": hidden})
		if err != nil {
			return err
		}
		values, ok := logits["logits"].Data.([]float32)
		if !ok {
			return errors.New("Janus head must return FP32 logits")
		}
		if imageMode {
			if len(values) != 2*16384 {
				return errors.New("invalid Janus image logits")
			}
			for i := 0; i < 16384; i++ {
				values[i] = values[16384+i] + 5*(values[i]-values[16384+i])
			}
			values = values[:16384]
		}
		token, err := greedy(values)
		if err != nil {
			return err
		}
		if !imageMode && token == 100001 {
			break
		}
		tokens = append(tokens, token)
		if !imageMode {
			text, err := n.decode(tokens)
			if err != nil {
				return err
			}
			previous, err := n.decode(tokens[:len(tokens)-1])
			if err != nil {
				return err
			}
			if !strings.HasPrefix(text, previous) {
				return errors.New("Janus tokenizer changed an emitted text prefix")
			}
			if err = emit(AnyToAnyEvent{Type: "text", Text: strings.TrimPrefix(text, previous)}); err != nil {
				return err
			}
		}
		previousLength, length = length, length+1
		name, inputName := "embed_tokens.onnx", "input_ids"
		if imageMode {
			name, inputName = "gen_img_embeds.onnx", "image_ids"
		}
		feedback := []int64{token}
		if imageMode {
			feedback = append(feedback, token)
		}
		embed, err = n.run(ctx, name, map[string]backends.Tensor{inputName: {Shape: []int64{batch, 1}, Data: feedback}})
		if err != nil {
			return err
		}
		embeddings = embed["inputs_embeds"]
	}
	if imageMode {
		out, err := n.run(ctx, "image_decode.onnx", map[string]backends.Tensor{"generated_tokens": {Shape: []int64{1, int64(len(tokens))}, Data: tokens}})
		if err != nil {
			return err
		}
		img, err := decodeJanusImage(out["decoded_image"])
		if err != nil {
			return err
		}
		return emit(AnyToAnyEvent{Type: "image", Image: img})
	}
	return nil
}

func decodeJanusImage(t backends.Tensor) (image.Image, error) {
	values, ok := t.Data.([]float32)
	if !ok || len(t.Shape) != 4 || t.Shape[0] != 1 || t.Shape[1] != 3 || t.Shape[2] != 384 || t.Shape[3] != 384 || len(values) != 3*384*384 {
		return nil, fmt.Errorf("Janus image output must be FP32 [1,3,384,384], got %v", t.Shape)
	}
	img := image.NewNRGBA(image.Rect(0, 0, 384, 384))
	for y := 0; y < 384; y++ {
		for x := 0; x < 384; x++ {
			pixel := [3]uint8{}
			for c := 0; c < 3; c++ {
				v := values[c*384*384+y*384+x]
				if !finite32(v) {
					return nil, errors.New("nonfinite decoded image")
				}
				pixel[c] = uint8(math.Max(0, math.Min(255, float64((v+1)*127.5))))
			}
			img.SetNRGBA(x, y, color.NRGBA{R: pixel[0], G: pixel[1], B: pixel[2], A: 255})
		}
	}
	return img, nil
}
