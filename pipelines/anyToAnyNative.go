package pipelines

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"slices"
	"strings"
	"unicode/utf8"

	"github.com/gomlx/go-huggingface/tokenizers/api"
	"github.com/gomlx/go-huggingface/tokenizers/hftokenizer"
	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/util/fileutil"
)

// AnyToAnyResult owns a generated response. Audio samples use Audio.SampleRate;
// images are decoded RGB pixels. Model capabilities determine populated fields.
type AnyToAnyResult struct {
	Text   string
	Images []image.Image
	Audio  *AudioWaveform
}

// AnyToAnyEvent carries a text delta or a complete generated media artifact.
// Events for each conversation are ordered; Sequence identifies the batch item.
// Audio is emitted after codec reconstruction, not as speculative PCM chunks.
type AnyToAnyEvent struct {
	Sequence int
	Type     string // text, image, audio or done
	Text     string
	Image    image.Image
	Audio    *AudioWaveform
}

type AnyToAnyOutput struct {
	Responses   []string
	Results     []AnyToAnyResult
	TokenStream chan backends.SequenceDelta
	EventStream chan AnyToAnyEvent
	ErrorStream chan error
}

func (o *AnyToAnyOutput) GetOutput() []any {
	if o.EventStream != nil {
		return []any{o.EventStream, o.ErrorStream}
	}
	if o.TokenStream != nil {
		return []any{o.TokenStream, o.ErrorStream}
	}
	if o.Results != nil {
		out := make([]any, len(o.Results))
		for i := range out {
			out[i] = o.Results[i]
		}
		return out
	}
	out := make([]any, len(o.Responses))
	for i := range out {
		out[i] = o.Responses[i]
	}
	return out
}

type nativeMultimodal struct {
	family           string
	p                *AnyToAnyPipeline
	tokenizer        *hftokenizer.Tokenizer
	lfmEmbeddingTable *lfmEmbeddingTable
}

func newNativeMultimodal(ctx context.Context, p *AnyToAnyPipeline) (*nativeMultimodal, error) {
	if p.Model.ORTModel == nil {
		return nil, errors.New("native multimodal graph bundles currently require the ORT backend")
	}
	var config struct {
		ModelType     string   `json:"model_type"`
		Architectures []string `json:"architectures"`
	}
	if err := readNativeConfig(ctx, p.Model.Path, "config.json", &config); err != nil {
		return nil, fmt.Errorf("native any-to-any requires model config.json: %w", err)
	}
	family := ""
	switch {
	case config.ModelType == "multi_modality":
		family = "janus"
	case slices.Contains(config.Architectures, "Lfm2AudioForConditionalGeneration"):
		family = "lfm2_audio"
	default:
		return nil, fmt.Errorf("unsupported native any-to-any model type %q with architectures %v", config.ModelType, config.Architectures)
	}
	if p.Model.Tokenizer == nil || p.Model.Tokenizer.GoTokenizer == nil {
		return nil, errors.New("native multimodal bundle requires tokenizer.json")
	}
	data, err := fileutil.ReadFileBytes(ctx, fileutil.PathJoinSafe(p.Model.Path, "tokenizer.json"))
	if err != nil {
		return nil, err
	}
	tokenizer, err := newNativeTokenizer(data, family)
	if err != nil {
		return nil, err
	}
	n := &nativeMultimodal{family: family, p: p, tokenizer: tokenizer}
	if family == "lfm2_audio" {
		n.lfmEmbeddingTable = &lfmEmbeddingTable{}
	}
	return n, nil
}

func newNativeTokenizer(data []byte, family string) (*hftokenizer.Tokenizer, error) {
	// The tokenizer dependency skips ByteLevel inside a Sequence decoder.
	// Unwrap equivalent single-step sequences without changing tokenization.
	var tokenizerConfig map[string]json.RawMessage
	if err := json.Unmarshal(data, &tokenizerConfig); err != nil {
		return nil, err
	}
	var err error
	var decoder struct {
		Type     string            `json:"type"`
		Decoders []json.RawMessage `json:"decoders"`
	}
	if err = json.Unmarshal(tokenizerConfig["decoder"], &decoder); err != nil {
		return nil, err
	}
	if decoder.Type == "Sequence" && len(decoder.Decoders) == 1 {
		tokenizerConfig["decoder"] = decoder.Decoders[0]
		data, err = json.Marshal(tokenizerConfig)
		if err != nil {
			return nil, err
		}
	}
	// hftokenizer does not implement Digits and falls back to whitespace
	// splitting, which loses Janus prompt spaces/newlines. An isolated numeric
	// split is equivalent for the export's individual_digits=true contract.
	if family == "janus" {
		var pre struct {
			Type  string           `json:"type"`
			Steps []map[string]any `json:"pretokenizers"`
		}
		if err = json.Unmarshal(tokenizerConfig["pre_tokenizer"], &pre); err != nil {
			return nil, err
		}
		for i, step := range pre.Steps {
			if step["type"] == "Digits" {
				if step["individual_digits"] != true {
					return nil, errors.New("Janus requires individual digit tokenization")
				}
				pre.Steps[i] = map[string]any{"type": "Split", "pattern": map[string]string{"Regex": `\p{N}`}, "behavior": "Isolated", "invert": false}
			}
		}
		tokenizerConfig["pre_tokenizer"], err = json.Marshal(pre)
		if err != nil {
			return nil, err
		}
		data, err = json.Marshal(tokenizerConfig)
		if err != nil {
			return nil, err
		}
	}
	tokenizer, err := hftokenizer.NewFromContent(nil, data)
	if err != nil {
		return nil, err
	}
	if err = tokenizer.With(api.EncodeOptions{AddSpecialTokens: false}); err != nil {
		return nil, err
	}
	return tokenizer, nil
}

func (n *nativeMultimodal) validateMode(mode string) error {
	if n.family == "janus" && (mode == "text" || mode == "image") {
		return nil
	}
	if n.family == "lfm2_audio" && (mode == "text" || mode == "audio" || mode == "interleaved") {
		return nil
	}
	return fmt.Errorf("%s does not support generation mode %q", n.family, mode)
}

func (n *nativeMultimodal) graph(ctx context.Context, name string) (*backends.Model, error) {
	if n.p.Model.OnnxFilename == "onnx/"+name {
		return n.p.Model, nil
	}
	return n.p.Model.LoadGraph(ctx, "onnx/"+name)
}

func (n *nativeMultimodal) run(ctx context.Context, name string, inputs map[string]backends.Tensor) (map[string]backends.Tensor, error) {
	model, err := n.graph(ctx, name)
	if err != nil {
		return nil, err
	}
	return runNativeTensors(ctx, n.p.BasePipeline, model, inputs)
}

func (n *nativeMultimodal) encode(text string) []int64 {
	ids := n.tokenizer.Encode(text)
	result := make([]int64, len(ids))
	for i, id := range ids {
		result[i] = int64(id)
	}
	return result
}

func (n *nativeMultimodal) decode(ids []int64) (string, error) {
	u := make([]int, len(ids))
	for i, id := range ids {
		u[i] = int(id)
	}
	text := n.tokenizer.Decode(u)
	// Hold incomplete UTF-8 bytes until subsequent tokens finish the rune.
	for offset := 0; offset < len(text); {
		if !utf8.FullRuneInString(text[offset:]) {
			text = text[:offset]
			break
		}
		_, size := utf8.DecodeRuneInString(text[offset:])
		offset += size
	}
	return strings.ToValidUTF8(text, "�"), nil
}

func (p *AnyToAnyPipeline) runNativeMultimodal(ctx context.Context, inputs [][]backends.Message) (*AnyToAnyOutput, error) {
	if len(inputs) == 0 {
		return nil, errors.New("any-to-any requires conversations")
	}
	merged, cancel := context.WithCancel(ctx)
	stop := func() bool { return false }
	if p.SessionContext != nil {
		stop = context.AfterFunc(p.SessionContext, cancel)
	}
	cleanup := func() { stop(); cancel() }
	events := make(chan AnyToAnyEvent, 8)
	errs := make(chan error, 1)
	work := func() {
		defer cleanup()
		defer close(events)
		defer close(errs)
		for i, conversation := range inputs {
			emit := func(event AnyToAnyEvent) error {
				event.Sequence = i
				select {
				case events <- event:
					return nil
				case <-merged.Done():
					return merged.Err()
				}
			}
			var err error
			if p.native.family == "janus" {
				err = p.native.generateJanus(merged, conversation, emit)
			} else {
				err = p.native.generateLFM(merged, conversation, emit)
			}
			if err != nil {
				errs <- fmt.Errorf("conversation %d: %w", i, err)
				return
			}
			if err = emit(AnyToAnyEvent{Type: "done"}); err != nil {
				errs <- err
				return
			}
		}
	}
	go work()
	if p.Streaming {
		return &AnyToAnyOutput{EventStream: events, ErrorStream: errs}, nil
	}
	out := &AnyToAnyOutput{Results: make([]AnyToAnyResult, len(inputs)), Responses: make([]string, len(inputs))}
	for event := range events {
		result := &out.Results[event.Sequence]
		switch event.Type {
		case "text":
			result.Text += event.Text
		case "image":
			result.Images = append(result.Images, event.Image)
		case "audio":
			result.Audio = event.Audio
		}
	}
	for err := range errs {
		if err != nil {
			return nil, err
		}
	}
	for i, result := range out.Results {
		out.Responses[i] = result.Text
	}
	return out, nil
}

func nativeMessageBlocks(message backends.Message) ([]backends.ContentBlock, error) {
	if len(message.ContentBlocks) != 0 {
		if message.Content != "" || len(message.ImageURLs) != 0 || len(message.AudioURLs) != 0 {
			return nil, errors.New("content blocks cannot be combined with attachments")
		}
		return message.ContentBlocks, nil
	}
	blocks := []backends.ContentBlock{}
	for _, url := range message.AudioURLs {
		blocks = append(blocks, backends.ContentBlock{Type: "audio", URL: url})
	}
	blocks = append(blocks, backends.ContentBlock{Type: "text", Text: message.Content})
	for _, url := range message.ImageURLs {
		blocks = append(blocks, backends.ContentBlock{Type: "image", URL: url})
	}
	return blocks, nil
}

func nativeRole(role string) (string, error) {
	if role == "" {
		return "user", nil
	}
	if role != "user" && role != "assistant" && role != "system" {
		return "", fmt.Errorf("unsupported conversation role %q", role)
	}
	return role, nil
}

// initialCaches specializes only the cache axes; all other unknown dimensions
// are rejected instead of inventing sizes for an incompatible graph.
func initialCaches(model *backends.Model, batch int64) (map[string]backends.Tensor, error) {
	inputs := map[string]backends.Tensor{}
	for _, meta := range model.InputsMeta {
		if !strings.HasPrefix(meta.Name, "past_") {
			continue
		}
		shape := append([]int64(nil), meta.Dimensions...)
		for axis, dim := range shape {
			if dim < 0 {
				switch {
				case axis == 0:
					shape[axis] = batch
				case axis == 2 && strings.HasPrefix(meta.Name, "past_key_values"):
					shape[axis] = 0
				default:
					return nil, fmt.Errorf("unsupported cache shape %s %v", meta.Name, shape)
				}
			}
		}
		count := int64(1)
		for _, d := range shape {
			count *= d
		}
		inputs[meta.Name] = backends.Tensor{Shape: shape, Data: make([]float32, int(count))}
	}
	return inputs, nil
}

func updateCaches(cache, outputs map[string]backends.Tensor) {
	for name, t := range outputs {
		if strings.HasPrefix(name, "present.") {
			cache[strings.Replace(name, "present.", "past_key_values.", 1)] = t
		}
		if strings.HasPrefix(name, "present_conv.") {
			cache[strings.Replace(name, "present_conv.", "past_conv.", 1)] = t
		}
	}
}

func lastHidden(t backends.Tensor) (backends.Tensor, error) {
	data, ok := t.Data.([]float32)
	if !ok || len(t.Shape) != 3 || t.Shape[0] < 1 || t.Shape[1] < 1 || t.Shape[2] < 1 || int64(len(data)) != t.Shape[0]*t.Shape[1]*t.Shape[2] {
		return backends.Tensor{}, errors.New("invalid hidden-state tensor")
	}
	b, s, h := int(t.Shape[0]), int(t.Shape[1]), int(t.Shape[2])
	last := make([]float32, b*h)
	for i := 0; i < b; i++ {
		copy(last[i*h:], data[(i*s+s-1)*h:(i*s+s)*h])
	}
	return backends.Tensor{Shape: []int64{int64(b), 1, int64(h)}, Data: last}, nil
}

func greedy(values []float32) (int64, error) {
	if len(values) == 0 {
		return 0, errors.New("empty logits")
	}
	best := 0
	for i, v := range values {
		if !finite32(v) {
			return 0, errors.New("nonfinite logits")
		}
		if v > values[best] {
			best = i
		}
	}
	return int64(best), nil
}
