package pipelines

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"strings"
	"sync"

	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/util/fileutil"
)

// lfmEmbeddingTable holds the LFM text embedding matrix loaded from
// onnx/embed_tokens.bin. Upstream bundles ship this layer as raw weight
// files (embed_tokens.bin + embed_tokens.json) rather than as a runnable
// ONNX graph, so this struct reproduces the graph's single Gather axis=0
// operation directly in Go.
type lfmEmbeddingTable struct {
	mu     sync.Mutex
	path   string
	vocab  int
	hidden int
	values []float32
}

func (t *lfmEmbeddingTable) load(ctx context.Context, path string) error {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.values != nil && t.path == path {
		return nil
	}
	var meta struct {
		VocabSize  int    `json:"vocab_size"`
		HiddenSize int    `json:"hidden_size"`
		Dtype      string `json:"dtype"`
		ByteOrder  string `json:"byte_order"`
	}
	if err := readNativeConfig(ctx, path, "onnx/embed_tokens.json", &meta); err != nil {
		return fmt.Errorf("read embed_tokens.json: %w", err)
	}
	if meta.Dtype != "float32" {
		return fmt.Errorf("unsupported embed_tokens dtype %q", meta.Dtype)
	}
	var order binary.ByteOrder = binary.LittleEndian
	if meta.ByteOrder == "big" {
		order = binary.BigEndian
	}
	data, err := fileutil.ReadFileBytes(ctx, fileutil.PathJoinSafe(path, "onnx/embed_tokens.bin"))
	if err != nil {
		return fmt.Errorf("read embed_tokens.bin: %w", err)
	}
	elems := int64(meta.VocabSize) * int64(meta.HiddenSize)
	if int64(len(data)) != elems*4 {
		return fmt.Errorf("embed_tokens.bin size %d bytes does not match vocab=%d hidden=%d as float32 (%d bytes)",
			len(data), meta.VocabSize, meta.HiddenSize, elems*4)
	}
	values := make([]float32, elems)
	for i := int64(0); i < elems; i++ {
		values[i] = math.Float32frombits(order.Uint32(data[i*4 : i*4+4]))
	}
	t.path = path
	t.vocab = meta.VocabSize
	t.hidden = meta.HiddenSize
	t.values = values
	return nil
}

func (t *lfmEmbeddingTable) lookup(ids []int64) []float32 {
	t.mu.Lock()
	defer t.mu.Unlock()
	out := make([]float32, 0, len(ids)*t.hidden)
	for _, id := range ids {
		if id < 0 || int(id) >= t.vocab {
			return nil
		}
		start := id * int64(t.hidden)
		out = append(out, t.values[start:start+int64(t.hidden)]...)
	}
	return out
}

func (n *nativeMultimodal) lfmEmbeddings(ctx context.Context, ids []int64) (backends.Tensor, error) {
	for _, id := range ids {
		if id < 0 || id >= 65536 {
			return backends.Tensor{}, errors.New("LFM token ID outside vocabulary")
		}
	}
	loadCtx := ctx
	cancel := func() {}
	if sessionCtx := n.p.SessionContext; sessionCtx != nil {
		loadCtx, cancel = context.WithCancel(sessionCtx)
		stop := context.AfterFunc(ctx, cancel)
		defer func() {
			stop()
			cancel()
		}()
	}
	if err := n.lfmEmbeddingTable.load(loadCtx, n.p.Model.Path); err != nil {
		return backends.Tensor{}, err
	}
	if err := loadCtx.Err(); err != nil {
		return backends.Tensor{}, err
	}
	out := n.lfmEmbeddingTable.lookup(ids)
	if out == nil {
		return backends.Tensor{}, errors.New("LFM token ID outside vocabulary")
	}
	return backends.Tensor{Shape: []int64{1, int64(len(ids)), int64(n.lfmEmbeddingTable.hidden)}, Data: out}, nil
}

func (n *nativeMultimodal) lfmPrompt(messages []backends.Message) (string, error) {
	if len(messages) == 0 {
		return "", errors.New("empty conversation")
	}
	system := n.p.SystemPrompt
	if system == "" {
		switch n.p.GenerationMode {
		case "audio":
			system = "Perform TTS. Use the UK female voice."
		case "interleaved":
			system = "Respond with interleaved text and audio."
		}
	}
	var prompt strings.Builder
	prompt.WriteString("<|startoftext|>")
	if system != "" {
		prompt.WriteString("<|im_start|>system\n" + system + "<|im_end|>\n")
	}
	for _, message := range messages {
		role, err := nativeRole(message.Role)
		if err != nil {
			return "", err
		}
		blocks, err := nativeMessageBlocks(message)
		if err != nil {
			return "", err
		}
		var text strings.Builder
		for _, block := range blocks {
			if block.Type != "text" || block.URL != "" {
				return "", errors.New("LFM media generation currently accepts text conversations; use Phi-4 for audio/image/video input")
			}
			text.WriteString(block.Text)
		}
		if strings.TrimSpace(text.String()) == "" {
			return "", errors.New("empty LFM message")
		}
		prompt.WriteString("<|im_start|>" + role + "\n" + text.String() + "<|im_end|>\n")
	}
	prompt.WriteString("<|im_start|>assistant\n")
	return prompt.String(), nil
}

func (n *nativeMultimodal) generateLFM(ctx context.Context, messages []backends.Message, emit func(AnyToAnyEvent) error) error {
	prompt, err := n.lfmPrompt(messages)
	if err != nil {
		return err
	}
	ids := n.encode(prompt)
	if len(ids) == 0 || len(ids) >= n.p.MaxLength {
		return errors.New("LFM prompt exceeds maximum length")
	}
	embeds, err := n.lfmEmbeddings(ctx, ids)
	if err != nil {
		return err
	}
	model, err := n.graph(ctx, "decoder_q4.onnx")
	if err != nil {
		return err
	}
	cache, err := initialCaches(model, 1)
	if err != nil {
		return err
	}
	length := len(ids)
	audioMode, textDone := false, false
	textCount, audioCount := 0, 0
	textTokens := []int64{}
	codes := make([][]int64, 0)
	for step := 0; step < n.p.MaxNewTokens; step++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		if length > n.p.MaxLength {
			return errors.New("LFM generation exceeds maximum length")
		}
		cache["inputs_embeds"] = embeds
		mask := make([]int64, length)
		for i := range mask {
			mask[i] = 1
		}
		cache["attention_mask"] = backends.Tensor{Shape: []int64{1, int64(length)}, Data: mask}
		outputs, err := runNativeTensors(ctx, n.p.BasePipeline, model, cache)
		if err != nil {
			return err
		}
		updateCaches(cache, outputs)
		if audioMode {
			hidden, err := lastHidden(outputs["hidden_states"])
			if err != nil {
				return err
			}
			frame, err := n.lfmAudioFrame(ctx, hidden)
			if err != nil {
				return err
			}
			endAudio := frame[0] == 2048
			if endAudio {
				for i := range frame {
					frame[i] = 2048
				}
				audioMode = false
			} else {
				codes = append(codes, frame)
			}
			feedback := make([]int64, 8)
			for i, code := range frame {
				feedback[i] = int64(i*2049) + code
			}
			audio, err := n.run(ctx, "audio_embedding_q4.onnx", map[string]backends.Tensor{"audio_codes": {Shape: []int64{1, 8}, Data: feedback}})
			if err != nil {
				return err
			}
			values, ok := audio["audio_embeds"].Data.([]float32)
			if !ok || len(values) != 8*2048 {
				return errors.New("invalid LFM audio embeddings")
			}
			sum := make([]float32, 2048)
			for c := 0; c < 8; c++ {
				for h := 0; h < 2048; h++ {
					sum[h] += values[c*2048+h]
				}
			}
			embeds = backends.Tensor{Shape: []int64{1, 1, 2048}, Data: sum}
			audioCount++
			if n.p.GenerationMode == "interleaved" && !textDone && audioCount == 12 {
				audioMode = false
				audioCount = 0
				textCount = 0
			}
		} else {
			token, err := lastTokenArgmax(outputs["logits"])
			if err != nil {
				return err
			}
			if token == 7 || token == 2 {
				break
			}
			textCount++
			if n.p.GenerationMode == "text" && (token == 128 || token == 130) {
				break
			}
			if n.p.GenerationMode == "audio" && token == 128 {
				audioMode = true
			}
			if token == 130 {
				textDone = true
			}
			if token != 128 && token != 130 {
				textTokens = append(textTokens, token)
				text, err := n.decode(textTokens)
				if err != nil {
					return err
				}
				previous, err := n.decode(textTokens[:len(textTokens)-1])
				if err != nil {
					return err
				}
				if !strings.HasPrefix(text, previous) {
					return errors.New("LFM tokenizer changed an emitted text prefix")
				}
				if err = emit(AnyToAnyEvent{Type: "text", Text: strings.TrimPrefix(text, previous)}); err != nil {
					return err
				}
			}
			if n.p.GenerationMode == "interleaved" && (textCount == 6 || textDone) {
				audioMode = true
				audioCount = 0
			}
			embeds, err = n.lfmEmbeddings(ctx, []int64{token})
			if err != nil {
				return err
			}
		}
		length++
	}
	if len(codes) != 0 {
		flat := make([]int64, 8*len(codes))
		for i, frame := range codes {
			for c, code := range frame {
				if code > 2047 {
					code = 2047
				}
				flat[c*len(codes)+i] = code
			}
		}
		out, err := n.run(ctx, "audio_detokenizer_q4.onnx", map[string]backends.Tensor{"audio_codes": {Shape: []int64{1, 8, int64(len(codes))}, Data: flat}})
		if err != nil {
			return err
		}
		wave, err := decodeLFMSTFT(ctx, out["stft_features"])
		if err != nil {
			return err
		}
		return emit(AnyToAnyEvent{Type: "audio", Audio: &wave})
	}
	if n.p.GenerationMode == "audio" || n.p.GenerationMode == "interleaved" {
		return errors.New("generation ended before producing audio; increase maximum new tokens or use a speech prompt")
	}
	return nil
}

func (n *nativeMultimodal) lfmAudioFrame(ctx context.Context, hidden backends.Tensor) ([]int64, error) {
	hidden.Shape = []int64{1, 2048}
	cache := map[string]backends.Tensor{
		"hidden_states":   hidden,
		"depth_slices_in": {Shape: []int64{1, 8, 1024}, Data: make([]float32, 8*1024)},
		"past_keys":       {Shape: []int64{6, 1, 8, 0, 32}, Data: []float32{}},
		"past_values":     {Shape: []int64{6, 1, 8, 0, 32}, Data: []float32{}},
	}
	frame := make([]int64, 8)
	previous := int64(0)
	for step := 0; step < 8; step++ {
		cache["step_idx"] = backends.Tensor{Shape: []int64{}, Data: []int64{int64(step)}}
		cache["prev_token"] = backends.Tensor{Shape: []int64{1}, Data: []int64{previous}}
		cache["seqlens_k"] = backends.Tensor{Shape: []int64{1}, Data: []int32{int32(step)}}
		cache["total_seq_len"] = backends.Tensor{Shape: []int64{}, Data: []int32{int32(step + 1)}}
		out, err := n.run(ctx, "vocoder_depthformer_q4.onnx", cache)
		if err != nil {
			return nil, err
		}
		values, ok := out["logits"].Data.([]float32)
		if !ok || len(values) != 2049 {
			return nil, errors.New("invalid LFM depthformer logits")
		}
		previous, err = greedy(values)
		if err != nil {
			return nil, err
		}
		frame[step] = previous
		cache["depth_slices_in"], cache["past_keys"], cache["past_values"] = out["depth_slices"], out["new_keys"], out["new_values"]
	}
	return frame, nil
}

// decodeLFMSTFT reconstructs the model's 24 kHz waveform with its periodic Hann
// window and same padding. The 1280-point FFT is factored into five 256 FFTs.
func decodeLFMSTFT(ctx context.Context, t backends.Tensor) (AudioWaveform, error) {
	values, ok := t.Data.([]float32)
	if !ok || len(t.Shape) != 3 || t.Shape[0] != 1 || t.Shape[1] <= 0 || t.Shape[2] != 1282 || int64(len(values)) != t.Shape[1]*1282 {
		return AudioWaveform{}, fmt.Errorf("LFM STFT output must be FP32 [1,frames,1282], got %v", t.Shape)
	}
	frames := int(t.Shape[1])
	const fftSize = 1280
	const hop = 320
	window := make([]float64, fftSize)
	for i := range window {
		window[i] = .5 - .5*math.Cos(2*math.Pi*float64(i)/fftSize)
	}
	wave, envelope := make([]float64, (frames-1)*hop+fftSize), make([]float64, (frames-1)*hop+fftSize)
	for frame := 0; frame < frames; frame++ {
		if err := ctx.Err(); err != nil {
			return AudioWaveform{}, err
		}
		frequency := make([]complex128, fftSize)
		for bin := 0; bin <= fftSize/2; bin++ {
			magnitude, phase := values[frame*1282+bin], values[frame*1282+641+bin]
			if !finite32(magnitude) || !finite32(phase) {
				return AudioWaveform{}, errors.New("nonfinite audio spectrum")
			}
			m := math.Exp(float64(magnitude))
			frequency[bin] = complex(m*math.Cos(float64(phase)), -m*math.Sin(float64(phase)))
			if bin > 0 && bin < fftSize/2 {
				frequency[fftSize-bin] = complex(real(frequency[bin]), -imag(frequency[bin]))
			}
		}
		parts := make([][]complex128, 5)
		for r := 0; r < 5; r++ {
			parts[r] = make([]complex128, 256)
			for m := 0; m < 256; m++ {
				parts[r][m] = frequency[5*m+r]
			}
			fftInPlace(parts[r])
		}
		for k := 0; k < fftSize; k++ {
			var sum complex128
			for r := 0; r < 5; r++ {
				angle := -2 * math.Pi * float64(r*k) / fftSize
				sum += parts[r][k%256] * complex(math.Cos(angle), math.Sin(angle))
			}
			index := frame*hop + k
			wave[index] += real(sum) / fftSize * window[k]
			envelope[index] += window[k] * window[k]
		}
	}
	const pad = (fftSize - hop) / 2
	samples := make([]float32, len(wave)-2*pad)
	for i := range samples {
		v := wave[i+pad] / envelope[i+pad]
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return AudioWaveform{}, errors.New("nonfinite reconstructed audio")
		}
		samples[i] = float32(v)
	}
	return AudioWaveform{Samples: samples, SampleRate: 24000}, nil
}
