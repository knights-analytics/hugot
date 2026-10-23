package pipelines

import (
	"context"
	"math"
	"testing"

	"github.com/knights-analytics/hugot/backends"
	"github.com/stretchr/testify/require"
)

func TestNativeTokenizerPreservesPromptWhitespace(t *testing.T) {
	// A tiny byte-level vocabulary exercises the vendor's Digits pre-tokenizer
	// and single-step Sequence decoder without requiring a downloaded model.
	data := []byte(`{"version":"1.0","normalizer":null,"pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Digits","individual_digits":true},{"type":"ByteLevel","add_prefix_space":false}]},"post_processor":null,"decoder":{"type":"Sequence","decoders":[{"type":"ByteLevel"}]},"model":{"type":"BPE","vocab":{"A":0,"Ġ":1,"1":2,"2":3,"Ċ":4},"merges":[]}}`)
	tokenizer, err := newNativeTokenizer(data, "janus")
	require.NoError(t, err)
	ids := tokenizer.Encode("A 12\n")
	require.Equal(t, []int{0, 1, 2, 3, 4}, ids)
	require.Equal(t, "A 12\n", tokenizer.Decode(ids))
}

func TestJanusSystemContentValidation(t *testing.T) {
	n := &nativeMultimodal{p: &AnyToAnyPipeline{multimodalGeneration: &multimodalGeneration{}, GenerationMode: "text"}}
	user := backends.Message{Content: "Say hello."}
	prompt, err := n.janusPrompt([]backends.Message{{Role: "system", ContentBlocks: []backends.ContentBlock{{Type: "text", Text: "Be concise."}}}, user})
	require.NoError(t, err)
	require.Contains(t, prompt, "Be concise.")
	for _, system := range []backends.Message{
		{Role: "system", ImageURLs: []string{"ignored.png"}},
		{Role: "system", Content: "ambiguous", ContentBlocks: []backends.ContentBlock{{Type: "text", Text: "other"}}},
		{Role: "system", ContentBlocks: []backends.ContentBlock{{Type: "audio", URL: "ignored.wav"}}},
		{Role: "system"},
	} {
		_, err = n.janusPrompt([]backends.Message{system, user})
		require.Error(t, err)
	}
}

func TestLFMInverseSTFTSamePadding(t *testing.T) {
	// A constant DC spectrum reconstructs 1/N per windowed frame. Check the
	// overlap-add normalization and model's exact same-padding output length.
	values := make([]float32, 4*1282)
	for frame := 0; frame < 4; frame++ {
		for bin := 0; bin < 641; bin++ {
			values[frame*1282+bin] = -100
		}
	}
	for frame := 0; frame < 4; frame++ {
		values[frame*1282] = 0
	}
	wave, err := decodeLFMSTFT(t.Context(), backends.Tensor{Shape: []int64{1, 4, 1282}, Data: values})
	require.NoError(t, err)
	require.Equal(t, 24000, wave.SampleRate)
	require.Len(t, wave.Samples, 4*320)
	for i, sample := range wave.Samples {
		index := i + 480
		var sum, envelope float64
		for frame := 0; frame < 4; frame++ {
			k := index - frame*320
			if k >= 0 && k < 1280 {
				window := .5 - .5*math.Cos(2*math.Pi*float64(k)/1280)
				sum += window / 1280
				envelope += window * window
			}
		}
		require.InDelta(t, sum/envelope, sample, 1e-8)
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	_, err = decodeLFMSTFT(ctx, backends.Tensor{Shape: []int64{1, 4, 1282}, Data: values})
	require.ErrorIs(t, err, context.Canceled)
}
