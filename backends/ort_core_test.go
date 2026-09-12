//go:build cgo && (ORT || ALL)

package backends

import (
	"context"
	"testing"

	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/ortgenai"
	ort "github.com/microsoft/onnxruntime/go/onnxruntime"
	"github.com/stretchr/testify/require"
)

func TestConvertCoreORTIO(t *testing.T) {
	converted := convertCoreORTIO([]ort.IOInfo{{
		Name:     "input_ids",
		DataType: ort.TensorElementDataTypeInt64,
		Shape:    []int64{-1, 128},
	}})

	require.Equal(t, []InputOutputInfo{{Name: "input_ids", Dimensions: Shape{-1, 128}}}, converted)
}

func TestCoreORTSessionRejectsUninitializedSession(t *testing.T) {
	_, err := (&coreORTSession{}).Run(context.Background(), nil)
	require.EqualError(t, err, "ORT session is not initialized")
}

func TestCoreORTTensorCloseIsNilSafe(t *testing.T) {
	require.NoError(t, (*coreORTTensor)(nil).Close())
}

func TestCreateORTMessagesAddsImageTags(t *testing.T) {
	got := createORTMessages([][]Message{
		{
			{Role: "user", Content: "Describe these images", ImageURLs: []string{"first.jpg", "second.jpg"}},
			{Role: "assistant", Content: "I will inspect them."},
			{Role: "user", Content: "And this one", ImageURLs: []string{"third.jpg"}},
		},
		{{Role: "user", Content: "Describe this image", ImageURLs: []string{"fourth.jpg"}}},
	}, "System prompt")

	require.Equal(t, [][]ortgenai.Message{
		{
			{Role: "system", Content: "System prompt"},
			{Role: "user", Content: "Describe these images\n<|image_1|>\n<|image_2|>"},
			{Role: "assistant", Content: "I will inspect them."},
			{Role: "user", Content: "And this one\n<|image_3|>"},
		},
		{
			{Role: "system", Content: "System prompt"},
			{Role: "user", Content: "Describe this image\n<|image_1|>"},
		},
	}, got)
}

func TestCreateMessagesORTPreservesPerConversationMedia(t *testing.T) {
	inputs := [][]Message{
		{{Role: "user", Content: "first", ImageURLs: []string{"first.jpg"}, AudioURLs: []string{"first.wav"}}},
		{{Role: "user", Content: "second", ImageURLs: []string{"second.jpg"}}},
	}
	batch := NewBatch(len(inputs))

	require.NoError(t, CreateMessagesORT(batch, inputs, ""))
	require.Equal(t, inputs, batch.MultimodalMessages)
	inputs[0][0].ImageURLs[0] = "changed.jpg"
	inputs[0][0].AudioURLs[0] = "changed.wav"
	require.Equal(t, "first.jpg", batch.MultimodalMessages[0][0].ImageURLs[0])
	require.Equal(t, "first.wav", batch.MultimodalMessages[0][0].AudioURLs[0])

	messages, ok := batch.InputValues.([][]ortgenai.Message)
	require.True(t, ok)
	require.Equal(t, "first\n<|image_1|>", messages[0][0].Content)
	require.Equal(t, "second\n<|image_1|>", messages[1][0].Content)
	// Contract: CreateMessagesORT is pure — it must not touch batch.Images;
	// native tensor loading happens later, in the dispatch path.
	require.Nil(t, batch.Images)
}

func TestConversationHasImages(t *testing.T) {
	require.False(t, conversationHasImages(nil))
	require.False(t, conversationHasImages([]Message{{Role: "user", Content: "no images"}}))
	require.True(t, conversationHasImages([]Message{{Role: "user", Content: "hi", ImageURLs: []string{"a.jpg"}}}))
	require.True(t, conversationHasImages([]Message{
		{Role: "user", Content: "a"},
		{Role: "assistant", Content: "b"},
		{Role: "user", Content: "c", ImageURLs: []string{"c.jpg"}},
	}))
}

func TestFlattenImageURLs(t *testing.T) {
	require.Empty(t, flattenImageURLs(nil))
	require.Empty(t, flattenImageURLs([]Message{{Role: "user", Content: "no images"}}))
	require.Equal(t, []string{"a.jpg"}, flattenImageURLs([]Message{{Role: "user", Content: "hi", ImageURLs: []string{"a.jpg"}}}))
	require.Equal(t, []string{"a.jpg", "b.jpg", "c.jpg"}, flattenImageURLs([]Message{
		{Role: "user", Content: "a", ImageURLs: []string{"a.jpg"}},
		{Role: "assistant", Content: "b"},
		{Role: "user", Content: "c", ImageURLs: []string{"b.jpg", "c.jpg"}},
	}))
}

func TestConversationHasAudio(t *testing.T) {
	require.False(t, conversationHasAudio(nil))
	require.False(t, conversationHasAudio([]Message{{Role: "user", Content: "no audio"}}))
	require.True(t, conversationHasAudio([]Message{{Role: "user", Content: "hi", AudioURLs: []string{"a.wav"}}}))
	require.True(t, conversationHasAudio([]Message{
		{Role: "user", Content: "a"},
		{Role: "assistant", Content: "b"},
		{Role: "user", Content: "c", AudioURLs: []string{"c.wav"}},
	}))
}

func TestFlattenAudioURLs(t *testing.T) {
	require.Empty(t, flattenAudioURLs(nil))
	require.Empty(t, flattenAudioURLs([]Message{{Role: "user", Content: "no audio"}}))
	require.Equal(t, []string{"a.wav"}, flattenAudioURLs([]Message{{Role: "user", Content: "hi", AudioURLs: []string{"a.wav"}}}))
	require.Equal(t, []string{"a.wav", "b.wav", "c.wav"}, flattenAudioURLs([]Message{
		{Role: "user", Content: "a", AudioURLs: []string{"a.wav"}},
		{Role: "assistant", Content: "b"},
		{Role: "user", Content: "c", AudioURLs: []string{"b.wav", "c.wav"}},
	}))
}

func TestToORTMessagesStripsMedia(t *testing.T) {
	conv := []Message{
		{Role: "user", Content: "hello", ImageURLs: []string{"a.jpg"}, AudioURLs: []string{"a.wav"}},
		{Role: "assistant", Content: "hi back"},
	}
	got := toORTMessages(conv)
	require.Len(t, got, 2)
	require.Equal(t, "user", got[0].Role)
	require.Equal(t, "hello", got[0].Content)
	require.Equal(t, "assistant", got[1].Role)
	require.Equal(t, "hi back", got[1].Content)
	// Media references are not embedded in ortgenai.Message; they travel
	// on the sibling Images/Audios fields. This is a type-level guarantee
	// here (the struct has no Image/Audio fields), so a compile error in
	// toORTMessages itself would surface the drift.
}

func TestValidateGenerativeEngineOptionsRejectsRuntimeControls(t *testing.T) {
	cases := []struct {
		name string
		opts *options.OrtOptions
	}{
		{"gpu_device_id", func() *options.OrtOptions {
			v := 1
			return &options.OrtOptions{GenAIGPUDeviceID: &v}
		}()},
		{"log_file", func() *options.OrtOptions {
			v := "/tmp/orlog.log"
			return &options.OrtOptions{GenAILogFile: &v}
		}()},
		{"log_stream", func() *options.OrtOptions {
			v := "stderr"
			return &options.OrtOptions{GenAILogStream: &v}
		}()},
		{"adapters", &options.OrtOptions{
			GenAIAdapters: []options.GenAIAdapterConfig{{Path: "/tmp/adp", Name: "a"}},
		}},
		{"active_adapter", func() *options.OrtOptions {
			v := "a"
			return &options.OrtOptions{GenAIActiveAdapter: &v}
		}()},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := validateGenerativeEngineOptions(tc.opts)
			require.Error(t, err)
			require.Contains(t, err.Error(), "GenAI runtime controls")
		})
	}
}

func TestValidateGenerativeEngineOptionsAcceptsDefaults(t *testing.T) {
	// Empty OrtOptions (no adapters, no GPU/log, no EPs) is a valid engine-mode
	// configuration.
	require.NoError(t, validateGenerativeEngineOptions(&options.OrtOptions{}))
}

func TestValidateGenerativeEngineOptionsRejectsMTP(t *testing.T) {
	// MTP is a session-path runtime control (dispatched at call time via the
	// adapter, not a session option), so the engine-path generator must reject
	// it the same way it rejects the other WithGenAI* runtime controls.
	on := true
	off := false
	cases := []struct {
		name string
		opts *options.OrtOptions
	}{
		{"mtp_enabled", &options.OrtOptions{UseMTP: &on}},
		{"mtp_disabled", &options.OrtOptions{UseMTP: &off}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := validateGenerativeEngineOptions(tc.opts)
			require.Error(t, err)
			require.Contains(t, err.Error(), "GenAI runtime controls")
		})
	}

	// Sanity: only UseMTP set alongside valid defaults must still be rejected
	// (no EP flags, no session-only knobs) — isolates the UseMTP clause above
	// rather than a catch-all.
	require.Contains(t, validateGenerativeEngineOptions(&options.OrtOptions{UseMTP: &on}).Error(), "GenAI runtime controls")
}
