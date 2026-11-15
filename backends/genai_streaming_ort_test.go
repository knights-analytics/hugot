//go:build cgo && (ORT || ALL)

package backends

import (
	"context"
	"errors"
	"reflect"
	"testing"
	"time"

	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/ortgenai"
)

type fakeStreamingORTSession struct {
	chunks           [][]float32
	flushed          bool
	destroyed        chan struct{}
	processErr       error
	optionErr        error
	processorOptions map[string]string
	runtimeOptions   map[string]string
}

func (s *fakeStreamingORTSession) Process(_ context.Context, samples []float32) ([]ortgenai.ASRResult, error) {
	s.chunks = append(s.chunks, append([]float32(nil), samples...))
	return []ortgenai.ASRResult{{Text: "part", Words: []ortgenai.ASRTimestamp{{Text: "part", StartTime: 123, StopTime: 456}}}}, s.processErr
}

func (s *fakeStreamingORTSession) Flush(context.Context) ([]ortgenai.ASRResult, error) {
	s.flushed = true
	return []ortgenai.ASRResult{{Text: "end"}}, nil
}

func (s *fakeStreamingORTSession) SetOption(key, value string) error {
	s.processorOptions[key] = value
	return s.optionErr
}

func (s *fakeStreamingORTSession) SetRuntimeOption(key, value string) error {
	s.runtimeOptions[key] = value
	return s.optionErr
}

func (s *fakeStreamingORTSession) Destroy() { close(s.destroyed) }

func newTestStreamingORTProvider(t *testing.T, native *fakeStreamingORTSession) *streamingORTProvider {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	native.destroyed = make(chan struct{})
	native.processorOptions = map[string]string{}
	native.runtimeOptions = map[string]string{}
	s := &streamingORTProvider{ctx: ctx, cancel: cancel, create: func(string) (streamingORTSession, error) { return native, nil }}
	t.Cleanup(s.Close)
	return s
}

func collectTestASRStream(t *testing.T, stream <-chan ASRResult, errs <-chan error) ([]ASRResult, error) {
	t.Helper()
	var results []ASRResult
	var resultErr error
	timeout := time.After(3 * time.Second)
	for stream != nil || errs != nil {
		select {
		case result, ok := <-stream:
			if !ok {
				stream = nil
				continue
			}
			results = append(results, result)
		case err, ok := <-errs:
			if !ok {
				errs = nil
				continue
			}
			resultErr = errors.Join(resultErr, err)
		case <-timeout:
			t.Fatal("ASR stream did not close")
		}
	}
	return results, resultErr
}

func TestStreamingORTASRSplitsChunksAndFlushesTimestamps(t *testing.T) {
	native := &fakeStreamingORTSession{}
	provider := newTestStreamingORTProvider(t, native)
	chunks := make(chan []float32, 1)
	chunks <- []float32{1, 2, 3, 4, 5}
	close(chunks)
	stream, errs, err := provider.generate(context.Background(), nil, chunks, StreamingASROptions{
		ChunkSamples:     2,
		ProcessorOptions: map[string]string{"use_vad": "true"},
		RuntimeOptions:   map[string]string{"lang_id": "en"},
	})
	if err != nil {
		t.Fatal(err)
	}
	results, err := collectTestASRStream(t, stream, errs)
	if err != nil {
		t.Fatal(err)
	}
	provider.Close()
	if !reflect.DeepEqual(native.chunks, [][]float32{{1, 2}, {3, 4}, {5}}) || !native.flushed {
		t.Fatalf("wrong chunk boundaries or missing flush: %#v", native)
	}
	if len(results) != 4 || results[3].Text != "end" || results[0].Words[0].StartTime != 123 || results[0].Words[0].StopTime != 456 {
		t.Fatalf("lost transcript fragments or millisecond timestamps: %#v", results)
	}
	if native.processorOptions["use_vad"] != "true" || native.runtimeOptions["lang_id"] != "en" {
		t.Fatalf("missing processor/runtime options: %#v", native)
	}
}

func TestStreamingORTASRPreservesPartialResultsOnFailure(t *testing.T) {
	wantErr := errors.New("native chunk failed")
	native := &fakeStreamingORTSession{processErr: wantErr}
	provider := newTestStreamingORTProvider(t, native)
	chunks := make(chan []float32, 1)
	chunks <- []float32{1, 2, 3}
	close(chunks)
	stream, errs, err := provider.generate(context.Background(), nil, chunks, StreamingASROptions{ChunkSamples: 2})
	if err != nil {
		t.Fatal(err)
	}
	results, err := collectTestASRStream(t, stream, errs)
	provider.Close()
	if !errors.Is(err, wantErr) || len(results) != 1 || native.flushed || len(native.chunks) != 1 {
		t.Fatalf("failure lost partial results or replayed/flushed failed audio: results=%v err=%v native=%#v", results, err, native)
	}
}

func TestStreamingORTASRCloseCancelsBlockedStreamAndJoinsCleanup(t *testing.T) {
	native := &fakeStreamingORTSession{}
	provider := newTestStreamingORTProvider(t, native)
	stream, errs, err := provider.generate(context.Background(), nil, make(chan []float32), StreamingASROptions{ChunkSamples: 2})
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan struct{})
	go func() { provider.Close(); close(done) }()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("Close did not cancel blocked stream")
	}
	select {
	case <-native.destroyed:
	default:
		t.Fatal("Close returned before native cleanup")
	}
	_, err = collectTestASRStream(t, stream, errs)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v, want cancellation", err)
	}
	if _, _, err := provider.generate(context.Background(), nil, make(chan []float32), StreamingASROptions{ChunkSamples: 2}); err == nil {
		t.Fatal("closed provider accepted another stream")
	}
}

func TestStreamingORTASRConfigurationFailureReleasesNativeState(t *testing.T) {
	wantErr := errors.New("unsupported option")
	native := &fakeStreamingORTSession{optionErr: wantErr}
	provider := newTestStreamingORTProvider(t, native)
	stream, errs, err := provider.generate(context.Background(), nil, make(chan []float32), StreamingASROptions{ChunkSamples: 2, ProcessorOptions: map[string]string{"bad": "value"}})
	provider.Close()
	if !errors.Is(err, wantErr) || stream != nil || errs != nil {
		t.Fatalf("expected synchronous configuration error, got %v", err)
	}
	select {
	case <-native.destroyed:
	default:
		t.Fatal("configuration failure leaked native state")
	}
}

func TestStreamingORTASROptionsRejectIncompatibleModes(t *testing.T) {
	for _, opts := range []*options.OrtOptions{
		{UseEngine: true},
		{CudaOptions: map[string]string{}},
		{UseMTP: new(true)},
		{GenAIAdapters: []options.GenAIAdapterConfig{{Path: "adapter", Name: "a"}}},
	} {
		if err := validateStreamingORTOptions(opts); err == nil {
			t.Fatalf("accepted incompatible options: %#v", opts)
		}
	}
	if err := validateStreamingORTOptions(&options.OrtOptions{UseStreamingASR: true, GenAIGPUDeviceID: new(0), GenAILogStream: new("stderr")}); err != nil {
		t.Fatal(err)
	}
}
