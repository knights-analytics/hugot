//go:build cgo && (ORT || ALL)

package backends

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"runtime"
	"sync"

	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/util/fileutil"
	"github.com/knights-analytics/ortgenai"
)

type streamingORTSession interface {
	Process(context.Context, []float32) ([]ortgenai.ASRResult, error)
	Flush(context.Context) ([]ortgenai.ASRResult, error)
	SetOption(string, string) error
	SetRuntimeOption(string, string) error
	Destroy()
}

// streamingORTProvider tracks streams so closing a model cancels and joins all
// native work before the process-wide GenAI environment can be released.
type streamingORTProvider struct {
	modelPath string
	ctx       context.Context
	cancel    context.CancelFunc
	mu        sync.Mutex
	closed    bool
	workers   sync.WaitGroup
	create    func(string) (streamingORTSession, error)
}

func createORTStreamingASRProvider(ctx context.Context, model *Model, opts *options.Options) error {
	if err := validateStreamingORTOptions(opts.ORTOptions); err != nil {
		return err
	}
	local, err := fileutil.IsLocalPath(ctx, model.Path)
	if err != nil {
		return err
	}
	if !local {
		return errors.New("streaming ASR requires a local model directory")
	}
	if err := initialiseORTGenAI(ctx, opts); err != nil {
		return err
	}
	streamCtx, cancel := context.WithCancel(ctx)
	provider := &streamingORTProvider{
		modelPath: model.Path, ctx: streamCtx, cancel: cancel,
		create: func(path string) (streamingORTSession, error) { return ortgenai.CreateStreamingASR(path) },
	}
	model.ORTModel = &ORTModel{Generative: &generativeORTProvider{streaming: provider}, Options: opts.ORTOptions}
	return nil
}

func validateStreamingORTOptions(opts *options.OrtOptions) error {
	if opts == nil {
		return errors.New("ORT options are required for streaming ASR")
	}
	if opts.UseEngine || opts.EngineMemory != (options.GenAIEngineMemoryOptions{}) || opts.EngineImageMarker != "" {
		return errors.New("WithGenAIStreamingASR cannot be combined with generative engine options")
	}
	// This constructor uses the package's providers and search defaults. Global
	// device/logging settings still apply; session and adapter settings do not.
	copyOptions := *opts
	copyOptions.UseStreamingASR = false
	copyOptions.GenAIGPUDeviceID = nil
	copyOptions.GenAILogFile = nil
	copyOptions.GenAILogStream = nil
	if err := validateGenerativeEngineOptions(&copyOptions); err != nil {
		return fmt.Errorf("streaming ASR uses package settings and does not support session, adapter, MTP, or custom provider options: %w", err)
	}
	return nil
}

func (s *streamingORTProvider) Close() {
	s.mu.Lock()
	if !s.closed {
		s.closed = true
		s.cancel()
	}
	s.mu.Unlock()
	s.workers.Wait()
}

func runStreamingORTASR(ctx context.Context, p *BasePipeline, chunks <-chan []float32, opts StreamingASROptions) (<-chan ASRResult, <-chan error, error) {
	if p == nil || p.Model == nil || p.Model.ORTModel == nil || p.Model.ORTModel.Generative == nil || p.Model.ORTModel.Generative.streaming == nil {
		return nil, nil, errors.New("streaming ASR requires WithGenAIStreamingASR")
	}
	return p.Model.ORTModel.Generative.streaming.generate(ctx, p.SessionContext, chunks, opts)
}

func (s *streamingORTProvider) generate(ctx, sessionCtx context.Context, chunks <-chan []float32, opts StreamingASROptions) (<-chan ASRResult, <-chan error, error) {
	if chunks == nil || opts.ChunkSamples <= 0 {
		return nil, nil, errors.New("streaming ASR requires an audio channel and a positive ChunkSamples setting")
	}
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	opts.ProcessorOptions = maps.Clone(opts.ProcessorOptions)
	opts.RuntimeOptions = maps.Clone(opts.RuntimeOptions)
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil, nil, errors.New("streaming ASR provider is closed")
	}
	s.workers.Add(1)
	s.mu.Unlock()
	streamCtx, cancel := context.WithCancel(ctx)
	stopProvider := context.AfterFunc(s.ctx, cancel) //nolint:contextcheck // Both the provider lifetime and the caller's context must cancel the stream.
	stopSession := func() bool { return true }
	if sessionCtx != nil {
		stopSession = context.AfterFunc(sessionCtx, cancel)
	}
	results := make(chan ASRResult, 8)
	errs := make(chan error, 1)
	ready := make(chan error, 1)
	go func() {
		defer s.workers.Done()
		defer close(results)
		defer close(errs)
		defer cancel()
		defer stopProvider()
		defer stopSession()
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		if err := streamCtx.Err(); err != nil {
			ready <- err
			return
		}
		native, err := s.create(s.modelPath)
		if err != nil {
			ready <- err
			return
		}
		defer native.Destroy()
		for key, value := range opts.ProcessorOptions {
			if err := native.SetOption(key, value); err != nil {
				ready <- fmt.Errorf("setting streaming ASR processor option %q: %w", key, err)
				return
			}
		}
		for key, value := range opts.RuntimeOptions {
			if err := native.SetRuntimeOption(key, value); err != nil {
				ready <- fmt.Errorf("setting streaming ASR runtime option %q: %w", key, err)
				return
			}
		}
		ready <- nil
		if err := transcribeORTChunks(streamCtx, native, chunks, opts.ChunkSamples, results); err != nil {
			errs <- err
		}
	}()
	if err := <-ready; err != nil {
		return nil, nil, fmt.Errorf("creating streaming ASR: %w", err)
	}
	return results, errs, nil
}

func transcribeORTChunks(ctx context.Context, native streamingORTSession, chunks <-chan []float32, chunkSamples int, output chan<- ASRResult) error {
	emit := func(results []ortgenai.ASRResult, nativeErr error) error {
		for _, fragment := range results {
			var result ASRResult
			appendORTASRResult(&result, fragment)
			select {
			case output <- result:
			case <-ctx.Done():
				return errors.Join(nativeErr, ctx.Err())
			}
		}
		return nativeErr
	}
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case samples, ok := <-chunks:
			if !ok {
				return emit(native.Flush(ctx))
			}
			if len(samples) == 0 {
				return errors.New("streaming ASR audio chunk is empty")
			}
			for offset := 0; offset < len(samples); {
				end := offset + min(chunkSamples, len(samples)-offset)
				if err := emit(native.Process(ctx, samples[offset:end])); err != nil {
					return err
				}
				offset = end
			}
		}
	}
}
