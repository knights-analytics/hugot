//go:build cgo && (ORT || ALL)

package backends

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"

	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/util/fileutil"
	"github.com/knights-analytics/ortgenai"
)

type ORTModel struct {
	Session        *coreORTSession
	Generative     *generativeORTProvider
	SessionOptions *coreORTOptions
	Options        *options.OrtOptions
}

type ortSessionStopFilter struct {
	stops      []string
	maxStopLen int
	tail       string
	stopped    bool
}

func newORTSessionGenerationContext(ctx, sessionContext context.Context) (context.Context, context.CancelFunc) {
	generateCtx, cancel := context.WithCancel(ctx)
	stopSession := context.AfterFunc(sessionContext, cancel)
	return generateCtx, func() {
		stopSession()
		cancel()
	}
}

func newORTSessionStopFilter(stops []string) ortSessionStopFilter {
	filter := ortSessionStopFilter{}
	for _, stop := range stops {
		if stop == "" {
			continue
		}
		filter.stops = append(filter.stops, stop)
		if len(stop) > filter.maxStopLen {
			filter.maxStopLen = len(stop)
		}
	}
	return filter
}

func (f *ortSessionStopFilter) push(token string) (string, bool) {
	if f.stopped {
		return "", true
	}
	combined := f.tail + token
	matchIndex := -1
	for _, stop := range f.stops {
		if index := strings.Index(combined, stop); index >= 0 && (matchIndex < 0 || index < matchIndex) {
			matchIndex = index
		}
	}
	if matchIndex >= 0 {
		f.tail = ""
		f.stopped = true
		return combined[:matchIndex], true
	}
	if f.maxStopLen == 0 {
		return combined, false
	}
	maxSuffixLen := min(len(combined), f.maxStopLen-1)
	for suffixLen := maxSuffixLen; suffixLen > 0; suffixLen-- {
		suffix := combined[len(combined)-suffixLen:]
		for _, stop := range f.stops {
			if len(stop) >= suffixLen && strings.HasPrefix(stop, suffix) {
				f.tail = suffix
				return combined[:len(combined)-suffixLen], false
			}
		}
	}
	f.tail = ""
	return combined, false
}

func (f *ortSessionStopFilter) flush() string {
	tail := f.tail
	f.tail = ""
	return tail
}

func (m *ORTModel) Close() error {
	if m == nil {
		return nil
	}
	var closeErr error
	if m.Session != nil {
		closeErr = errors.Join(closeErr, m.Session.Close())
		m.Session = nil
	}
	if m.Generative != nil {
		closeErr = errors.Join(closeErr, m.Generative.Close())
		m.Generative = nil
	}
	return closeErr
}

var generativeBackendMutex = sync.Mutex{}

func mapORTOptions(options *options.Options) ([]string, map[string]map[string]string, error) {
	if options == nil || options.ORTOptions == nil {
		return []string{}, map[string]map[string]string{}, nil // let default EPs be used
	}
	ortOptions := options.ORTOptions

	var providers []string
	providerOptions := map[string]map[string]string{}

	// CUDA
	if ortOptions.CudaOptions != nil {
		providers = append(providers, "cuda")
		providerOptions["cuda"] = ortOptions.CudaOptions
	}

	// CoreML
	if ortOptions.CoreMLOptions != nil {
		providers = append(providers, "CoreMLExecutionProvider")
		providerOptions["CoreMLExecutionProvider"] = ortOptions.CoreMLOptions
	}

	// DirectML
	if ortOptions.DirectMLOptions != nil {
		providers = append(providers, "DmlExecutionProvider")
		// Map device id to string map expected by advanced session
		providerOptions["DmlExecutionProvider"] = map[string]string{
			"device_id": strconv.Itoa(*ortOptions.DirectMLOptions),
		}
	}

	// OpenVINO
	if ortOptions.OpenVINOOptions != nil {
		providers = append(providers, "OpenVINOExecutionProvider")
		providerOptions["OpenVINOExecutionProvider"] = ortOptions.OpenVINOOptions
	}

	// TensorRT
	if ortOptions.TensorRTOptions != nil {
		providers = append(providers, "TensorrtExecutionProvider")
		providerOptions["TensorrtExecutionProvider"] = ortOptions.TensorRTOptions
	}

	// TensorRT
	if ortOptions.NvTensorRTRTXOptions != nil {
		providers = append(providers, "NvTensorRTRTXExecutionProvider")
		providerOptions["NvTensorRTRTXExecutionProvider"] = ortOptions.NvTensorRTRTXOptions
	}

	// Extra EPs
	if len(ortOptions.ExtraExecutionProviders) > 0 {
		for _, ep := range ortOptions.ExtraExecutionProviders {
			providers = append(providers, ep.Name)
			providerOptions[ep.Name] = ep.Options
		}
	}
	return providers, providerOptions, nil
}

func createORTGenerativeSession(ctx context.Context, model *Model, options *options.Options) error {
	if strings.HasPrefix(model.Path, "s3:") {
		return errors.New("ORT Gen AI does not support S3 paths. Please download the model to a local directory and try again")
	}
	if options == nil || options.ORTOptions == nil {
		return errors.New("ORT options must be provided to create a generative model")
	}
	if options.ORTOptions.UseEngine {
		if err := validateGenerativeEngineOptions(options.ORTOptions); err != nil {
			return err
		}
		if err := initialiseORTGenAI(ctx, options); err != nil {
			return err
		}
		engine, err := newEngineORTOwner(model.Path)
		if err != nil {
			return fmt.Errorf("error creating ortgenai engine: %w", err)
		}
		model.ORTModel = &ORTModel{
			Generative: &generativeORTProvider{engine: engine},
			Options:    options.ORTOptions,
		}
		return nil
	}

	err := initialiseORTGenAI(ctx, options)
	if err != nil {
		return err
	}

	providers, providerOptions, err := mapORTOptions(options)
	if err != nil {
		return fmt.Errorf("error mapping ORT options for generative session: %w", err)
	}

	ortGenAiSession, err := ortgenai.CreateSessionWithOptions(model.Path, providers, providerOptions)
	if err != nil {
		return fmt.Errorf("error creating ortgenai session: %w", err)
	}
	// Adapters are session-scoped; apply them before the adapter manager is
	// released. Session.Destroy() (via generativeORTAdapter.Close()) destroys
	// any activeAdapters it holds, so no separate teardown is needed here.
	if err := applyGenAIAdapters(ortGenAiSession, options.ORTOptions); err != nil {
		ortGenAiSession.Destroy()
		return fmt.Errorf("applying GenAI adapters: %w", err)
	}
	model.ORTModel = &ORTModel{
		Generative: &generativeORTProvider{session: ortGenAiSession},
		Options:    options.ORTOptions,
	}
	return nil
}

func validateGenerativeEngineOptions(options *options.OrtOptions) error {
	if options == nil {
		return errors.New("ORT options must be provided to create a generative engine")
	}
	if options.CudaOptions != nil || options.CoreMLOptions != nil || options.DirectMLOptions != nil ||
		options.OpenVINOOptions != nil || options.TensorRTOptions != nil || options.NvTensorRTRTXOptions != nil ||
		len(options.ExtraExecutionProviders) != 0 {
		return errors.New("WithGenerativeEngine does not support custom execution providers; use session generation without WithGenerativeEngine to configure providers")
	}
	if options.Telemetry != nil || options.IntraOpNumThreads != nil || options.InterOpNumThreads != nil ||
		options.CPUMemArena != nil || options.MemPattern != nil || options.ParallelExecutionMode != nil ||
		options.IntraOpSpinning != nil || options.InterOpSpinning != nil || options.LogSeverityLevel != nil ||
		options.GraphOptimizationLevel != nil || options.OptimizedModelFilePath != nil ||
		options.ProfilingEnabled != nil || options.ProfilingFilePrefix != nil ||
		options.GenAIGPUDeviceID != nil || options.GenAILogFile != nil || options.GenAILogStream != nil ||
		len(options.GenAIAdapters) > 0 || options.GenAIActiveAdapter != nil || options.UseMTP != nil {
		return errors.New("WithGenerativeEngine does not support session-specific ORT options or GenAI runtime controls; use session generation without WithGenerativeEngine")
	}
	return nil
}

func initialiseORTGenAI(ctx context.Context, options *options.Options) error {
	generativeBackendMutex.Lock()
	defer generativeBackendMutex.Unlock()

	if !ortgenai.IsInitialized() {
		if options == nil || options.ORTOptions == nil {
			return fmt.Errorf("ORT options must be provided to initialize ortgenai")
		}
		LibraryDir := options.ORTOptions.LibraryDir
		if LibraryDir == nil || *LibraryDir == "" {
			return fmt.Errorf("ORT library path must be provided to initialize ortgenai")
		}

		var libraryFileName string
		switch runtime.GOOS {
		case "windows":
			libraryFileName = "onnxruntime-genai.dll"
		case "darwin":
			libraryFileName = "libonnxruntime-genai.dylib"
		case "linux":
			libraryFileName = "libonnxruntime-genai.so"
		}
		libraryPath := fileutil.PathJoinSafe(*LibraryDir, libraryFileName)
		exists, err := fileutil.FileExists(ctx, libraryPath)
		if err != nil {
			return fmt.Errorf("error checking ortgenai library path: %w", err)
		}
		if !exists {
			return fmt.Errorf("cannot find the ortgenai library at: %s", libraryPath)
		}
		ortgenai.SetSharedLibraryPath(libraryPath)
		err = ortgenai.InitializeEnvironment()
		if err != nil {
			return fmt.Errorf("error initializing the ort genai environment: %w", err)
		}
		if applyErr := applyGenAIProcessControls(options.ORTOptions); applyErr != nil {
			return fmt.Errorf("applying ORT GenAI process controls: %w", applyErr)
		}
	}
	return nil
}

// applyGenAIProcessControls sets process-wide GPU device and log options
// after the GenAI environment has been initialized. Safe to call with nil
// opts (no-op if all fields are nil).
func applyGenAIProcessControls(opts *options.OrtOptions) error {
	if opts == nil {
		return nil
	}
	var err error
	if opts.GenAIGPUDeviceID != nil {
		if e := ortgenai.SetGPUDeviceID(*opts.GenAIGPUDeviceID); e != nil {
			err = errors.Join(err, e)
		}
	}
	if opts.GenAILogFile != nil {
		if e := ortgenai.SetLogString("filename", *opts.GenAILogFile); e != nil {
			err = errors.Join(err, e)
		}
	}
	if opts.GenAILogStream != nil {
		if e := ortgenai.SetLogString("stream", *opts.GenAILogStream); e != nil {
			err = errors.Join(err, e)
		}
	}
	return err
}

// applyGenAIAdapters creates the adapter manager, loads all configured
// adapters into it, and activates the selected one. The session's Destroy
// handles final adapter cleanup (Destroy sets activeAdapters to nil first).
func applyGenAIAdapters(session *ortgenai.Session, opts *options.OrtOptions) error {
	if len(opts.GenAIAdapters) == 0 {
		return nil
	}
	if opts.GenAIActiveAdapter == nil {
		return fmt.Errorf("genAIActiveAdapter must be set when genAIAdapters is configured")
	}

	adapters, err := session.CreateAdapters()
	if err != nil {
		return fmt.Errorf("creating adapter manager: %w", err)
	}

	for _, a := range opts.GenAIAdapters {
		if err := adapters.Load(a.Path, a.Name); err != nil {
			adapters.Destroy()
			return fmt.Errorf("loading adapter %q: %w", a.Name, err)
		}
	}

	if err := session.SetActiveAdapter(adapters, *opts.GenAIActiveAdapter); err != nil {
		adapters.Destroy()
		return fmt.Errorf("setting active adapter %q: %w", *opts.GenAIActiveAdapter, err)
	}
	return nil
}

func runGenerativeORTSessionOnBatch(ctx context.Context, batch *PipelineBatch, p *BasePipeline, maxLength int, stopSequences []string, temperature *float64, topP *float64, seed *int, tools []string, guidance *Guidance) (chan SequenceDelta, chan error, error) {
	if p.SessionContext == nil {
		return nil, nil, errors.New("no session context")
	}
	select {
	case <-ctx.Done():
		return nil, nil, ctx.Err()
	case <-p.SessionContext.Done():
		return nil, nil, p.SessionContext.Err()
	default:
	}

	if p.Model.ORTModel.Generative == nil {
		return nil, nil, errors.New("ORT generative adapter is not initialized")
	}
	session := p.Model.ORTModel.Generative.session
	engine := p.Model.ORTModel.Generative.engine
	if session == nil && engine == nil {
		return nil, nil, errors.New("ORT generative session/engine is not initialized")
	}

	inputs, ok := batch.InputValues.([][]ortgenai.Message)
	if !ok {
		return nil, nil, fmt.Errorf("invalid input type %T for generative ORT session", batch.InputValues)
	}
	if engine != nil {
		if batch.Images != nil || batch.Audios != nil {
			return nil, nil, errors.New("WithGenerativeEngine supports text-only generation; use session generation for image or audio inputs")
		}
		return engine.generate(ctx, p.SessionContext, batch, inputs, maxLength, stopSequences, temperature, topP, seed, tools, guidance)
	}
	if session == nil {
		return nil, nil, errors.New("ORT generative session is not initialized")
	}

	// Check if we have multimodal tensors to use instead of text tokenization
	var ortTokenStream <-chan ortgenai.SequenceDelta
	var ortErrorStream <-chan error
	var err error

	// Map optional guidance config to ortgenai type.
	var ortGuidance *ortgenai.Guidance
	if guidance != nil {
		ortGuidance = &ortgenai.Guidance{
			Type:           ortgenai.GuidanceType(guidance.Type),
			Data:           guidance.Data,
			EnableFFTokens: guidance.EnableFFTokens,
		}
	}

	generateCtx, cancel := newORTSessionGenerationContext(ctx, p.SessionContext)

	// Detect any media (image or audio) in the batch. CreateMessagesORT is
	// pure (no native calls); this dispatch point is guaranteed to have the
	// ORT GenAI environment initialized and the session ready. The native
	// multimodal API is single-prompt, so the batch-safe wrapper in ortgenai
	// (plan Step 1) drives one generation per conversation and relabels
	// sequence indexes back to the batch.
	hasMedia := false
	for _, conv := range batch.MultimodalMessages {
		if conversationHasImages(conv) || conversationHasAudio(conv) {
			hasMedia = true
			break
		}
	}
	if !hasMedia {
		// MTP is runtime, not session, so it dispatches here at call time:
		// opt-in via OrtOptions.UseMTP, text-only, session path only (the
		// engine path rejects it in validateGenerativeEngineOptions). The
		// native path is greedy, so temperature/top-p/seed/guidance are
		// ignored by ortgenai.GenerateMTP and MaxLength is the only option
		// it honors.
		useMTP := p.Model.ORTModel.Options != nil && p.Model.ORTModel.Options.UseMTP != nil && *p.Model.ORTModel.Options.UseMTP
		if useMTP {
			ortTokenStream, ortErrorStream, err = session.GenerateMTP(generateCtx, inputs, tools, &ortgenai.GenerationOptions{MaxLength: maxLength})
			if err != nil {
				cancel()
				return nil, nil, fmt.Errorf("error during MTP generation start: %w", err)
			}
		} else {
			ortTokenStream, ortErrorStream, err = session.Generate(generateCtx, inputs, tools, &ortgenai.GenerationOptions{MaxLength: maxLength, Temperature: temperature, TopP: topP, Seed: seed, Guidance: ortGuidance})
			if err != nil {
				cancel()
				return nil, nil, fmt.Errorf("error during generation start: %w", err)
			}
		}
	} else {
		if session == nil {
			cancel()
			return nil, nil, errors.New("multimodal generation requires a session, but only engine is initialized")
		}
		convs, destroyers, loadErr := loadMultimodalConversations(batch.MultimodalMessages, inputs)
		if loadErr != nil {
			cancel()
			return nil, nil, loadErr
		}
		if len(destroyers) > 0 {
			previousDestroy := batch.DestroyMultimodal
			batch.DestroyMultimodal = func() error {
				// ortgenai.Images/Audios Destroy() are void, so we can't
				// collect errors from them; we only chain a prior hook (if
				// any) which may return its own error.
				for _, d := range destroyers {
					d()
				}
				if previousDestroy != nil {
					return previousDestroy()
				}
				return nil
			}
		}
		ortTokenStream, ortErrorStream, err = session.GenerateMultimodal(generateCtx, convs, tools, &ortgenai.GenerationOptions{MaxLength: maxLength, Temperature: temperature, TopP: topP, Seed: seed, Guidance: ortGuidance})
		if err != nil {
			cancel()
			return nil, nil, fmt.Errorf("error during multimodal generation start: %w", err)
		}
	}

	tokenStream := make(chan SequenceDelta, 10)
	errorStream := make(chan error, 1)

	completeSequences := map[int]bool{}

	stopFilters := make([]ortSessionStopFilter, len(inputs))
	for i := range stopFilters {
		stopFilters[i] = newORTSessionStopFilter(stopSequences)
	}

	completedCount := 0
	totalSequences := len(inputs)
	tokenForwardDone := make(chan struct{})

	go func() {
		defer func() {
			destroyErr := batch.Destroy()
			cancel()
			if destroyErr != nil {
				select {
				case errorStream <- destroyErr:
				default:
				}
			}
			close(tokenStream)
			close(tokenForwardDone)
		}()
		for {
			select {
			case <-generateCtx.Done():
				return
			case tokenDelta, ok := <-ortTokenStream:
				if !ok {
					for sequence := range stopFilters {
						if !completeSequences[sequence] {
							if tail := stopFilters[sequence].flush(); tail != "" {
								if !forwardORTSessionDelta(generateCtx, tokenStream, SequenceDelta{Token: tail, Sequence: sequence}) {
									return
								}
							}
						}
					}
					return
				}
				sequence := tokenDelta.Sequence
				if completeSequences[sequence] {
					// Already complete; ignore further tokens for this sequence.
					continue
				}
				if tokenDelta.EOSReached {
					// EOS terminates sequence; no token content to forward.
					if tail := stopFilters[sequence].flush(); tail != "" {
						if !forwardORTSessionDelta(generateCtx, tokenStream, SequenceDelta{Token: tail, Sequence: sequence}) {
							return
						}
					}
					completeSequences[sequence] = true
					completedCount++
					if completedCount == totalSequences {
						cancel()
						return
					}
					continue
				}

				output, stopped := stopFilters[sequence].push(tokenDelta.Token)
				if output != "" {
					if !forwardORTSessionDelta(generateCtx, tokenStream, SequenceDelta{Token: output, Sequence: sequence}) {
						return
					}
				}
				if stopped {
					completeSequences[sequence] = true
					completedCount++
					if completedCount == totalSequences {
						cancel()
						return
					}
				}
			}
		}
	}()

	go func() {
		defer func() {
			<-tokenForwardDone
			close(errorStream)
		}()
		for {
			select {
			case <-generateCtx.Done():
				return
			case err, ok := <-ortErrorStream:
				if !ok {
					return
				}
				if err != nil {
					select {
					case errorStream <- err:
					case <-generateCtx.Done():
						return
					}
				}
			}
		}
	}()
	return tokenStream, errorStream, nil
}

func forwardORTSessionDelta(ctx context.Context, stream chan<- SequenceDelta, delta SequenceDelta) bool {
	select {
	case stream <- delta:
		return true
	case <-ctx.Done():
		return false
	}
}

func createORTModelBackend(model *Model, options *options.Options) error {
	sessionOptions, ok := options.BackendOptions.(*coreORTOptions)
	if !ok || sessionOptions == nil {
		return errors.New("invalid ORT session options")
	}

	var cwd string
	var err error
	var onnxBytes []byte

	if model.OnnxReader != nil {
		onnxBytes, err = io.ReadAll(model.OnnxReader)
		if err != nil {
			return err
		}
	} else {
		// TODO: currently models with external data can only load from regular filesystems, and require dir change
		cwd, err = os.Getwd()
		if err != nil {
			return err
		}
		err = os.Chdir(model.Path)
		if err != nil {
			return err
		}

	}

	session, inputs, outputs, err := newCoreORTSession(model.OnnxPath, onnxBytes, sessionOptions)
	if err != nil {
		return err
	}

	model.ORTModel = &ORTModel{
		Session:        session,
		SessionOptions: sessionOptions,
		Options:        options.ORTOptions,
	}
	model.InputsMeta = inputs
	model.OutputsMeta = outputs
	if cwd != "" {
		err = os.Chdir(cwd)
	}

	return err
}

func createInputTensorsORT(batch *PipelineBatch, model *Model) error {
	batchSize := batch.Size
	maxSequenceLength := batch.MaxSequenceLength
	queryCount := batch.QueryCount
	if queryCount < 1 {
		queryCount = 1
	}

	// 1) prepare result containers - now we use all inputs, not filtering
	inputVals := make([]*coreORTTensor, len(model.InputsMeta))
	masks := make([][]bool, batchSize)

	// 2) build each tensor
	for mi, meta := range model.InputsMeta {
		if strings.EqualFold(meta.Name, "input_features") {
			backing, dimensions, err := flattenAudioFeatures(batch.AudioFeatures, batch.Size)
			if err != nil {
				return err
			}
			t, err := newCoreFloat32Tensor(dimensions, backing)
			if err != nil {
				return err
			}
			inputVals[mi] = t
			continue
		}
		if strings.EqualFold(meta.Name, "is_longer") {
			if len(batch.AudioIsLonger) != batch.Size {
				return fmt.Errorf("audio duration flags do not match batch size")
			}
			shape := make([]int64, len(meta.Dimensions))
			shape[0] = int64(batch.Size)
			for i := 1; i < len(shape); i++ {
				shape[i] = 1
				if meta.Dimensions[i] > 0 {
					shape[i] = meta.Dimensions[i]
				}
			}
			backing := make([]bool, batch.Size)
			copy(backing, batch.AudioIsLonger)
			t, err := newCoreBoolTensor(shape, backing)
			if err != nil {
				return err
			}
			inputVals[mi] = t
			continue
		}
		if isImageInput(meta.Name) {
			backing, dimensions, err := flattenImageValues(model, batch.Size, batch.ImageValues)
			if err != nil {
				return err
			}
			t, err := newCoreFloat32Tensor(dimensions, backing)
			if err != nil {
				return err
			}
			inputVals[mi] = t
			continue
		}
		textRank := len(meta.Dimensions)
		if textRank != 2 && textRank != 3 {
			return fmt.Errorf("unsupported text input rank %d for %s", textRank, meta.Name)
		}
		if textRank == 3 && batch.QueryCount == 0 {
			return fmt.Errorf("input %s requires a query dimension", meta.Name)
		}
		textBatchSize, textQueryCount := textTensorBatchCounts(textRank, batchSize, queryCount, batch.QueryCount > 0)
		backing := make([]int64, textBatchSize*textQueryCount*maxSequenceLength)
		idx := 0
		switch meta.Name {
		case "input_ids":
			for bi := range textBatchSize {
				for qi := range textQueryCount {
					inputIndex := bi
					if textRank == 3 {
						inputIndex = qi
					}
					inp := batch.Input[inputIndex]
					seqLen := len(inp.TokenIDs)
					maskRow := make([]bool, maxSequenceLength)
					for pos := range maxSequenceLength {
						if pos < seqLen {
							backing[idx] = int64(inp.TokenIDs[pos])
							maskRow[pos] = true
						}
						idx++
					}
					if bi < len(masks) && qi == 0 {
						masks[bi] = maskRow
					}
				}
			}
		case "token_type_ids":
			for bi := range textBatchSize {
				for qi := range textQueryCount {
					inputIndex := bi
					if textRank == 3 {
						inputIndex = qi
					}
					inp := batch.Input[inputIndex]
					for pos := range maxSequenceLength {
						if pos < len(inp.TypeIDs) {
							backing[idx] = int64(inp.TypeIDs[pos])
						}
						idx++
					}
				}
			}
		case "attention_mask":
			for bi := range textBatchSize {
				for qi := range textQueryCount {
					inputIndex := bi
					if textRank == 3 {
						inputIndex = qi
					}
					inp := batch.Input[inputIndex]
					for pos := range maxSequenceLength {
						if pos < len(inp.AttentionMask) {
							backing[idx] = int64(inp.AttentionMask[pos])
						}
						idx++
					}
				}
			}
		case "position_ids":
			for range textBatchSize {
				for range textQueryCount {
					for pos := range maxSequenceLength {
						backing[idx] = int64(pos + 1)
						idx++
					}
				}
			}
		default:
			return fmt.Errorf("unrecognized input %q", meta.Name)
		}

		// create the ONNX Runtime tensor for regular inputs
		dimensions := []int64{int64(textBatchSize), int64(maxSequenceLength)}
		if textRank == 3 {
			dimensions = []int64{int64(textBatchSize), int64(textQueryCount), int64(maxSequenceLength)}
		}
		t, err := newCoreInt64Tensor(dimensions, backing)
		if err != nil {
			return err
		}
		inputVals[mi] = t
	}

	// 3) assign and prepare cleanup
	batch.InputValues = inputVals
	batch.PaddingMask = masks
	batch.DestroyInputs = func() error {
		var agg error
		if values, ok := batch.InputValues.([]*coreORTTensor); ok {
			for _, t := range values {
				agg = errors.Join(agg, t.Close())
			}
		} else {
			agg = errors.Join(agg, errors.New("batch.InputValues has incorrect type"))
		}
		return agg
	}
	return nil
}

func runORTSessionOnBatch(ctx context.Context, batch *PipelineBatch, p *BasePipeline) error {
	if p.SessionContext == nil {
		return errors.New("no session context")
	}
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-p.SessionContext.Done():
		return p.SessionContext.Err()
	default:
	}

	outputTensors, err := p.Model.ORTModel.Session.Run(ctx, batch.InputValues.([]*coreORTTensor))
	if err != nil {
		return err
	}
	defer func() {
		for _, tensor := range outputTensors {
			err = errors.Join(tensor.Close())
		}
	}()

	convertedOutput := make([]any, len(outputTensors))
	for i, t := range outputTensors {
		dimensions := Shape(t.tensor.Shape())
		if values, dataErr := t.Float32Data(); dataErr == nil {
			convertedOutput[i] = reshapeOutput(values, dimensions, batch.Size, batch.PaddingMask)
			continue
		}
		if values, dataErr := t.Int64Data(); dataErr == nil {
			convertedOutput[i] = reshapeOutput(values, dimensions, batch.Size, batch.PaddingMask)
			continue
		}
		return fmt.Errorf("unsupported ORT output type for %q", p.Model.OutputsMeta[i].Name)
	}
	// store resulting tensors
	batch.OutputValues = convertedOutput
	return err
}

// createTabularTensorsORT flattens [][]float32 features into a [batch, feature_dim] tensor.
// Currently supports models with a single input of 2D shape (batch, features).
func createAudioTensorsORT(batch *PipelineBatch, model *Model, samples [][]float32) error {
	if len(samples) == 0 || len(samples) != batch.Size {
		return errors.New("audio samples do not match batch size")
	}
	if len(model.InputsMeta) != 1 {
		return errors.New("audio models with multiple inputs are not supported")
	}
	dims := model.InputsMeta[0].Dimensions
	if len(dims) != 2 && len(dims) != 3 {
		return fmt.Errorf("unsupported audio input rank %d", len(dims))
	}
	maxSamples := 0
	for _, waveform := range samples {
		if len(waveform) > maxSamples {
			maxSamples = len(waveform)
		}
	}
	backing := make([]float32, len(samples)*maxSamples)
	for i, waveform := range samples {
		copy(backing[i*maxSamples:], waveform)
	}
	shape := []int64{int64(len(samples)), int64(maxSamples)}
	if len(dims) == 3 {
		shape = []int64{int64(len(samples)), 1, int64(maxSamples)}
	}
	tensor, err := newCoreFloat32Tensor(shape, backing)
	if err != nil {
		return err
	}
	batch.InputValues = []*coreORTTensor{tensor}
	batch.DestroyInputs = tensor.Close
	return nil
}

func createTabularTensorsORT(batch *PipelineBatch, model *Model, features [][]float32) error {
	if len(features) != batch.Size {
		return fmt.Errorf("features batch size %d does not match PipelineBatch size %d", len(features), batch.Size)
	}
	if len(model.InputsMeta) < 1 {
		return fmt.Errorf("model has no input metadata")
	}
	// Assume first input is the tabular data.
	inMeta := model.InputsMeta[0]
	dims := []int64(inMeta.Dimensions)
	if len(dims) != 2 {
		return fmt.Errorf("expected 2D input shape for tabular model, got %d dims", len(dims))
	}
	featDim := int(dims[len(dims)-1])
	if featDim <= 0 {
		// dynamic feature dim: infer from first sample
		featDim = len(features[0])
	}
	// Validate feature lengths
	for i := range features {
		if len(features[i]) != featDim {
			return fmt.Errorf("input %d has %d features, expected %d", i, len(features[i]), featDim)
		}
	}

	backing := make([]float32, batch.Size*featDim)

	idx := 0
	for _, featVec := range features {
		for _, val := range featVec {
			backing[idx] = val
			idx++
		}
	}

	t, err := newCoreFloat32Tensor([]int64{int64(batch.Size), int64(featDim)}, backing)
	if err != nil {
		return err
	}
	values := make([]*coreORTTensor, len(model.InputsMeta))
	values[0] = t
	batch.InputValues = values
	batch.DestroyInputs = func() error { return t.Close() }
	// No padding mask for tabular
	batch.PaddingMask = nil
	batch.MaxSequenceLength = 0
	return nil
}

func createImageTensorsORT(batch *PipelineBatch, model *Model, preprocessed [][][][]float32) error {
	if len(preprocessed) == 0 {
		return errors.New("no preprocessed images provided")
	}

	n, c, h, w := len(preprocessed), len(preprocessed[0]), len(preprocessed[0][0]), len(preprocessed[0][0][0])
	imgBacking := make([]float32, n*c*h*w)
	idx := 0
	for i := range n {
		for ch := range c {
			for y := range h {
				for x := range w {
					imgBacking[idx] = preprocessed[i][ch][y][x]
					idx++
				}
			}
		}
	}
	imgTensor, err := newCoreFloat32Tensor([]int64{int64(n), int64(c), int64(h), int64(w)}, imgBacking)
	if err != nil {
		return err
	}
	success := false
	var destroyers []func() error
	defer func() {
		if !success {
			err = errors.Join(imgTensor.Close())
			for _, destroy := range destroyers {
				err = errors.Join(destroy())
			}
		}
	}()

	// Prepare inputs slice according to model input metadata order.
	values := make([]*coreORTTensor, len(model.InputsMeta))
	destroyers = make([]func() error, 0, len(values))

	// Helper to infer mask dims
	inferMaskDims := func(s Shape) (int64, int64) {
		// Try to find known H and W; fallback to image h,w
		var mh, mw int64
		if len(s) >= 2 {
			for _, d := range s {
				if d > 1 && mh == 0 {
					mh = d
					continue
				}
				if d > 1 && mh != 0 && mw == 0 {
					mw = d
					break
				}
			}
		}
		if mh == 0 || mw == 0 {
			mh, mw = int64(h), int64(w)
		}
		return mh, mw
	}

	for i, meta := range model.InputsMeta {
		lower := strings.ToLower(meta.Name)
		if strings.Contains(lower, "mask") {
			// Build pixel_mask tensor of ones using int64 dtype, shape [n, H, W] or [n,1,H,W] depending on meta.
			mh, mw := inferMaskDims(meta.Dimensions)
			// Default to 3D [n,H,W]
			var shape []int64
			if len(meta.Dimensions) == 4 {
				// Some models expect [n,1,H,W]
				shape = []int64{int64(n), 1, mh, mw}
			} else {
				shape = []int64{int64(n), mh, mw}
			}
			maskSize := 1
			for _, d := range shape {
				maskSize *= int(d)
			}
			maskBacking := make([]int64, maskSize)
			for j := range maskBacking {
				maskBacking[j] = 1
			}
			maskTensor, mErr := newCoreInt64Tensor(shape, maskBacking)
			if mErr != nil {
				// If creating 4D fails, try 3D fallback
				if len(shape) == 4 {
					shape = []int64{int64(n), mh, mw}
					maskSize = n * int(mh) * int(mw)
					maskBacking = make([]int64, maskSize)
					for j := range maskBacking {
						maskBacking[j] = 1
					}
					maskTensor, mErr = newCoreInt64Tensor(shape, maskBacking)
				}
				if mErr != nil {
					return mErr
				}
			}
			values[i] = maskTensor
			destroyers = append(destroyers, maskTensor.Close)
		} else {
			values[i] = imgTensor
			// Only destroy once; avoid double-destroy if multiple inputs map to same tensor
		}
	}
	// If only one input, just that tensor
	if len(values) == 1 {
		values[0] = imgTensor
	}
	batch.InputValues = values
	batch.DestroyInputs = func() error {
		var agg error
		agg = errors.Join(agg, imgTensor.Close())
		for _, d := range destroyers {
			agg = errors.Join(agg, d())
		}
		return agg
	}
	success = true
	return err
}

func CreateMessagesORT(batch *PipelineBatch, inputs any, systemPrompt string) error {
	batch.MultimodalMessages = nil
	switch inputCast := inputs.(type) {
	case []string:
		ortMessages := make([][]ortgenai.Message, len(inputCast))
		addSystemPrompt := systemPrompt != ""
		systemPromptMessage := ortgenai.Message{Role: "system", Content: systemPrompt}
		for i, input := range inputCast {
			if addSystemPrompt {
				m := make([]ortgenai.Message, 2)
				m[0] = systemPromptMessage
				m[1] = ortgenai.Message{Role: "user", Content: input}
				ortMessages[i] = m
			} else {
				ortMessages[i] = []ortgenai.Message{
					{Role: "user", Content: input},
				}
			}
		}
		batch.InputValues = ortMessages
	case [][]Message:
		batch.MultimodalMessages = cloneORTMessages(inputCast)
		ortMessages := createORTMessages(inputCast, systemPrompt)
		batch.InputValues = ortMessages
	default:
		return fmt.Errorf("invalid input type %T for CreateMessagesORT", inputCast)
	}
	return nil
}

func cloneORTMessages(messages [][]Message) [][]Message {
	cloned := make([][]Message, len(messages))
	for i, conversation := range messages {
		cloned[i] = make([]Message, len(conversation))
		for j, message := range conversation {
			cloned[i][j] = message
			cloned[i][j].ImageURLs = append([]string(nil), message.ImageURLs...)
			cloned[i][j].AudioURLs = append([]string(nil), message.AudioURLs...)
		}
	}
	return cloned
}

// conversationHasImages reports whether any message in the conversation has image URLs.
func conversationHasImages(conversation []Message) bool {
	for _, message := range conversation {
		if len(message.ImageURLs) > 0 {
			return true
		}
	}
	return false
}

// flattenImageURLs collects every image URL in message order across a conversation.
func flattenImageURLs(conversation []Message) []string {
	var paths []string
	for _, message := range conversation {
		paths = append(paths, message.ImageURLs...)
	}
	return paths
}

// conversationHasAudio reports whether any message in the conversation has audio URLs.
func conversationHasAudio(conversation []Message) bool {
	for _, message := range conversation {
		if len(message.AudioURLs) > 0 {
			return true
		}
	}
	return false
}

// flattenAudioURLs collects every audio URL in message order across a conversation.
func flattenAudioURLs(conversation []Message) []string {
	var paths []string
	for _, message := range conversation {
		paths = append(paths, message.AudioURLs...)
	}
	return paths
}

// toORTMessages converts a per-conversation []backends.Message into the
// []ortgenai.Message expected by ortgenai.MultimodalConversation. Media is
// passed separately on the conversation, so only role and text are copied.
func toORTMessages(conv []Message) []ortgenai.Message {
	out := make([]ortgenai.Message, len(conv))
	for i, m := range conv {
		out[i] = ortgenai.Message{Role: m.Role, Content: m.Content}
	}
	return out
}

// loadMultimodalConversations builds the per-conversation multimodal inputs for
// the batch-safe session wrapper, loading each conversation's image and audio
// paths independently so media stays scoped to its conversation. Prepared messages
// retain the image tags and system prompt added by CreateMessagesORT. It returns the
// conversations plus one destroy function per loaded native resource; callers
// should chain these into batch.DestroyMultimodal and invoke them (reversibly)
// on a failed load.
func loadMultimodalConversations(messageConvs [][]Message, prepared [][]ortgenai.Message) ([]ortgenai.MultimodalConversation, []func(), error) {
	if len(messageConvs) != len(prepared) {
		return nil, nil, errors.New("multimodal media and prepared messages must have the same conversation count")
	}
	out := make([]ortgenai.MultimodalConversation, len(messageConvs))
	var destroyers []func()
	release := func() {
		for i := len(destroyers) - 1; i >= 0; i-- {
			destroyers[i]()
		}
	}
	for i, conv := range messageConvs {
		c := ortgenai.MultimodalConversation{Messages: prepared[i]}
		if paths := flattenImageURLs(conv); len(paths) > 0 {
			images, err := ortgenai.LoadImages(paths)
			if err != nil {
				release()
				return nil, nil, fmt.Errorf("loading images for conversation %d: %w", i, err)
			}
			c.Images = images
			destroyers = append(destroyers, images.Destroy)
		}
		if paths := flattenAudioURLs(conv); len(paths) > 0 {
			audios, err := ortgenai.LoadAudios(paths)
			if err != nil {
				release()
				return nil, nil, fmt.Errorf("loading audio for conversation %d: %w", i, err)
			}
			c.Audios = audios
			destroyers = append(destroyers, audios.Destroy)
		}
		out[i] = c
	}
	return out, destroyers, nil
}

func createORTMessages(inputCast [][]Message, systemPrompt string) [][]ortgenai.Message {
	ortMessages := make([][]ortgenai.Message, len(inputCast))
	addSystemPrompt := systemPrompt != ""
	systemPromptMessage := ortgenai.Message{Role: "system", Content: systemPrompt}
	for i, inputMessages := range inputCast {
		imageIndex := 1
		additionalLength := 0
		if addSystemPrompt {
			additionalLength = 1
		}
		out := make([]ortgenai.Message, len(inputMessages)+additionalLength)
		offset := 0
		if addSystemPrompt {
			out[0] = systemPromptMessage
			offset = 1
		}
		for j, message := range inputMessages {
			content := message.Content
			for range message.ImageURLs {
				if content != "" {
					content += "\n"
				}
				content += fmt.Sprintf("<|image_%d|>", imageIndex)
				imageIndex++
			}
			out[offset+j] = ortgenai.Message{Role: message.Role, Content: content}
		}
		ortMessages[i] = out
	}
	return ortMessages
}
