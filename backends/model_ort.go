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

	ort "github.com/yalue/onnxruntime_go"
	"golang.org/x/sync/errgroup"

	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/util/fileutil"
	"github.com/knights-analytics/ortgenai"
)

type ORTModel struct {
	Session           *ort.DynamicAdvancedSession
	GenerativeSession *ortgenai.Session
	GenerativeEngine  *ortgenai.Engine
	SessionOptions    *ort.SessionOptions
	Options           *options.OrtOptions
	Destroy           func() error
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
		providerOptions["NvTensorRTRTXExecutionProvider"] = ortOptions.TensorRTOptions
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

	err := initialiseORTGenAI(ctx, options)
	if err != nil {
		return err
	}

	providers, providerOptions, err := mapORTOptions(options)
	if err != nil {
		return fmt.Errorf("error mapping ORT options for generative session: %w", err)
	}

	if options.ORTOptions.UseEngine {
		ortGenAiEngine, err := ortgenai.CreateEngineWithOptions(model.Path, providers, providerOptions)
		if err != nil {
			return fmt.Errorf("error creating ortgenai engine: %w", err)
		}
		model.ORTModel = &ORTModel{
			GenerativeEngine: ortGenAiEngine,
			Options:          options.ORTOptions,
			Destroy: func() error {
				ortGenAiEngine.Destroy()
				return nil
			},
		}
	} else {
		ortGenAiSession, err := ortgenai.CreateSessionWithOptions(model.Path, providers, providerOptions)
		if err != nil {
			return fmt.Errorf("error creating ortgenai session: %w", err)
		}
		model.ORTModel = &ORTModel{
			GenerativeSession: ortGenAiSession,
			Options:           options.ORTOptions,
			Destroy: func() error {
				ortGenAiSession.Destroy()
				return nil
			},
		}
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
			libraryFileName = "libonnxruntime-genai.dll"
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

	session := p.Model.ORTModel.GenerativeSession
	engine := p.Model.ORTModel.GenerativeEngine
	if session == nil && engine == nil {
		return nil, nil, errors.New("ORT generative session/engine is not initialized")
	}

	inputs, ok := batch.InputValues.([][]ortgenai.Message)
	if !ok {
		return nil, nil, fmt.Errorf("invalid input type %T for generative ORT session", batch.InputValues)
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

	generateCtx, cancel := context.WithCancel(ctx)

	if batch.Images != nil {
		if session == nil {
			cancel()
			return nil, nil, errors.New("multimodal generation requires a session, but only engine is initialized")
		}
		images, ok := batch.Images.(*ortgenai.Images)
		if !ok {
			cancel()
			return nil, nil, fmt.Errorf("invalid images type %T for generative ORT session", batch.Images)
		}
		ortTokenStream, ortErrorStream, err = session.GenerateWithImages(generateCtx, inputs, images, tools, &ortgenai.GenerationOptions{MaxLength: maxLength, BatchSize: len(inputs), Temperature: temperature, TopP: topP, Seed: seed, Guidance: ortGuidance})
		if err != nil {
			cancel()
			return nil, nil, fmt.Errorf("error during multimodal generation start: %w", err)
		}
	} else {
		if engine != nil {
			ortTokenStream, ortErrorStream, err = engine.Generate(generateCtx, inputs, tools, &ortgenai.GenerationOptions{MaxLength: maxLength, Temperature: temperature, TopP: topP, Seed: seed, Guidance: ortGuidance})
		} else {
			ortTokenStream, ortErrorStream, err = session.Generate(generateCtx, inputs, tools, &ortgenai.GenerationOptions{MaxLength: maxLength, Temperature: temperature, TopP: topP, Seed: seed, Guidance: ortGuidance})
		}
		if err != nil {
			cancel()
			return nil, nil, fmt.Errorf("error during generation start: %w", err)
		}
	}

	tokenStream := make(chan SequenceDelta, 10)
	errorStream := make(chan error, 1)

	completeSequences := map[int]bool{}

	// Precompute stop-sequence data once (avoid scanning growing buffers).
	filteredStops := make([]string, 0, len(stopSequences))
	maxStopLen := 0
	for _, s := range stopSequences {
		if s == "" {
			continue
		}
		filteredStops = append(filteredStops, s)
		if len(s) > maxStopLen {
			maxStopLen = len(s)
		}
	}

	// Keep only a rolling tail per sequence (enough to detect stop sequences).
	// This avoids O(n^2) behavior from (a) growing strings and (b) repeatedly searching the whole prefix.
	tails := make([]string, len(inputs))

	completedCount := 0
	totalSequences := len(inputs)

	go func() {
		defer close(tokenStream)
		defer func() {
			destroyErr := batch.Destroy()
			if destroyErr != nil {
				errorStream <- destroyErr
			}
		}()
		for {
			select {
			case <-ctx.Done():
				return
			case tokenDelta, ok := <-ortTokenStream:
				if !ok {
					return
				}
				sequence := tokenDelta.Sequence
				if completeSequences[sequence] {
					// Already complete; ignore further tokens for this sequence.
					continue
				}
				if tokenDelta.EOSReached {
					// EOS terminates sequence; no token content to forward.
					completeSequences[sequence] = true
					completedCount++
					if completedCount == totalSequences {
						cancel()
						return
					}
					continue
				}

				// Forward token immediately.
				tokenStream <- SequenceDelta{Token: tokenDelta.Token, Sequence: sequence}

				// Detect stop sequences using a bounded rolling window.
				// Window size is maxStopLen + len(current token) to catch:
				//  - stops spanning the boundary between previous tail and this token
				//  - stops fully contained inside a single (possibly long) token
				if len(filteredStops) > 0 && maxStopLen > 0 {
					tail := tails[sequence] + tokenDelta.Token
					keep := maxStopLen + len(tokenDelta.Token)
					if keep < maxStopLen {
						keep = maxStopLen
					}
					if len(tail) > keep {
						tail = tail[len(tail)-keep:]
					}
					tails[sequence] = tail

					for _, s := range filteredStops {
						if strings.Contains(tail, s) {
							completeSequences[sequence] = true
							completedCount++
							if completedCount == totalSequences {
								cancel()
								return
							}
							break
						}
					}
				}
			}
		}
	}()

	go func() {
		defer close(errorStream)
		for {
			select {
			case <-ctx.Done():
				return
			case err, ok := <-ortErrorStream:
				if !ok {
					return
				}
				if err != nil {
					errorStream <- err
				}
			}
		}
	}()
	return tokenStream, errorStream, nil
}

func createORTModelBackend(model *Model, options *options.Options) error {
	sessionOptions := options.BackendOptions.(*ort.SessionOptions)

	var inputs, outputs []InputOutputInfo
	var cwd string
	var err error
	var onnxBytes []byte

	if model.OnnxReader != nil {
		onnxBytes, err = io.ReadAll(model.OnnxReader)
		if err != nil {
			return err
		}
		inputs, outputs, err = loadInputOutputMetaORTReader(onnxBytes)
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

		inputs, outputs, err = loadInputOutputMetaORTFile(model.OnnxPath)
	}
	if err != nil {
		return err
	}

	inputNames := make([]string, len(inputs))
	outputNames := make([]string, len(outputs))
	for i, v := range inputs {
		inputNames[i] = v.Name
	}
	for i, v := range outputs {
		outputNames[i] = v.Name
	}

	var session *ort.DynamicAdvancedSession
	if len(onnxBytes) > 0 {
		session, err = ort.NewDynamicAdvancedSessionWithONNXData(
			onnxBytes,
			inputNames,
			outputNames,
			sessionOptions,
		)
	} else {
		session, err = ort.NewDynamicAdvancedSession(
			model.OnnxPath,
			inputNames,
			outputNames,
			sessionOptions,
		)
	}
	if err != nil {
		return err
	}

	model.ORTModel = &ORTModel{
		Session:        session,
		SessionOptions: sessionOptions,
		Options:        options.ORTOptions,
		Destroy: func() error {
			return session.Destroy()
		},
	}
	model.InputsMeta = inputs
	model.OutputsMeta = outputs
	if cwd != "" {
		err = os.Chdir(cwd)
	}

	return err
}

func loadInputOutputMetaORTReader(onnxBytes []byte) ([]InputOutputInfo, []InputOutputInfo, error) {
	inputs, outputs, err := ort.GetInputOutputInfoWithONNXData(onnxBytes)
	if err != nil {
		return nil, nil, err
	}
	return convertORTInputOutputs(inputs), convertORTInputOutputs(outputs), nil
}

func loadInputOutputMetaORTFile(onnxPath string) ([]InputOutputInfo, []InputOutputInfo, error) {
	inputs, outputs, err := ort.GetInputOutputInfo(onnxPath)
	if err != nil {
		return nil, nil, err
	}
	return convertORTInputOutputs(inputs), convertORTInputOutputs(outputs), nil
}

func createInputTensorsORT(batch *PipelineBatch, model *Model) error {
	batchSize := batch.Size
	maxSequenceLength := batch.MaxSequenceLength
	total := batchSize * maxSequenceLength

	// 1) prepare result containers - now we use all inputs, not filtering
	inputVals := make([]ort.Value, len(model.InputsMeta))
	masks := make([][]bool, batchSize)

	// 2) build each tensor
	for mi, meta := range model.InputsMeta {
		// Handle regular input tensors
		backing := make([]int64, total)
		idx := 0
		switch meta.Name {
		case "input_ids":
			for bi, inp := range batch.Input {
				seqLen := len(inp.TokenIDs)
				maskRow := make([]bool, maxSequenceLength)
				for pos := range maxSequenceLength {
					if pos < seqLen {
						backing[idx] = int64(inp.TokenIDs[pos])
						maskRow[pos] = true
					}
					idx++
				}
				masks[bi] = maskRow
			}
		case "token_type_ids":
			for _, inp := range batch.Input {
				seqLen := len(inp.TokenIDs)
				for pos := range maxSequenceLength {
					if pos < seqLen {
						backing[idx] = int64(inp.TypeIDs[pos])
					}
					idx++
				}
			}
		case "attention_mask":
			for _, inp := range batch.Input {
				for pos := range maxSequenceLength {
					if pos < len(inp.TokenIDs) {
						backing[idx] = int64(inp.AttentionMask[pos])
					}
					idx++
				}
			}
		case "position_ids":
			for range batch.Input {
				for pos := range maxSequenceLength {
					// 1-indexed positions
					backing[idx] = int64(pos + 1)
					idx++
				}
			}
		default:
			return fmt.Errorf("unrecognized input %q", meta.Name)
		}

		// create the ONNX Runtime tensor for regular inputs
		t, err := ort.NewTensor(ort.NewShape(int64(batchSize), int64(maxSequenceLength)), backing)
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
		if values, ok := batch.InputValues.([]ort.Value); ok {
			for _, t := range values {
				agg = errors.Join(agg, t.Destroy())
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

	outputTensors := make([]ort.Value, len(p.Model.OutputsMeta))
	doneChannel := make(chan bool, 1)
	eg := errgroup.Group{}
	eg.Go(func() (err error) {
		defer func() {
			// C code does not support context, so cancelling a context and/or session will usually trigger a segfault(panic).
			// recover this here, so context can be cancelled gracefully and return an error.
			if r := recover(); r != nil {
				err = fmt.Errorf("recovered from panic: %v", r)
			}
			close(doneChannel)
		}()
		return errors.Join(err, p.Model.ORTModel.Session.Run(batch.InputValues.([]ort.Value), outputTensors))
	})

	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-p.SessionContext.Done():
		return p.SessionContext.Err()
	case <-doneChannel:
		if err := eg.Wait(); err != nil {
			return err
		}
	}

	convertedOutput := make([]any, len(outputTensors))
	for i, t := range outputTensors {
		switch v := t.(type) {
		case *ort.Tensor[float32]:
			convertedOutput[i] = ReshapeOutput(v.GetData(), p.Model.OutputsMeta[i], batch.Size, batch.PaddingMask, batch.MaxSequenceLength)
		case *ort.Tensor[int64]:
			convertedOutput[i] = ReshapeOutput(v.GetData(), p.Model.OutputsMeta[i], batch.Size, batch.PaddingMask, batch.MaxSequenceLength)
		}
	}
	// store resulting tensors
	batch.OutputValues = convertedOutput
	return nil
}

func convertORTInputOutputs(inputOutputs []ort.InputOutputInfo) []InputOutputInfo {
	inputOutputsStandardised := make([]InputOutputInfo, len(inputOutputs))
	for i, inputOutput := range inputOutputs {
		inputOutputsStandardised[i] = InputOutputInfo{
			Name:       inputOutput.Name,
			Dimensions: Shape(inputOutput.Dimensions),
		}
	}
	return inputOutputsStandardised
}

// createTabularTensorsORT flattens [][]float32 features into a [batch, feature_dim] tensor.
// Currently supports models with a single input of 2D shape (batch, features).
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

	t, err := ort.NewTensor(ort.NewShape(int64(batch.Size), int64(featDim)), backing)
	if err != nil {
		return err
	}
	values := make([]ort.Value, len(model.InputsMeta))
	values[0] = t
	batch.InputValues = values
	batch.DestroyInputs = func() error { return t.Destroy() }
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
	imgTensor, err := ort.NewTensor(ort.NewShape(int64(n), int64(c), int64(h), int64(w)), imgBacking)
	if err != nil {
		return err
	}

	// Prepare inputs slice according to model input metadata order.
	values := make([]ort.Value, len(model.InputsMeta))
	destroyers := make([]func() error, 0, len(values))

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
			maskTensor, mErr := ort.NewTensor(ort.NewShape(shape...), maskBacking)
			if mErr != nil {
				// If creating 4D fails, try 3D fallback
				if len(shape) == 4 {
					shape = []int64{int64(n), mh, mw}
					maskSize = n * int(mh) * int(mw)
					maskBacking = make([]int64, maskSize)
					for j := range maskBacking {
						maskBacking[j] = 1
					}
					maskTensor, mErr = ort.NewTensor(ort.NewShape(shape...), maskBacking)
				}
				if mErr != nil {
					return mErr
				}
			}
			values[i] = maskTensor
			destroyers = append(destroyers, maskTensor.Destroy)
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
		agg = errors.Join(agg, imgTensor.Destroy())
		for _, d := range destroyers {
			agg = errors.Join(agg, d())
		}
		return agg
	}
	return nil
}

func CreateMessagesORT(batch *PipelineBatch, inputs any, systemPrompt string) error {
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
		// Check if any messages contain images
		hasImages := false
		var allImageURLs []string
		for _, msgList := range inputCast {
			for _, msg := range msgList {
				if len(msg.ImageURLs) > 0 {
					hasImages = true
					allImageURLs = append(allImageURLs, msg.ImageURLs...)
				}
			}
		}
		if hasImages {
			images, err := ortgenai.LoadImages(allImageURLs)
			if err != nil {
				return fmt.Errorf("failed to load images: %w", err)
			}
			batch.Images = images
			batch.DestroyMultimodal = func() error {
				images.Destroy()
				return nil
			}
		}

		ortMessages := make([][]ortgenai.Message, len(inputCast))
		addSystemPrompt := systemPrompt != ""
		systemPromptMessage := ortgenai.Message{Role: "system", Content: systemPrompt}
		for i, inputMessages := range inputCast {
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
				out[offset+j] = ortgenai.Message{Role: message.Role, Content: message.Content}
			}
			ortMessages[i] = out
		}
		batch.InputValues = ortMessages
	default:
		return fmt.Errorf("invalid input type %T for CreateMessagesORT", inputCast)
	}
	return nil
}
