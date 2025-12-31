//go:build cgo && (ORT || ALL)

package backends

import (
	"context"
	"errors"
	"fmt"
	"os"
	"runtime"
	"strconv"
	"strings"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/util/fileutil"
	"github.com/knights-analytics/ortgenai"
)

type ORTModel struct {
	Session           *ort.DynamicAdvancedSession
	GenerativeSession *ortgenai.Session
	SessionOptions    *ort.SessionOptions
	Options           *options.OrtOptions
	Destroy           func() error
}

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
		providers = append(providers, "CoreML")
		providerOptions["CoreML"] = ortOptions.CoreMLOptions
	}

	// DirectML
	if ortOptions.DirectMLOptions != nil {
		providers = append(providers, "DML")
		// Map device id to string map expected by advanced session
		providerOptions["DML"] = map[string]string{
			"device_id": strconv.Itoa(*ortOptions.DirectMLOptions),
		}
	}

	// OpenVINO
	if ortOptions.OpenVINOOptions != nil {
		providers = append(providers, "OpenVINO")
		providerOptions["OpenVINO"] = ortOptions.OpenVINOOptions
	}

	// TensorRT
	if ortOptions.TensorRTOptions != nil {
		providers = append(providers, "NvTensorRtRtx")
		providerOptions["NvTensorRtRtx"] = ortOptions.TensorRTOptions
	}
	return providers, providerOptions, nil
}

func createORTGenerativeSession(model *Model, options *options.Options) error {
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
		exists, err := fileutil.FileExists(libraryPath)
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
	providers, providerOptions, err := mapORTOptions(options)
	if err != nil {
		return fmt.Errorf("error mapping ORT options for generative session: %w", err)
	}
	ortGenAiSession, err := ortgenai.CreateGenerativeSessionAdvanced(model.Path, providers, providerOptions)
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
	return nil
}

func runGenerativeORTSessionOnBatch(ctx context.Context, batch *PipelineBatch, p *BasePipeline, maxLength int) (chan SequenceDelta, chan error, error) {
	session := p.Model.ORTModel.GenerativeSession
	if session == nil {
		return nil, nil, errors.New("ORT generative session is not initialized")
	}

	inputs, ok := batch.InputValues.([][]ortgenai.Message)
	if !ok {
		return nil, nil, fmt.Errorf("invalid input type %T for generative ORT session", batch.InputValues)
	}

	// Check if we have multimodal tensors to use instead of text tokenization
	var ortTokenStream <-chan ortgenai.SequenceDelta
	var ortErrorStream <-chan error
	var err error

	if batch.MultimodalTensors != nil {
		// Multimodal path: use named tensors
		namedTensors, ok := batch.MultimodalTensors.(*ortgenai.NamedTensors)
		if !ok {
			return nil, nil, fmt.Errorf("invalid multimodal tensors type %T", batch.MultimodalTensors)
		}
		ortTokenStream, ortErrorStream, err = session.GenerateWithTensors(ctx, namedTensors, &ortgenai.GenerationOptions{MaxLength: maxLength, BatchSize: len(inputs)})
		if err != nil {
			return nil, nil, fmt.Errorf("error during multimodal generation start: %w", err)
		}
	} else {
		// Text-only path: use messages
		ortTokenStream, ortErrorStream, err = session.Generate(ctx, inputs, &ortgenai.GenerationOptions{MaxLength: maxLength})
		if err != nil {
			return nil, nil, fmt.Errorf("error during generation start: %w", err)
		}
	}

	tokenStream := make(chan SequenceDelta, 10)
	errorStream := make(chan error, 1)

	go func() {
		defer close(tokenStream)
		defer close(errorStream)

		for {
			// If both upstream channels are closed, finish and close our outputs.
			if ortTokenStream == nil && ortErrorStream == nil {
				return
			}
			select {
			case <-ctx.Done():
				return
			case err, ok := <-ortErrorStream:
				if !ok {
					ortErrorStream = nil
					continue
				}
				select {
				case errorStream <- fmt.Errorf("error during generation: %w", err):
				default:
				}
			case tokenDelta, ok := <-ortTokenStream:
				if !ok {
					ortTokenStream = nil
					continue
				}
				tokenStream <- SequenceDelta{
					Token: tokenDelta.Tokens, // TODO rename to token in ortgenai?
					Index: tokenDelta.Sequence,
				}
			}
		}
	}()
	return tokenStream, errorStream, nil
}

func createORTModelBackend(model *Model, loadAsBytes bool, options *options.Options) error {

	sessionOptions := options.BackendOptions.(*ort.SessionOptions)

	var inputs, outputs []InputOutputInfo
	var cwd string
	var err error
	if loadAsBytes {
		inputs, outputs, err = loadInputOutputMetaORTBytes(model.OnnxBytes)
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
	if loadAsBytes {
		session, err = ort.NewDynamicAdvancedSessionWithONNXData(
			model.OnnxBytes,
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

func loadInputOutputMetaORTBytes(onnxBytes []byte) ([]InputOutputInfo, []InputOutputInfo, error) {
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
	padLeft := len(model.EosTokenIDs) > 0
	batchSize := batch.Size
	maxSequenceLength := batch.MaxSequenceLength
	total := batchSize * maxSequenceLength

	// 1) prepare result containers - now we use all inputs, not filtering
	inputVals := make([]ort.Value, len(model.InputsMeta))
	masks := make([][]bool, batchSize)

	// 2) build each tensor
	for mi, meta := range model.InputsMeta {
		switch {
		case strings.HasPrefix(meta.Name, "past_key"):
			// Create key cache tensor
			cacheTensor, err := createSingleCacheTensorORT(
				batchSize,
				model.NumKeyValueHeads,
				model.FixedCacheSize,
				model.HeadDim,
			)
			if err != nil {
				return err
			}
			inputVals[mi] = cacheTensor

		default:
			// Handle regular input tensors
			backing := make([]int64, total)
			idx := 0
			switch meta.Name {
			case "input_ids":
				for bi, inp := range batch.Input {
					seqLen := len(inp.TokenIDs)
					padLen := maxSequenceLength - seqLen
					maskRow := make([]bool, maxSequenceLength)
					for pos := range maxSequenceLength {
						if padLeft {
							if pos < padLen {
								backing[idx] = model.PadToken
							} else {
								backing[idx] = int64(inp.TokenIDs[pos-padLen])
								maskRow[pos] = true
							}
						} else {
							if pos < seqLen {
								backing[idx] = int64(inp.TokenIDs[pos])
								maskRow[pos] = true
							}
						}
						idx++
					}
					masks[bi] = maskRow
				}
			case "token_type_ids":
				for _, inp := range batch.Input {
					seqLen := len(inp.TokenIDs)
					for pos := range maxSequenceLength {
						// always right-pad
						if pos < seqLen {
							backing[idx] = int64(inp.TypeIDs[pos])
						}
						idx++
					}
				}
			case "attention_mask":
				if model.IsGenerative {
					for range batch.Input {
						for range maxSequenceLength {
							backing[idx] = 1
							idx++
						}
					}
				} else {
					// For non-generative models, take the input mask from the tokenizer output
					for _, inp := range batch.Input {
						for pos := range maxSequenceLength {
							if pos < len(inp.TokenIDs) {
								backing[idx] = int64(inp.AttentionMask[pos])
							}
							idx++
						}
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

// createSingleCacheTensorORT creates a single cache tensor.
func createSingleCacheTensorORT(batchSize, numKeyValueHeads, maxSeqLen, headDim int) (ort.Value, error) {
	tensorSize := batchSize * numKeyValueHeads * maxSeqLen * headDim
	slice := make([]float32, tensorSize)
	return ort.NewTensor(
		ort.NewShape(int64(batchSize), int64(numKeyValueHeads), int64(maxSeqLen), int64(headDim)),
		slice,
	)
}

func runORTSessionOnBatch(batch *PipelineBatch, p *BasePipeline) error {
	var err error

	outputTensors := make([]ort.Value, len(p.Model.OutputsMeta))
	errOnnx := p.Model.ORTModel.Session.Run(batch.InputValues.([]ort.Value), outputTensors)
	if errOnnx != nil {
		return errOnnx
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
	return err
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
	switch inputs := inputs.(type) {
	case []string:
		ortMessages := make([][]ortgenai.Message, len(inputs))
		addSystemPrompt := systemPrompt != ""
		systemPrompt := ortgenai.Message{Role: "system", Content: systemPrompt}
		for i, input := range inputs {
			if addSystemPrompt {
				m := make([]ortgenai.Message, 2)
				m[0] = systemPrompt
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
		for _, msgList := range inputs {
			for _, msg := range msgList {
				if len(msg.ImageURLs) > 0 {
					hasImages = true
					allImageURLs = append(allImageURLs, msg.ImageURLs...)
				}
			}
		}

		ortMessages := make([][]ortgenai.Message, len(inputs))
		addSystemPrompt := systemPrompt != ""
		systemPrompt := ortgenai.Message{Role: "system", Content: systemPrompt}
		for i, inputMessages := range inputs {
			additionalLength := 0
			if addSystemPrompt {
				additionalLength = 1
			}
			out := make([]ortgenai.Message, len(inputMessages)+additionalLength)
			offset := 0
			if addSystemPrompt {
				out[0] = systemPrompt
				offset = 1
			}
			for j, message := range inputMessages {
				out[offset+j] = ortgenai.Message{Role: message.Role, Content: message.Content}
			}
			ortMessages[i] = out
		}
		batch.InputValues = ortMessages

		// If multimodal, process images
		if hasImages {
			// Load images
			images, err := ortgenai.LoadImages(allImageURLs)
			if err != nil {
				return fmt.Errorf("failed to load images: %w", err)
			}

			// Create multimodal processor
			processor, err := ortgenai.CreateMultiModalProcessor(model.ORTModel.GenerativeSession.GetModel())
			if err != nil {
				images.Destroy()
				return fmt.Errorf("failed to create multimodal processor: %w", err)
			}

			// Combine all text prompts (simplified - using first message content)
			// In a real implementation, you might want to format this differently
			prompt := ""
			if len(inputs) > 0 && len(inputs[0]) > 0 {
				prompt = inputs[0][0].Content
			}

			// Process images with prompt
			tensors, err := processor.ProcessImages(prompt, images)
			if err != nil {
				images.Destroy()
				processor.Destroy()
				return fmt.Errorf("failed to process images: %w", err)
			}

			// Store tensors in batch
			batch.MultimodalTensors = tensors
			batch.DestroyMultimodal = func() error {
				tensors.Destroy()
				processor.Destroy()
				images.Destroy()
				return nil
			}
		}
	default:
		return fmt.Errorf("invalid input type %T for CreateMessagesORT", inputs)
	}
	return nil
}
