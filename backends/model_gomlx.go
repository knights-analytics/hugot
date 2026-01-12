package backends

import (
	"errors"
	"fmt"
	"io"
	"strconv"
	"strings"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/onnx-gomlx/onnx"

	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/util/fileutil"
)

// Default shape buckets for batch size and sequence length.
// These provide a good balance between padding overhead and JIT cache size.
var (
	defaultBatchBuckets    = []int{1, 8, 32}
	defaultSequenceBuckets = []int{32, 128, 512}
)

type GoMLXModel struct {
	Backend         backends.Backend
	OnnxModel       *onnx.Model
	Ctx             *context.Context // ctx with the model's weights.
	Exec            *context.Exec    // exec is used to execute the model with a context.
	Call            func(ctx *context.Context, inputs []*graph.Node) []*graph.Node
	Destroy         func()
	BatchBuckets    []int // BatchBuckets defines bucket sizes for batch dimension padding.
	SequenceBuckets []int // SequenceBuckets defines bucket sizes for sequence length padding.
	MaxCache        int   // MaxCache sets the maximum number of unique input shapes to cache.
}

func loadExternalData(path string, model *onnx.Model) error {
	externalMap := map[string][]byte{}
	// load external data from same dir as the base model ONNX file
	for _, proto := range model.Proto.Graph.Initializer {
		// proto.Datalocation is 1 if data is external, 0 otherwise
		if proto.DataLocation == 1 {
			externalPath := ""
			offset := int64(0)
			length := int64(-1)

			for _, entry := range proto.ExternalData {
				switch entry.Key {
				case "location":
					externalPath = entry.Value
				case "offset":
					parsedOffset, err := strconv.ParseInt(entry.Value, 10, 64)
					if err != nil {
						return fmt.Errorf("parsing offset failed with err %w", err)
					}
					offset = parsedOffset
				case "length":
					parsedLength, err := strconv.ParseInt(entry.Value, 10, 64)
					if err != nil {
						return fmt.Errorf("parsing length failed with err %w", err)
					}
					length = parsedLength
				}
			}

			weightsPath := fileutil.PathJoinSafe(path, externalPath)

			if _, ok := externalMap[externalPath]; !ok {
				bytes, err := fileutil.ReadFileBytes(weightsPath)
				if err != nil {
					return err
				}
				externalMap[externalPath] = bytes
			}

			fullBytes := externalMap[externalPath]
			end := int64(len(fullBytes))
			if length >= 0 && offset+length <= int64(len(fullBytes)) {
				end = offset + length
			}
			subsetBytes := fullBytes[offset:end]
			proto.RawData = subsetBytes
		}
	}
	return nil
}

func createGoMLXModelBackend(model *Model, loadAsBytes bool, options *options.Options) error {
	var modelParsed *onnx.Model
	var err error
	if loadAsBytes {
		modelParsed, err = onnx.Parse(model.OnnxBytes)
	} else {
		modelParsed, err = onnx.ReadFile(fileutil.PathJoinSafe(model.Path, model.OnnxPath))
	}
	if err != nil {
		return err
	}

	inputs, outputs := loadInputOutputMetaGoMLX(modelParsed)
	var outputNames []string
	for _, v := range outputs {
		outputNames = append(outputNames, v.Name)
	}

	if err = loadExternalData(model.Path, modelParsed); err != nil {
		return err
	}

	ctx := context.New()
	// Mark it to reuse variables: it will be an error to create a new variable – for safety.
	ctx = ctx.Reuse()

	// Read variables from ONNX model.
	err = modelParsed.VariablesToContext(ctx)
	if err != nil {
		return err
	}

	config := "go"
	if options.GoMLXOptions.TPU {
		config = "xla:tpu"
	} else if options.GoMLXOptions.Cuda {
		config = "xla:cuda"
	} else if options.GoMLXOptions.XLA {
		config = "xla:cpu"
	}

	backend, backendErr := backends.NewWithConfig(config)
	if backendErr != nil {
		return backendErr
	}

	// Create model executor.
	callFunc := func(ctx *context.Context, inputs []*graph.Node) []*graph.Node {
		inputsMap := map[string]*graph.Node{}
		for i, inputMeta := range model.InputsMeta {
			inputsMap[inputMeta.Name] = inputs[i]
		}
		return modelParsed.CallGraph(ctx, inputs[0].Graph(), inputsMap, outputNames...)
	}

	exec, contextErr := context.NewExec(backend, ctx, callFunc)
	if contextErr != nil {
		return contextErr
	}

	maxCache, batchBuckets, sequenceBuckets := getCacheAndBucketSizes(options, model, config)
	exec.SetMaxCache(maxCache)

	model.GoMLXModel = &GoMLXModel{
		Backend:         backend,
		OnnxModel:       modelParsed,
		Ctx:             ctx,
		Exec:            exec,
		Call:            callFunc,
		MaxCache:        maxCache,
		BatchBuckets:    batchBuckets,
		SequenceBuckets: sequenceBuckets,
		Destroy: func() {
			exec.Finalize()
			ctx.Finalize()
			backend.Finalize()
		},
	}
	model.InputsMeta = inputs
	model.OutputsMeta = outputs

	return err
}

func getCacheAndBucketSizes(options *options.Options, model *Model, backend string) (int, []int, []int) {
	bucketsSpecified := false
	// Use configured buckets or fall back to defaults.
	batchBuckets := defaultBatchBuckets
	if len(options.GoMLXOptions.BatchBuckets) > 0 {
		batchBuckets = options.GoMLXOptions.BatchBuckets
		bucketsSpecified = true
	}
	var sequenceBuckets []int
	if len(options.GoMLXOptions.SequenceBuckets) > 0 {
		sequenceBuckets = options.GoMLXOptions.SequenceBuckets
		bucketsSpecified = true
	} else {
		sequenceBuckets = defaultSequenceBuckets
		// Ensure that sequence buckets cover the max sequence length.
		if sequenceBuckets[len(sequenceBuckets)-1] < model.MaxPositionEmbeddings {
			sequenceBuckets = append(sequenceBuckets, model.MaxPositionEmbeddings)
		}
	}

	// If using simpleGo, and user hasnt specified custom buckets, set max cache to unlimitted and disable bucketing
	if backend == "go" && !bucketsSpecified {
		return -1, []int{}, []int{}
	}

	maxCache := len(batchBuckets) * len(sequenceBuckets)
	return maxCache, batchBuckets, sequenceBuckets
}

func loadInputOutputMetaGoMLX(model *onnx.Model) ([]InputOutputInfo, []InputOutputInfo) {
	var inputs, outputs []InputOutputInfo

	for i, name := range model.InputsNames {
		shape := model.InputsShapes[i]
		dimensions := make([]int64, len(shape.Dimensions))
		for j, y := range shape.Dimensions {
			dimensions[j] = int64(y)
		}
		inputs = append(inputs, InputOutputInfo{
			Name:       name,
			Dimensions: dimensions,
		})
	}
	for i, name := range model.OutputsNames {
		shape := model.OutputsShapes[i]
		dimensions := make([]int64, len(shape.Dimensions))
		for j, y := range shape.Dimensions {
			dimensions[j] = int64(y)
		}
		outputs = append(outputs, InputOutputInfo{
			Name:       name,
			Dimensions: dimensions,
		})
	}
	return inputs, outputs
}

func createInputTensorsGoMLX(batch *PipelineBatch, model *Model, padBatchDimension bool, padSequenceDimension bool) error {
	leftPad := len(model.EosTokenIDs) > 0

	// TODO: replace this once dynamic input shapes fixed
	model.FixedCacheSize = 150

	var err error
	batchSize := batch.Size
	if padBatchDimension {
		batchSize, err = shapeBucket(batchSize, model.GoMLXModel.BatchBuckets)
		if err != nil {
			return fmt.Errorf("batch size larger than max bucket, please adjust WithGoMLXBatchBuckets: %w", err)
		}
	}
	maxSeqLength := batch.MaxSequenceLength
	if padSequenceDimension && !leftPad {
		maxSeqLength, err = shapeBucket(maxSeqLength, model.GoMLXModel.SequenceBuckets)
		if err != nil {
			return fmt.Errorf("sequence length larger than max bucket, please adjust WithGoMLXSequenceBuckets: %w", err)
		}
	}
	total := batchSize * maxSeqLength

	// 1) prepare result containers - now we use all inputs, not filtering
	inputTensors := make([]*tensors.Tensor, len(model.InputsMeta))
	paddingMasks := make([][]bool, batch.Size)

	// 2) build each tensor
	for mi, meta := range model.InputsMeta {
		switch {
		case strings.HasPrefix(meta.Name, "past_key"):
			// Create key cache tensor
			cacheTensor := createSingleCacheTensorGoMLX(
				batchSize,
				model.NumKeyValueHeads,
				model.FixedCacheSize,
				model.HeadDim,
			)
			inputTensors[mi] = cacheTensor

		case strings.HasPrefix(meta.Name, "past_value"):
			// Create value cache tensor
			cacheTensor := createSingleCacheTensorGoMLX(
				batchSize,
				model.NumKeyValueHeads,
				model.FixedCacheSize,
				model.HeadDim,
			)
			inputTensors[mi] = cacheTensor

		default:
			// Handle regular input tensors
			backing := make([]int64, total)
			idx := 0
			switch meta.Name {
			case "input_ids":
				for bi, inp := range batch.Input {
					seqLen := len(inp.TokenIDs)
					padLen := maxSeqLength - seqLen
					maskRow := make([]bool, maxSeqLength)
					for pos := 0; pos < maxSeqLength; pos++ {
						if leftPad {
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
					paddingMasks[bi] = maskRow
				}
			case "token_type_ids":
				for _, inp := range batch.Input {
					seqLen := len(inp.TokenIDs)
					for pos := range maxSeqLength {
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
						for range maxSeqLength {
							backing[idx] = 1
							idx++
						}
					}
				} else {
					// For non-generative models, take the input mask from the tokenizer output
					for _, inp := range batch.Input {
						for pos := range maxSeqLength {
							if pos < len(inp.TokenIDs) {
								backing[idx] = int64(inp.AttentionMask[pos])
							}
							idx++
						}
					}
				}

			case "position_ids":
				for range batch.Input {
					for pos := range maxSeqLength {
						backing[idx] = int64(pos + 1)
						idx++
					}
				}
			default:
				return fmt.Errorf("unknown input meta name %s", meta.Name)
			}
			inputTensors[mi] = tensors.FromFlatDataAndDimensions(backing, batchSize, maxSeqLength)
		}
	}

	// 3) assign and prepare cleanup
	batch.InputValues = inputTensors
	batch.PaddingMask = paddingMasks
	batch.DestroyInputs = func() error {
		var err error
		for _, t := range inputTensors {
			tensorErr := t.FinalizeAll()
			if err == nil {
				err = tensorErr
			}
		}
		return err
	}
	return nil
}

// createSingleCacheTensorGoMLX creates a single cache tensor (either key or value).
func createSingleCacheTensorGoMLX(batchSize, numKeyValueHeads, maxSeqLen, headDim int) *tensors.Tensor {
	return tensors.FromScalarAndDimensions(
		float32(0), batchSize, numKeyValueHeads, maxSeqLen, headDim)
}

func runGoMLXSessionOnBatch(batch *PipelineBatch, p *BasePipeline) error {
	// Non-generative models
	outputTensors, err := p.Model.GoMLXModel.Exec.Exec(batch.InputValues.([]*tensors.Tensor))
	if err != nil {
		return err
	}

	defer func() {
		for _, t := range outputTensors {
			err = errors.Join(t.FinalizeAll())
		}
	}()

	// Transfer output tensors from device (TPU/GPU) to local (CPU) memory immediately.
	//
	// Go's GC doesn't see the large device-side allocations, so it doesn't feel pressure
	// to reclaim tensor wrappers. By explicitly transferring to local memory and
	// invalidating device copies, we release TPU/GPU memory as soon as the computation
	// completes rather than waiting for eventual GC.
	for _, t := range outputTensors {
		t.MaterializeLocal()         // Copy data from device to local memory
		err = t.InvalidateOnDevice() // Free device memory immediately
		if err != nil {
			return err
		}
	}

	convertedOutput := make([]any, len(outputTensors))
	for i, t := range outputTensors {
		var rawOutput []float32
		err = tensors.ConstFlatData(t, func(flat []float32) {
			rawOutput = flat
		})
		if err != nil {
			return err
		}
		convertedOutput[i] = ReshapeOutput(rawOutput, p.Model.OutputsMeta[i], batch.Size, batch.PaddingMask, batch.MaxSequenceLength)
	}

	batch.OutputValues = convertedOutput
	return err
}

// shapeBucket quantizes input dimensions to coarse buckets to reduce JIT cache pressure.
//
// XLA compiles a separate program for each unique input shape. Using power-of-2 padding
// creates up to 42 unique shapes per model (6 batch sizes × 7 sequence lengths), which
// can consume 2-8GB of memory for cached compiled programs.
//
// Coarse bucketing reduces this to just 9 shapes (3 batch × 3 sequence), significantly
// reducing memory usage while maintaining reasonable padding overhead:
//
//	Batch buckets:    1, 8, 32      (covers 1-32 batch sizes)
//	Sequence buckets: 32, 128, 512  (covers 1-512 token sequences)
//
// Trade-off: Slightly more padding waste for small inputs (e.g., batch=2 pads to 8),
// but dramatically fewer compiled programs in memory.
//
// Example memory savings with 2 models (embedder + reranker):
//   - Power-of-2: 42 shapes × 2 models × 100MB avg = 8.4GB
//   - Coarse buckets: 9 shapes × 2 models × 100MB avg = 1.8GB
func shapeBucket(n int, buckets []int) (int, error) {
	for _, bucket := range buckets {
		if n <= bucket {
			return bucket, nil
		}
	}
	return 0, fmt.Errorf("input shape %d exceeds maximum bucket size %v", n, buckets)
}

func (goMLXModel *GoMLXModel) Save(w io.Writer) error {
	if err := goMLXModel.OnnxModel.ContextToONNX(goMLXModel.Ctx); err != nil {
		return err
	}
	err := goMLXModel.OnnxModel.Write(w)
	if err != nil {
		return err
	}
	return nil
}

func createImageTensorsGoXLA(batch *PipelineBatch, model *Model, preprocessed [][][][]float32) error {
	if len(preprocessed) == 0 {
		return errors.New("no preprocessed images provided")
	}
	n, c, h, w := len(preprocessed), len(preprocessed[0]), len(preprocessed[0][0]), len(preprocessed[0][0][0])
	backing := make([]float32, n*c*h*w)
	idx := 0
	for i := range n {
		for ch := range c {
			for y := range h {
				for x := range w {
					backing[idx] = preprocessed[i][ch][y][x]
					idx++
				}
			}
		}
	}
	inputTensors := []*tensors.Tensor{tensors.FromFlatDataAndDimensions(backing, n, c, h, w)}

	// Optionally add pixel_mask as ones if required by model.
	if len(model.InputsMeta) > 1 {
		for _, meta := range model.InputsMeta {
			lower := strings.ToLower(meta.Name)
			if strings.Contains(lower, "mask") {
				// Infer mask dims
				var mh, mw int
				for _, d := range meta.Dimensions {
					if d > 1 && mh == 0 {
						mh = int(d)
						continue
					}
					if d > 1 && mh != 0 && mw == 0 {
						mw = int(d)
						break
					}
				}
				if mh == 0 || mw == 0 {
					mh, mw = h, w
				}
				// Default to 3D [n,H,W]; if meta has 4 dims, use [n,1,H,W]
				var maskTensor *tensors.Tensor
				if len(meta.Dimensions) == 4 {
					maskBacking := make([]int64, n*1*mh*mw)
					for i := range maskBacking {
						maskBacking[i] = 1
					}
					maskTensor = tensors.FromFlatDataAndDimensions(maskBacking, n, 1, mh, mw)
				} else {
					maskBacking := make([]int64, n*mh*mw)
					for i := range maskBacking {
						maskBacking[i] = 1
					}
					maskTensor = tensors.FromFlatDataAndDimensions(maskBacking, n, mh, mw)
				}
				inputTensors = append(inputTensors, maskTensor)
				break
			}
		}
	}
	batch.InputValues = inputTensors
	batch.DestroyInputs = func() error {
		var err error
		for _, input := range inputTensors {
			tensorErr := input.FinalizeAll()
			if tensorErr != nil {
				err = tensorErr
			}
		}
		return err
	}
	return nil
}
