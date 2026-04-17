package backends

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/gomlx/go-huggingface/tokenizers/bucket"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	gomlxcontext "github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/onnx-gomlx/onnx"
	"github.com/gomlx/onnx-gomlx/onnx/parser"
	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/util/fileutil"
	"golang.org/x/sync/errgroup"
)

// Default shape buckets for batch size and sequence length.
// These provide a good balance between padding overhead and JIT cache size.
var (
	defaultBatchBuckets    []int
	defaultSequenceBuckets []int
)

type GoMLXModel struct {
	Backend         backends.Backend
	OnnxModel       onnx.Model
	Ctx             *gomlxcontext.Context // ctx with the model's weights.
	Exec            *gomlxcontext.Exec    // exec is used to execute the model with a context.
	Call            func(ctx *gomlxcontext.Context, inputs []*graph.Node) []*graph.Node
	Destroy         func()
	BatchBuckets    []int // BatchBuckets defines bucket sizes for batch dimension padding.
	SequenceBuckets []int // SequenceBuckets defines bucket sizes for sequence length padding.
	MaxCache        int   // MaxCache sets the maximum number of unique input shapes to cache.
}

func createGoMLXModelBackend(model *Model, options *options.Options) error {
	var modelParsed onnx.Model
	var err error
	if model.OnnxReader != nil {
		modelParsed, err = parser.ParseReader(model.OnnxReader)
	} else {
		modelParsed, err = parser.ParseFile(fileutil.PathJoinSafe(model.Path, model.OnnxPath))
		if err != nil && modelParsed != nil {
			modelParsed = modelParsed.WithBaseDir(model.Path)
		}
	}
	if err != nil {
		return err
	}

	inputs, outputs := loadInputOutputMetaGoMLX(modelParsed)
	var outputNames []string
	for _, v := range outputs {
		outputNames = append(outputNames, v.Name)
	}

	ctx := gomlxcontext.New()
	// Mark it to reuse variables: it will be an error to create a new variable – for safety.
	ctx = ctx.Reuse()

	// Read variables from ONNX model.
	err = modelParsed.VariablesToContext(ctx)
	if err != nil {
		return errors.Join(err, modelParsed.Close())
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
		return errors.Join(backendErr, modelParsed.Close())
	}

	// Create model executor.
	callFunc := func(ctx *gomlxcontext.Context, inputs []*graph.Node) []*graph.Node {
		inputsMap := map[string]*graph.Node{}
		for i, inputMeta := range model.InputsMeta {
			inputsMap[inputMeta.Name] = inputs[i]
		}
		return modelParsed.CallGraph(ctx, inputs[0].Graph(), inputsMap, outputNames...)
	}

	exec, contextErr := gomlxcontext.NewExec(backend, ctx, callFunc)
	if contextErr != nil {
		return errors.Join(contextErr, modelParsed.Close())
	}

	maxCache, batchBuckets, sequenceBuckets := getCacheAndBucketSizes(options, config)
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

func getCacheAndBucketSizes(options *options.Options, backend string) (int, []int, []int) {
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
	}

	// If using simpleGo, and user hasnt specified custom buckets, set max cache to unlimitted and disable bucketing
	if backend == "go" && !bucketsSpecified {
		return -1, []int{}, []int{}
	}

	maxCache := 512
	if bucketsSpecified {
		maxCache = max(len(batchBuckets), 1) * max(len(sequenceBuckets), 1)
	}
	return maxCache, batchBuckets, sequenceBuckets
}

func loadInputOutputMetaGoMLX(model onnx.Model) ([]InputOutputInfo, []InputOutputInfo) {
	var inputs, outputs []InputOutputInfo

	inputNames, inputShapes := model.Inputs()
	for i, name := range inputNames {
		shape := inputShapes[i]
		dimensions := make([]int64, len(shape.Dimensions))
		for j, y := range shape.Dimensions {
			dimensions[j] = int64(y)
		}
		inputs = append(inputs, InputOutputInfo{
			Name:       name,
			Dimensions: dimensions,
		})
	}

	outputNames, outputShapes := model.Outputs()
	for i, name := range outputNames {
		shape := outputShapes[i]
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
	var err error
	batchSize := batch.Size
	if padBatchDimension {
		batchSize, err = shapeBucket(batchSize, model.GoMLXModel.BatchBuckets)
		if err != nil {
			return fmt.Errorf("batch size larger than max bucket, please adjust WithGoMLXBatchBuckets: %w", err)
		}
		batch.PaddedBatchSize = batchSize
	}
	maxSeqLength := batch.MaxSequenceLength
	if padSequenceDimension {
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
		backing := make([]int64, total)
		idx := 0
		switch meta.Name {
		case "input_ids":
			for bi, inp := range batch.Input {
				seqLen := len(inp.TokenIDs)
				maskRow := make([]bool, maxSeqLength)
				for pos := 0; pos < maxSeqLength; pos++ {
					if pos < seqLen {
						backing[idx] = int64(inp.TokenIDs[pos])
						maskRow[pos] = true
					}
					idx++
				}
				paddingMasks[bi] = maskRow
			}
		case "token_type_ids":
			for _, inp := range batch.Input {
				seqLen := len(inp.TokenIDs)
				for pos := range maxSeqLength {
					if pos < seqLen {
						backing[idx] = int64(inp.TypeIDs[pos])
					}
					idx++
				}
			}
		case "attention_mask":
			for _, inp := range batch.Input {
				for pos := range maxSeqLength {
					if pos < len(inp.TokenIDs) {
						backing[idx] = int64(inp.AttentionMask[pos])
					}
					idx++
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

func runGoMLXSessionOnBatch(ctx context.Context, batch *PipelineBatch, p *BasePipeline) error {
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

	var outputTensors []*tensors.Tensor
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
		var execErr error
		outputTensors, execErr = p.Model.GoMLXModel.Exec.Exec(batch.InputValues.([]*tensors.Tensor))
		return errors.Join(execErr, err)
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

	var err error
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
		// When the batch dimension was padded to a bucket (e.g. batch=2 → bucket=8),
		// the flat output contains paddedBatch × paddedSeq elements. Passing paddedBatch
		// rows to ReshapeOutput with batch.Size=2 causes flatDataTo2D to compute the
		// wrong row width (totalElems/2 instead of paddedSeq), mapping row-1 onto
		// padding-batch rows whose NaN logits (from all-zero attention masks) propagate
		// through softmax and make every score comparison with bestScore(-1) false.
		// Strip excess batch rows here so ReshapeOutput only sees batch.Size rows.
		if batch.PaddedBatchSize > batch.Size && len(rawOutput) > 0 {
			paddedSeqLen := len(rawOutput) / batch.PaddedBatchSize
			trimmed := make([]float32, batch.Size*paddedSeqLen)
			for row := 0; row < batch.Size; row++ {
				copy(trimmed[row*paddedSeqLen:], rawOutput[row*paddedSeqLen:(row+1)*paddedSeqLen])
			}
			rawOutput = trimmed
		}
		convertedOutput[i] = ReshapeOutput(rawOutput, p.Model.OutputsMeta[i], batch.Size, batch.PaddingMask, batch.MaxSequenceLength)
	}

	batch.OutputValues = convertedOutput
	return err
}

// shapeBucket quantizes input dimensions to coarse buckets to reduce JIT cache pressure.
//
// XLA compiles a separate program for each unique input shape.
//
// By default we use "two-bit" bucketing, which provides a good balance between
// padding overhead and JIT cache size (approx. 1.41x increase between buckets).
//
//	Two-bit buckets: 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, ...
//
// Trade-off: Slightly more padding waste for small inputs (e.g., batch=5 pads to 6),
// but dramatically fewer compiled programs in memory.
func shapeBucket(n int, buckets []int) (int, error) {
	if len(buckets) == 0 {
		return bucket.TwoBitBucketLen(n), nil
	}
	for _, b := range buckets {
		if n <= b {
			return b, nil
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
					if d > 1 {
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

func createTabularTensorsGoMLX(batch *PipelineBatch, model *Model, features [][]float32) error {
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

	t := tensors.FromFlatDataAndDimensions(backing, batch.Size, featDim)
	inputTensors := []*tensors.Tensor{t}
	batch.InputValues = inputTensors
	batch.DestroyInputs = func() error {
		return t.FinalizeAll()
	}
	// No padding mask for tabular
	batch.PaddingMask = nil
	batch.MaxSequenceLength = 0
	return nil
}
