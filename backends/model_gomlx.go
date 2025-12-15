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

type GoMLXModel struct {
	Backend   backends.Backend
	OnnxModel *onnx.Model
	Ctx       *context.Context // ctx with the model's weights.
	Exec      *context.Exec    // exec is used to execute the model with a context.
	Call      func(ctx *context.Context, inputs []*graph.Node) []*graph.Node
	Destroy   func()
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

func createGoMLXModelBackend(model *Model, options *options.Options) error {
	modelParsed, err := onnx.Parse(model.OnnxBytes)
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

	exec.SetMaxCache(-1)

	model.GoMLXModel = &GoMLXModel{
		Backend:   backend,
		OnnxModel: modelParsed,
		Ctx:       ctx,
		Exec:      exec,
		Call:      callFunc,
		Destroy: func() {
			exec.Finalize()
			ctx.Finalize()
			backend.Finalize()
		},
	}
	model.InputsMeta = inputs
	model.OutputsMeta = outputs

	model.OnnxBytes = nil
	return err
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

	batchSize := batch.Size
	if padBatchDimension {
		batchSize = nextPowerOf2(batchSize)
	}
	maxSeqLength := batch.MaxSequenceLength
	if padSequenceDimension && !leftPad {
		maxSeqLength = nextPowerOf2(maxSeqLength)
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

func nextPowerOf2(n int) int {
	if n < 1 {
		return 1
	}

	// Check if n is a power of 2.
	if (n & (n - 1)) == 0 {
		return n
	}

	// Find the next power of 2.
	// This approach initially sets the result to 1 and
	// keeps shifting the bits until it finds a number greater or equal to n.
	pow := 1
	for pow < n {
		pow <<= 1
	}
	return pow
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
