package pipelineBackends

import (
	"errors"
	"fmt"
	"io"
	"strconv"
	"strings"

	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/graph"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/onnx-gomlx/onnx"

	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/util"

	_ "github.com/gomlx/gomlx/backends/simplego"
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
	lastSlashIndex := strings.LastIndex(path, "/")
	if lastSlashIndex == -1 {
		// No slash found
		return fmt.Errorf("invalid path for external data: %s", path)
	} else {
		path = path[:lastSlashIndex]
	}

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

			weightsPath := util.PathJoinSafe(path, externalPath)

			if _, ok := externalMap[externalPath]; !ok {
				bytes, err := util.ReadFileBytes(weightsPath)
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

func createGenerativeCallFunc(model *Model, modelParsed *onnx.Model, outputNames []string) func(ctx *context.Context, inputs []*graph.Node) []*graph.Node {
	return func(ctx *context.Context, inputs []*graph.Node) []*graph.Node {
		inputsMap := map[string]*graph.Node{
			"input_ids":    inputs[0],
			"position_ids": inputs[1]}

		for i := range model.NumHiddenLayers {
			key := fmt.Sprintf("past_key_values.%d.key", i)
			value := fmt.Sprintf("past_key_values.%d.value", i)

			inputsMap[key] = inputs[2*i+2]
			inputsMap[value] = inputs[2*i+3]
		}

		g := inputs[0].Graph()
		modelOutputs := modelParsed.CallGraph(ctx, g, inputsMap, outputNames...)
		logits := modelOutputs[0]
		kvCache := modelOutputs[1:]
		shape := logits.Shape().Dimensions
		vocabSize := shape[2]
		seqLen := shape[1]
		lastIdx := Scalar(g, dtypes.Int32, seqLen-1)
		logitsLast := DynamicSlice(
			logits,
			[]*Node{ScalarZero(g, dtypes.Int32), lastIdx, ScalarZero(g, dtypes.Int32)},
			[]int{shape[0], 1, vocabSize},
		)

		batchSize := shape[0]
		logitsLast = Reshape(logitsLast, batchSize, vocabSize)
		nextPredictedToken := ArgMax(logitsLast, 1, dtypes.Int64)
		nextPredictedToken = Reshape(nextPredictedToken, batchSize, 1)

		var terminate *graph.Node
		for i, eosID := range model.EosTokenIDs {
			eosNode := Squeeze(Equal(nextPredictedToken, Scalar(g, dtypes.Int64, eosID)))
			if i == 0 {
				terminate = eosNode
			} else {
				terminate = Or(terminate, eosNode)
			}
		}
		inputIDs := nextPredictedToken
		posShape := inputs[1].Shape().Dimensions
		posLen := posShape[1]
		lastIdxNode := Scalar(g, dtypes.Int32, posLen-1)
		prevPosLast := DynamicSlice(
			inputs[1],
			[]*Node{ScalarZero(g, dtypes.Int32), lastIdxNode},
			[]int{batchSize, 1},
		)
		positionIDs := OnePlus(prevPosLast)
		outputs := append(append([]*Node{inputIDs, positionIDs}, kvCache...), terminate)
		return outputs
	}
}

func createGoMLXModelBackend(model *Model, options *options.Options) error {
	var insideError, recoverErr error

	// we never want to panic so the calling program has a chance to shut down gracefully on error.
	// we therefore catch all panics from goMLX as errors.
	recoverErr = exceptions.TryCatch[error](func() {
		var modelParsed *onnx.Model
		modelParsed, insideError = onnx.Parse(model.OnnxBytes)
		if insideError != nil {
			return
		}

		inputs, outputs := loadInputOutputMetaGoMLX(modelParsed)
		var outputNames []string
		for _, v := range outputs {
			outputNames = append(outputNames, v.Name)
		}

		if insideError = loadExternalData(model.OnnxFilePath, modelParsed); insideError != nil {
			return
		}

		ctx := context.New()
		// Mark it to reuse variables: it will be an error to create a new variable – for safety.
		ctx = ctx.Reuse()

		// Read variables from ONNX model.
		insideError = modelParsed.VariablesToContext(ctx)
		if insideError != nil {
			return
		}

		config := "go"
		if options.GoMLXOptions.Cuda {
			config = "xla:cuda"
		} else if options.GoMLXOptions.XLA {
			config = "xla:cpu"
		}

		var backend backends.Backend
		backend, insideError = backends.NewWithConfig(config)
		if insideError != nil {
			return
		}

		// Create model executor.
		callFunc := func(ctx *context.Context, inputs []*graph.Node) []*graph.Node {
			inputsMap := map[string]*graph.Node{
				"input_ids":      inputs[0],
				"attention_mask": inputs[1]}
			if modelParsed.NumInputs() == 3 {
				inputsMap["token_type_ids"] = inputs[2]
			}
			return modelParsed.CallGraph(ctx, inputs[0].Graph(), inputsMap, outputNames...)
		}

		if len(model.EosTokenIDs) > 0 {
			callFunc = createGenerativeCallFunc(model, modelParsed, outputNames)
		}

		exec := context.NewExec(backend, ctx, callFunc)
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
	})
	return errors.Join(insideError, recoverErr)
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

func createInputTensorsGoMLX(batch *PipelineBatch, inputsMeta []InputOutputInfo, padBatchDimension bool) error {
	batchSize := len(batch.Input)
	batchSizePadded := batchSize
	if padBatchDimension {
		batchSizePadded = nextPowerOf2(batchSize)
	}
	maxSequenceLengthPadded := nextPowerOf2(batch.MaxSequenceLength)
	tensorSize := batchSizePadded * maxSequenceLengthPadded

	inputTensors := make([]*tensors.Tensor, 0, len(inputsMeta))
	paddingMasks := make([][]bool, batchSize)
	for _, inputMeta := range inputsMeta {
		backingSlice := make([]int64, tensorSize)
		counter := 0
		if strings.Contains(inputMeta.Name, "past_key_values") {
			continue
		}
		for j, input := range batch.Input {
			paddingMask := make([]bool, maxSequenceLengthPadded)
			length := len(input.TokenIDs)
			for k := range maxSequenceLengthPadded {
				if k+1 <= length {
					switch inputMeta.Name {
					case "input_ids":
						backingSlice[counter] = int64(input.TokenIDs[k])
						paddingMask[k] = true
					case "token_type_ids":
						backingSlice[counter] = int64(input.TypeIDs[k])
					case "attention_mask":
						backingSlice[counter] = int64(input.AttentionMask[k])
					case "position_ids":
						backingSlice[counter] = int64(k + 1)
					default:
						if strings.Contains(inputMeta.Name, "past_key_values") {
							continue
						} else {
							return fmt.Errorf("unknown input meta name %s", inputMeta.Name)
						}
					}
				} else {
					if inputMeta.Name == "position_ids" {
						backingSlice[counter] = int64(k + 1)
					}
					backingSlice[counter] = 0 // pad with zero
				}
				counter++
			}

			if inputMeta.Name == "input_ids" {
				paddingMasks[j] = paddingMask
			}
		}

		inputTensors = append(inputTensors, tensors.FromFlatDataAndDimensions(backingSlice, batchSizePadded, maxSequenceLengthPadded))
	}
	batch.InputValues = inputTensors
	batch.PaddingMask = paddingMasks
	batch.DestroyInputs = func() error {
		for _, input := range inputTensors {
			input.FinalizeAll()
		}
		return nil
	}
	return nil
}

func CreateGenerativeInputTensorsGoMLX(batch *PipelineBatch) error {
	for _, i := range batch.Input {
		batch.MaxSequenceLength = max(batch.MaxSequenceLength, len(i.TokenIDs))
	}
	batchSize := len(batch.Input)
	maxSeqLength := batch.MaxSequenceLength
	inputIDs := make([][]int64, batchSize)
	positionIDs := make([][]int64, batchSize)
	currentPositions := make([]int64, batchSize)
	for i := range batchSize {
		seq := batch.Input[i].TokenIDs
		seqLen := len(seq)
		padLen := maxSeqLength - seqLen
		paddedInput := make([]int64, maxSeqLength)
		for j := range seqLen {
			paddedInput[padLen+j] = int64(seq[j])
		}
		inputIDs[i] = paddedInput
		posIDs := make([]int64, maxSeqLength)
		for j := 0; j < maxSeqLength; j++ {
			posIDs[j] = int64(j + 1)
		}
		positionIDs[i] = posIDs
		currentPositions[i] = int64(maxSeqLength)
	}
	inputIDsTensor := tensors.FromAnyValue(inputIDs)
	positionIDsTensor := tensors.FromAnyValue(positionIDs)
	modelInputs := []*tensors.Tensor{inputIDsTensor, positionIDsTensor}
	batch.InputValues = modelInputs
	return nil
}

func RunGenerativeGoMLXSessionOnBatch(batch *PipelineBatch, p *BasePipeline) error {
	// Generative models
	batchSize := len(batch.Input)
	finished := make([]bool, batchSize)
	generatedTokens := make([][]int64, batchSize)

	var outputTensors []*tensors.Tensor

	for step := 0; step < batch.MaxNewTokens; step++ {
		outputTensors = p.Model.GoMLXModel.Exec.Call(batch.InputValues.([]*tensors.Tensor))

		genTensor := outputTensors[0]
		var flatTokens []int64
		tensors.ConstFlatData(genTensor, func(flat []int64) {
			flatTokens = flat
		})
		for i := range batchSize {
			if !finished[i] {
				generatedTokens[i] = append(generatedTokens[i], flatTokens[i])
			}
		}

		eosMatchTensor := outputTensors[len(outputTensors)-1]
		var doneFlags []bool
		tensors.ConstFlatData(eosMatchTensor, func(flat []bool) {
			doneFlags = flat
		})
		for i, flag := range doneFlags {
			if flag {
				finished[i] = true
			}
		}

		allDone := true
		for _, isDone := range finished {
			if !isDone {
				allDone = false
				break
			}
		}
		if allDone {
			break
		}
		batch.InputValues = outputTensors
	}

	defer func() {
		for _, t := range outputTensors {
			t.FinalizeAll()
		}
	}()

	batch.OutputValues = make([]any, batchSize)
	for i := range generatedTokens {
		batch.OutputValues[i] = generatedTokens[i]
	}

	return nil
}

func runGoMLXSessionOnBatch(batch *PipelineBatch, p *BasePipeline) error {
	// Non-generative models
	var outputTensors []*tensors.Tensor
	defer func() {
		for _, t := range outputTensors {
			t.FinalizeAll()
		}
	}()

	err := exceptions.TryCatch[error](func() {
		outputTensors = p.Model.GoMLXModel.Exec.Call(batch.InputValues.([]*tensors.Tensor))
	})
	if err != nil {
		return err
	}

	convertedOutput := make([]any, len(outputTensors))
	for i, t := range outputTensors {
		var rawOutput []float32
		tensors.ConstFlatData(t, func(flat []float32) {
			rawOutput = flat
		})
		convertedOutput[i] = ReshapeOutput(rawOutput, p.Model.OutputsMeta[i], batch.PaddingMask, batch.MaxSequenceLength)
	}

	batch.OutputValues = convertedOutput
	return nil
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
