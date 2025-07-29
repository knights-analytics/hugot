package pipelineBackends

import (
	"errors"
	"fmt"
	"io"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

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
	fmt.Println(inputIDs)
	fmt.Println(positionIDs)
	inputIDsTensor := tensors.FromAnyValue(inputIDs)
	positionIDsTensor := tensors.FromAnyValue(positionIDs)
	modelInputs := []*tensors.Tensor{inputIDsTensor, positionIDsTensor}
	batch.InputValues = modelInputs
	return nil
}

// func CreateGenerativeInputTensorsGoMLX(batch *PipelineBatch) error {
// 	for _, i := range batch.Input {
// 		batch.MaxSequenceLength = max(batch.MaxSequenceLength, len(i.TokenIDs))
// 	}
// 	paddedLength := nextPowerOf2(batch.MaxSequenceLength)
// 	batchSize := len(batch.Input)
// 	inputIDs := make([][]int64, batchSize)
// 	positionIDs := make([][]int64, batchSize)
// 	// attentionMask := make([][]int64, batchSize)
// 	currentPositions := make([]int64, batchSize)

// 	for i := range batchSize {
// 		seq := batch.Input[i].TokenIDs
// 		seqLen := len(seq)
// 		padLen := paddedLength - seqLen

// 		// Create padded input
// 		paddedInput := make([]int64, paddedLength)
// 		for j := range paddedInput {
// 			paddedInput[j] = 0 // <pad> token
// 		}
// 		for j := range seqLen {
// 			paddedInput[padLen+j] = int64(seq[j])
// 		}
// 		inputIDs[i] = paddedInput

// 		// Create position IDs
// 		posIDs := make([]int64, paddedLength)
// 		for j := range paddedLength {
// 			posIDs[j] = int64(j + 1)
// 		}
// 		positionIDs[i] = posIDs

// 		// Create attention mask: 0 for padding tokens, 1 for actual tokens
// 		attMask := make([]int64, paddedLength)
// 		for j := range paddedLength {
// 			if j < padLen {
// 				attMask[j] = 0 // Padding tokens
// 			} else {
// 				attMask[j] = 1 // Actual tokens
// 			}
// 		}
// 		// attentionMask[i] = attMask

// 		currentPositions[i] = int64(paddedLength)
// 	}

// 	fmt.Println(inputIDs)
// 	fmt.Println(positionIDs)

// 	inputIDsTensor := tensors.FromAnyValue(inputIDs)
// 	positionIDsTensor := tensors.FromAnyValue(positionIDs)

// 	modelInputs := []*tensors.Tensor{inputIDsTensor, positionIDsTensor}
// 	batch.InputValues = modelInputs

// 	return nil
// }

func sortedKeys(m map[string]*graph.Node) []string {
	type kv struct {
		key   string
		index int
	}

	re := regexp.MustCompile(`past_key_values\.(\d+)\.`)

	kvs := make([]kv, 0, len(m))
	for k := range m {
		match := re.FindStringSubmatch(k)
		if len(match) != 2 {
			continue
		}
		num, err := strconv.Atoi(match[1])
		if err != nil {
			continue
		}
		kvs = append(kvs, kv{key: k, index: num})
	}

	sort.Slice(kvs, func(i, j int) bool {
		if kvs[i].index == kvs[j].index {
			return kvs[i].key < kvs[j].key
		}
		return kvs[i].index < kvs[j].index
	})

	sorted := make([]string, len(kvs))
	for i, kv := range kvs {
		sorted[i] = kv.key
	}
	return sorted
}

func createGenerativeCallFunc(model *Model, modelParsed *onnx.Model, outputNames []string) func(ctx *context.Context, inputs []*graph.Node) []*graph.Node {
	return func(ctx *context.Context, inputs []*graph.Node) []*graph.Node {
		inputsMap := map[string]*graph.Node{
			"input_ids":    inputs[0],
			"position_ids": inputs[1],
			// "attention_mask": inputs[2],
		}
		pastKeyValues := make(map[string]*graph.Node)
		for i := range model.NumHiddenLayers {
			key := fmt.Sprintf("past_key_values.%d.key", i)
			value := fmt.Sprintf("past_key_values.%d.value", i)
			inputsMap[key] = inputs[2*i+2]
			inputsMap[value] = inputs[2*i+3]
			pastKeyValues[key] = inputs[2*i+2]
			pastKeyValues[value] = inputs[2*i+3]
		}
		g := inputs[0].Graph()
		modelOutputs := modelParsed.CallGraph(ctx, g, inputsMap, outputNames...)
		logits := modelOutputs[0]
		presentKeyValues := modelOutputs[1:]

		shape := logits.Shape().Dimensions
		vocabSize := shape[2]
		seqLen := shape[1]
		lastIdx := Scalar(g, dtypes.Int32, seqLen-1)

		// greedily select next token
		logitsLast := DynamicSlice(
			logits,
			[]*Node{ScalarZero(g, dtypes.Int32), lastIdx, ScalarZero(g, dtypes.Int32)},
			[]int{shape[0], 1, vocabSize},
		)
		batchSize := shape[0]
		logitsLast = Reshape(logitsLast, batchSize, vocabSize)
		nextPredictedToken := ArgMax(logitsLast, 1, dtypes.Int64)
		nextPredictedToken = Reshape(nextPredictedToken, batchSize, 1)

		// early stopping flags
		var terminate *graph.Node
		for i, eosID := range model.EosTokenIDs {
			eosNode := Squeeze(Equal(nextPredictedToken, Scalar(g, dtypes.Int64, eosID)))
			if i == 0 {
				terminate = eosNode
			} else {
				terminate = Or(terminate, eosNode)
			}
		}

		// prepare new input and position IDs
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
		offset := posLen

		currentPosValue := Squeeze(prevPosLast)
		var currentIteration *graph.Node

		if currentPosValue.Rank() == 0 {
			currentIteration = Sub(currentPosValue, Scalar(g, dtypes.Int64, offset))
		} else {
			first := Slice(currentPosValue, AxisElem(0))
			scalar := Squeeze(first)
			currentIteration = Sub(scalar, Scalar(g, dtypes.Int64, offset))
		}

		// deal with new cache
		updatedKvCache := make([]*Node, len(presentKeyValues))
		keys := sortedKeys(pastKeyValues)
		if model.FirstIteration {
			// past_key_values[key][:, :, :offset, :] = present_key_values[j][:, :, fixed_cache_value:, :]
			j := 0
			for _, key := range keys {
				fixedCacheValue := model.FixedCacheSize
				update := Slice(
					presentKeyValues[j],
					AxisRange(),
					AxisRange(),
					AxisRange(fixedCacheValue),
					AxisRange(),
				)

				updatedKvCache[j] = DynamicUpdateSlice(
					pastKeyValues[key],
					update,
					[]*Node{
						ScalarZero(g, dtypes.Int32),
						ScalarZero(g, dtypes.Int32),
						ScalarZero(g, dtypes.Int32),
						ScalarZero(g, dtypes.Int32),
					},
				)
				j++
			}
		} else {
			// past_key_values[key][:, :, offset+i:offset+i+1, :] = present_key_values[j][:, :, -1:, :]
			j := 0
			updatePos := ConvertType(Add(Scalar(g, dtypes.Int64, offset), currentIteration), dtypes.Int32)
			for _, key := range keys {
				update := Slice(
					presentKeyValues[j],
					AxisRange(),
					AxisRange(),
					AxisElem(-1),
					AxisRange(),
				)

				updatedKvCache[j] = DynamicUpdateSlice(
					pastKeyValues[key],
					update,
					[]*Node{
						ScalarZero(g, dtypes.Int32),
						ScalarZero(g, dtypes.Int32),
						updatePos,
						ScalarZero(g, dtypes.Int32),
					},
				)
				j++
			}
		}

		outputs := append(append([]*Node{inputIDs, positionIDs}, updatedKvCache...), terminate)
		return outputs
	}
}

func RunGenerativeGoMLXSessionOnBatch(batch *PipelineBatch, p *BasePipeline) error {
	// Generative models
	batchSize := len(batch.Input)
	finished := make([]bool, batchSize)
	generatedTokens := make([][]int64, batchSize)

	var outputTensors []*tensors.Tensor
	start := time.Now()
	for step := 0; step < batch.MaxNewTokens; step++ {
		if step == 0 {
			p.Model.FirstIteration = true
		} else {
			p.Model.FirstIteration = false
		}
		modelInputs := batch.InputValues.([]*tensors.Tensor)

		outputTensors = p.Model.GoMLXModel.Exec.Call(modelInputs)

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
	fmt.Println(time.Since(start))

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
