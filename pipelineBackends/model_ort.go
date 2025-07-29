//go:build ORT || ALL

package pipelineBackends

import (
	"errors"
	"fmt"
	"os"
	"time"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/knights-analytics/hugot/options"
)

type ORTModel struct {
	Session        *ort.DynamicAdvancedSession
	SessionOptions *ort.SessionOptions
	Destroy        func() error
}

func createORTModelBackend(model *Model, options *options.Options) error {
	os.Chdir("/home/testuser/repositories/hugot/models/KnightsAnalytics_gemma-3-1b-it-ONNX")
	sessionOptions := options.BackendOptions.(*ort.SessionOptions)

	inputs, outputs, err := loadInputOutputMetaORT(model.OnnxBytes)
	if err != nil {
		return err
	}

	var inputNames []string
	var outputNames []string
	for _, v := range inputs {
		inputNames = append(inputNames, v.Name)
	}
	for _, v := range outputs {
		outputNames = append(outputNames, v.Name)
	}
	session, errSession := ort.NewDynamicAdvancedSessionWithONNXData(
		model.OnnxBytes,
		inputNames,
		outputNames,
		sessionOptions,
	)
	if errSession != nil {
		return errSession
	}

	model.ORTModel = &ORTModel{Session: session, SessionOptions: sessionOptions, Destroy: func() error {
		return session.Destroy()
	}}
	model.InputsMeta = inputs
	model.OutputsMeta = outputs
	os.Chdir("/home/testuser/repositories/hugot")
	return nil
}

func loadInputOutputMetaORT(onnxBytes []byte) ([]InputOutputInfo, []InputOutputInfo, error) {
	inputs, outputs, err := ort.GetInputOutputInfoWithONNXData(onnxBytes)
	if err != nil {
		return nil, nil, err
	}
	return convertORTInputOutputs(inputs), convertORTInputOutputs(outputs), nil
}

func createInputTensorsORT(batch *PipelineBatch, inputsMeta []InputOutputInfo) error {
	batchSize := len(batch.Input)
	tensorSize := batchSize * batch.MaxSequenceLength

	inputTensors := make([]ort.Value, len(inputsMeta))
	var tensorCreationErr error

	paddingMasks := make([][]bool, batchSize)

	for i, inputMeta := range inputsMeta {
		backingSlice := make([]int64, tensorSize)
		counter := 0

		for j, input := range batch.Input {
			inputPaddingMask := make([]bool, batch.MaxSequenceLength)
			length := len(input.TokenIDs)
			for k := 0; k < batch.MaxSequenceLength; k++ {
				if k+1 <= length {
					switch inputMeta.Name {
					case "input_ids":
						backingSlice[counter] = int64(input.TokenIDs[k])
						inputPaddingMask[k] = true
					case "token_type_ids":
						backingSlice[counter] = int64(input.TypeIDs[k])
					case "attention_mask":
						backingSlice[counter] = int64(input.AttentionMask[k])
					default:
						return fmt.Errorf("input %s not recognized", inputMeta.Name)
					}
				} else {
					backingSlice[counter] = 0 // pad with zero
				}
				counter++
			}

			if inputMeta.Name == "input_ids" {
				paddingMasks[j] = inputPaddingMask
			}
		}
		inputTensors[i], tensorCreationErr = ort.NewTensor(ort.NewShape(int64(batchSize), int64(batch.MaxSequenceLength)), backingSlice)
		if tensorCreationErr != nil {
			return tensorCreationErr
		}
	}
	batch.InputValues = inputTensors
	batch.PaddingMask = paddingMasks
	batch.DestroyInputs = func() error {
		var destroyError error
		for _, ortTensor := range inputTensors {
			destroyError = errors.Join(destroyError, ortTensor.Destroy())
		}
		return destroyError
	}

	return nil
}

func CreateGenerativeInputTensorsORT(batch *PipelineBatch) error {
	for _, i := range batch.Input {
		batch.MaxSequenceLength = max(batch.MaxSequenceLength, len(i.TokenIDs))
	}
	batchSize := len(batch.Input)
	maxSeqLength := batch.MaxSequenceLength
	tensorSize := batchSize * maxSeqLength
	inputTensors := make([]ort.Value, 2)
	var tensorCreationErr error
	paddingMasks := make([][]bool, batchSize)
	inputIDsSlice := make([]int64, tensorSize)
	counter := 0
	for j, input := range batch.Input {
		seq := input.TokenIDs
		seqLen := len(seq)
		padLen := maxSeqLength - seqLen
		inputPaddingMask := make([]bool, maxSeqLength)

		for k := range maxSeqLength {
			if k < padLen {
				inputIDsSlice[counter] = 0 // padding
				inputPaddingMask[k] = false
			} else {
				inputIDsSlice[counter] = int64(seq[k-padLen])
				inputPaddingMask[k] = true
			}
			counter++
		}
		paddingMasks[j] = inputPaddingMask
	}

	inputTensors[0], tensorCreationErr = ort.NewTensor(ort.NewShape(int64(batchSize), int64(maxSeqLength)), inputIDsSlice)
	if tensorCreationErr != nil {
		return tensorCreationErr
	}

	positionIDsSlice := make([]int64, tensorSize)
	counter = 0
	for range batchSize {
		for j := range maxSeqLength {
			positionIDsSlice[counter] = int64(j + 1)
			counter++
		}
	}

	inputTensors[1], tensorCreationErr = ort.NewTensor(ort.NewShape(int64(batchSize), int64(maxSeqLength)), positionIDsSlice)
	if tensorCreationErr != nil {
		inputTensors[0].Destroy()
		return tensorCreationErr
	}

	batch.InputValues = inputTensors
	batch.PaddingMask = paddingMasks
	batch.DestroyInputs = func() error {
		var destroyError error
		for _, ortTensor := range inputTensors {
			destroyError = errors.Join(destroyError, ortTensor.Destroy())
		}
		return destroyError
	}

	return nil
}

func CreateCacheORT(batchSize, numLayers, numKeyValueHeads, maxSeqLen, headDim int) ([]ort.Value, error) {
	cache := make([]ort.Value, numLayers*2)
	tensorSize := batchSize * numKeyValueHeads * maxSeqLen * headDim
	for layer := range numLayers {
		keySlice := make([]float32, tensorSize)
		keyTensor, err := ort.NewTensor(
			ort.NewShape(int64(batchSize), int64(numKeyValueHeads), int64(maxSeqLen), int64(headDim)),
			keySlice,
		)
		if err != nil {
			for i := 0; i < layer*2; i++ {
				if cache[i] != nil {
					cache[i].Destroy()
				}
			}
			return nil, err
		}
		cache[layer*2] = keyTensor
		valueSlice := make([]float32, tensorSize)
		valueTensor, err := ort.NewTensor(
			ort.NewShape(int64(batchSize), int64(numKeyValueHeads), int64(maxSeqLen), int64(headDim)),
			valueSlice,
		)
		if err != nil {
			keyTensor.Destroy()
			for i := 0; i < layer*2; i++ {
				if cache[i] != nil {
					cache[i].Destroy()
				}
			}
			return nil, err
		}
		cache[layer*2+1] = valueTensor
	}
	return cache, nil
}

func runORTSessionOnBatch(batch *PipelineBatch, p *BasePipeline) error {
	actualBatchSize := int64(len(batch.Input))
	maxSequenceLength := int64(batch.MaxSequenceLength)
	var err error

	// allocate vectors with right dimensions for the output
	outputTensors := make([]ort.Value, len(p.Model.OutputsMeta))
	defer func() {
		for _, output := range outputTensors {
			err = errors.Join(err, output.Destroy())
		}
	}()

	for outputIndex, meta := range p.Model.OutputsMeta {
		var batchDimSet bool
		var tokenDimSet bool
		actualDims := make([]int64, 0, len(meta.Dimensions))

		for _, dim := range meta.Dimensions {
			if dim == -1 {
				if !batchDimSet {
					actualDims = append(actualDims, actualBatchSize)
					batchDimSet = true
				} else if !tokenDimSet {
					actualDims = append(actualDims, maxSequenceLength)
					tokenDimSet = true
				} else {
					return fmt.Errorf("only two axis can be dynamic (batch size and number of tokens)")
				}
			} else {
				actualDims = append(actualDims, dim)
			}
		}
		outputShape := ort.NewShape(actualDims...)
		outputTensors[outputIndex], err = ort.NewEmptyTensor[float32](outputShape)
		if err != nil {
			return err
		}
	}

	errOnnx := p.Model.ORTModel.Session.Run(batch.InputValues.([]ort.Value), outputTensors)
	if errOnnx != nil {
		return errOnnx
	}

	convertedOutput := make([]any, len(outputTensors))
	for i, t := range outputTensors {
		switch v := t.(type) {
		case *ort.Tensor[float32]:
			convertedOutput[i] = ReshapeOutput(v.GetData(), p.Model.OutputsMeta[i], batch.PaddingMask, batch.MaxSequenceLength)
		case *ort.Tensor[int64]:
			convertedOutput[i] = ReshapeOutput(v.GetData(), p.Model.OutputsMeta[i], batch.PaddingMask, batch.MaxSequenceLength)
		}
	}

	// store resulting tensors
	batch.OutputValues = convertedOutput

	return err
}

func argmax(logits [][][]float32) []int64 {
	batchSize := len(logits)
	if batchSize == 0 {
		return nil
	}

	output := make([]int64, batchSize)
	for i := range output {
		if len(logits[i]) == 0 {
			output[i] = 0
			continue
		}

		lastTokenLogits := logits[i][len(logits[i])-1]

		maxIdx := 0
		maxVal := lastTokenLogits[0]
		for j, val := range lastTokenLogits[1:] {
			if val > maxVal {
				maxVal = val
				maxIdx = j + 1
			}
		}

		output[i] = int64(maxIdx)
	}

	return output
}

func FlatTo3D(flat []float32, batchSize, vocabSize int) [][][]float32 {
	result := make([][][]float32, batchSize)
	for i := range batchSize {
		result[i] = make([][]float32, 1)
		result[i][0] = make([]float32, vocabSize)
		start := i * vocabSize
		copy(result[i][0], flat[start:start+vocabSize])
	}
	return result
}

func runGenerativeORTSessionOnBatch(batch *PipelineBatch, p *BasePipeline) error {
	start := time.Now()
	batchSize := int64(len(batch.Input))
	finished := make([]bool, batchSize)
	generatedTokens := make([][]int64, batchSize)
	eosTokenID := int64(p.Model.EosTokenIDs[1])

	for step := 0; step < batch.MaxNewTokens; step++ {
		inputTensors := batch.InputValues.([]ort.Value)
		outputTensors := make([]ort.Value, len(p.Model.OutputsMeta))
		errOnnx := p.Model.ORTModel.Session.Run(inputTensors, outputTensors)
		if errOnnx != nil {
			return errOnnx
		}

		logits := outputTensors[0].(*ort.Tensor[float32]).GetData()
		newCache := outputTensors[1:]

		var logitsReshaped [][][]float32
		if step == 0 {
			logitsReshaped = ReshapeOutput(logits, p.Model.OutputsMeta[0], batch.PaddingMask, batch.MaxSequenceLength).([][][]float32)
		} else {
			logitsReshaped = FlatTo3D(
				logits,
				int(batchSize),
				262144, // hardcoded vocab size
			)
		}

		// should give an array of batchSize amount of tokens
		greedyTokens := argmax(logitsReshaped)
		allFinished := true
		for i := range batchSize {
			if !finished[i] {
				generatedTokens[i] = append(generatedTokens[i], greedyTokens[i])
				if greedyTokens[i] == eosTokenID {
					finished[i] = true
				}
			}
			if !finished[i] {
				allFinished = false
			}
		}
		if allFinished {
			break
		}

		generatedTokenTensor, err := ort.NewTensor(
			ort.NewShape(batchSize, 1),
			greedyTokens,
		)
		if err != nil {
			return err
		}

		positionIDs := inputTensors[1].(*ort.Tensor[int64]).GetData()
		flatPositionIDs := flatDataTo2D(
			positionIDs,
			batch.PaddingMask,
			len(positionIDs)/int(batchSize),
		)

		newPositionIDs := make([]int64, batchSize)
		for i := range flatPositionIDs {
			newPositionIDs[i] = flatPositionIDs[i][len(flatPositionIDs[i])-1] + 1
		}

		newPositionIDsTensor, err := ort.NewTensor(
			ort.NewShape(batchSize, 1),
			newPositionIDs,
		)
		if err != nil {
			return err
		}

		newModelInputs := []ort.Value{generatedTokenTensor, newPositionIDsTensor}
		for i := range newCache {
			newModelInputs = append(newModelInputs, newCache[i])
		}

		batch.InputValues = newModelInputs

	}

	batch.OutputValues = make([]any, batchSize)
	for i := range generatedTokens {
		batch.OutputValues[i] = generatedTokens[i]
	}
	fmt.Println("generation loop runtime: ", time.Since(start))
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
