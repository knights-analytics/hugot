package pipelines

import (
	"fmt"

	ort "github.com/yalue/onnxruntime_go"
)

func loadInputOutputMetaORT(onnxBytes []byte) ([]InputOutputInfo, []InputOutputInfo, error) {
	inputs, outputs, err := ort.GetInputOutputInfoWithONNXData(onnxBytes)
	if err != nil {
		return nil, nil, err
	}
	return convertORTInputOutputs(inputs), convertORTInputOutputs(outputs), nil
}

func createORTSession(onnxBytes []byte, inputs, outputs []InputOutputInfo, options *ort.SessionOptions) (*ort.DynamicAdvancedSession, error) {
	var inputNames []string
	var outputNames []string
	for _, v := range inputs {
		inputNames = append(inputNames, v.Name)
	}
	for _, v := range outputs {
		outputNames = append(outputNames, v.Name)
	}
	session, err := ort.NewDynamicAdvancedSessionWithONNXData(
		onnxBytes,
		inputNames,
		outputNames,
		options,
	)
	return session, err
}

// createInputTensorsORT creates ort input tensors.
func createInputTensorsORT(batch *PipelineBatch, inputsMeta []InputOutputInfo) error {
	tensorSize := len(batch.Input) * (batch.MaxSequenceLength)
	batchSize := int64(len(batch.Input))

	inputTensors := make([]ort.Value, len(inputsMeta))
	var tensorCreationErr error

	for i, inputMeta := range inputsMeta {
		backingSlice := make([]int64, tensorSize)
		counter := 0

		for _, input := range batch.Input {
			length := len(input.TokenIDs)
			for j := 0; j < batch.MaxSequenceLength; j++ {
				if j+1 <= length {
					switch inputMeta.Name {
					case "input_ids":
						backingSlice[counter] = int64(input.TokenIDs[j])
					case "token_type_ids":
						backingSlice[counter] = int64(input.TypeIDs[j])
					case "attention_mask":
						backingSlice[counter] = int64(input.AttentionMask[j])
					default:
						return fmt.Errorf("input %s not recognized", inputMeta.Name)
					}
				} else {
					backingSlice[counter] = 0 // pad with zero
				}
				counter++
			}
		}
		inputTensors[i], tensorCreationErr = ort.NewTensor(ort.NewShape(batchSize, int64(batch.MaxSequenceLength)), backingSlice)
		if tensorCreationErr != nil {
			return tensorCreationErr
		}
	}
	batch.InputValuesORT = inputTensors
	return nil
}

func runORTSessionOnBatch(batch *PipelineBatch, session *ort.DynamicAdvancedSession, outputs []InputOutputInfo) error {
	actualBatchSize := int64(len(batch.Input))
	maxSequenceLength := int64(batch.MaxSequenceLength)

	// allocate vectors with right dimensions for the output
	outputTensors := make([]ort.Value, len(outputs))
	var outputCreationErr error

	for outputIndex, meta := range outputs {
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
		outputTensors[outputIndex], outputCreationErr = ort.NewEmptyTensor[float32](outputShape)
		if outputCreationErr != nil {
			return outputCreationErr
		}
	}

	errOnnx := session.Run(batch.InputValuesORT, outputTensors)
	if errOnnx != nil {
		return errOnnx
	}

	// store resulting tensors
	batch.OutputValuesORT = outputTensors
	return nil
}

func destroySession(tk *Tokenizer, session *ort.DynamicAdvancedSession) error {
	var finalErr error
	if tk.RustTokenizer != nil {
		errTokenizer := tk.RustTokenizer.Close()
		if errTokenizer != nil {
			finalErr = errTokenizer
		}
	}
	if session != nil {
		ortError := session.Destroy()
		if ortError != nil {
			finalErr = ortError
		}
	}
	return finalErr
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
