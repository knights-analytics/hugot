//go:build !NOORT || ALL

package pipelines

import (
	"errors"
	"fmt"

	"github.com/knights-analytics/hugot/options"

	ort "github.com/yalue/onnxruntime_go"
)

type ORTSession struct {
	Session    *ort.DynamicAdvancedSession
	ORTOptions *ort.SessionOptions
	Destroy    func() error
}

func createORTPipeline(pipeline *BasePipeline, onnxBytes []byte, options *options.Options) error {

	optionsCast := options.RuntimeOptions.(*ort.SessionOptions)

	inputs, outputs, err := loadInputOutputMetaORT(onnxBytes)
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
		onnxBytes,
		inputNames,
		outputNames,
		optionsCast,
	)
	if errSession != nil {
		return errSession
	}

	pipeline.ORTSession = &ORTSession{Session: session, ORTOptions: optionsCast, Destroy: func() error {
		return session.Destroy()
	}}
	pipeline.InputsMeta = inputs
	pipeline.OutputsMeta = outputs

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
		inputTensors[i], tensorCreationErr = ort.NewTensor(ort.NewShape(int64(batchSize), int64(batch.MaxSequenceLength)), backingSlice)
		if tensorCreationErr != nil {
			return tensorCreationErr
		}
	}
	batch.InputValues = inputTensors
	batch.DestroyInputs = func() error {
		var destroyError error
		for _, ortTensor := range inputTensors {
			destroyError = errors.Join(destroyError, ortTensor.Destroy())
		}
		return destroyError
	}

	return nil
}

func runORTSessionOnBatch(batch *PipelineBatch, p *BasePipeline) error {
	actualBatchSize := int64(len(batch.Input))
	maxSequenceLength := int64(batch.MaxSequenceLength)
	var err error

	// allocate vectors with right dimensions for the output
	outputTensors := make([]ort.Value, len(p.OutputsMeta))
	defer func() {
		for _, output := range outputTensors {
			err = errors.Join(err, output.Destroy())
		}
	}()

	for outputIndex, meta := range p.OutputsMeta {
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

	errOnnx := p.ORTSession.Session.Run(batch.InputValues.([]ort.Value), outputTensors)
	if errOnnx != nil {
		return errOnnx
	}

	convertedOutput := make([][]float32, len(outputTensors))
	for i, t := range outputTensors {
		convertedOutput[i] = t.(*ort.Tensor[float32]).GetData()
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
