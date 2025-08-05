//go:build ORT || ALL

package pipelineBackends

import (
	"errors"
	"fmt"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/knights-analytics/hugot/options"
)

type ORTModel struct {
	Session        *ort.DynamicAdvancedSession
	SessionOptions *ort.SessionOptions
	Destroy        func() error
}

func createORTModelBackend(model *Model, options *options.Options) error {

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

	return nil
}

func loadInputOutputMetaORT(onnxBytes []byte) ([]InputOutputInfo, []InputOutputInfo, error) {
	inputs, outputs, err := ort.GetInputOutputInfoWithONNXData(onnxBytes)
	if err != nil {
		return nil, nil, err
	}
	return convertORTInputOutputs(inputs), convertORTInputOutputs(outputs), nil
}

func createInputTensorsORT(batch *PipelineBatch, model *Model) error {
	batchSize := batch.Size
	tensorSize := batchSize * batch.MaxSequenceLength

	inputTensors := make([]ort.Value, len(model.InputsMeta))
	var tensorCreationErr error

	paddingMasks := make([][]bool, batchSize)

	for i, inputMeta := range model.InputsMeta {
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

func runORTSessionOnBatch(batch *PipelineBatch, p *BasePipeline) error {
	actualBatchSize := int64(batch.Size)
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
