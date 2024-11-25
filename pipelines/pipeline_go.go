package pipelines

import (
	"fmt"

	"github.com/advancedclimatesystems/gonnx"
	"gorgonia.org/tensor"
)

func createGoSession(onnxBytes []byte) (*gonnx.Model, []InputOutputInfo, []InputOutputInfo, error) {

	model, err := gonnx.NewModelFromBytes(onnxBytes)
	if err != nil {
		return nil, nil, nil, err
	}

	inputs, outputs, errLoad := loadInputOutputMetaGo(model)
	if errLoad != nil {
		return nil, nil, nil, errLoad
	}

	return model, inputs, outputs, err
}

func loadInputOutputMetaGo(model *gonnx.Model) ([]InputOutputInfo, []InputOutputInfo, error) {

	var inputs, outputs []InputOutputInfo

	inputShapes := model.InputShapes()
	for _, name := range model.InputNames() {
		shape := inputShapes[name]
		dimensions := make([]int64, len(shape))
		for i, y := range shape {
			dimensions[i] = y.Size
		}
		inputs = append(inputs, InputOutputInfo{
			Name:       name,
			Dimensions: dimensions,
		})
	}
	outputShapes := model.OutputShapes()
	for _, name := range model.OutputNames() {
		shape := outputShapes[name]
		dimensions := make([]int64, len(shape))
		for i, y := range shape {
			dimensions[i] = y.Size
		}
		outputs = append(inputs, InputOutputInfo{
			Name:       name,
			Dimensions: dimensions,
		})
	}
	return inputs, outputs, nil
}

// createInputTensorsGo creates ort input tensors.
func createInputTensorsGo(batch *PipelineBatch, inputsMeta []InputOutputInfo) error {

	inputMap := map[string]tensor.Tensor{}
	for _, inputMeta := range inputsMeta {
		backingSlice := make([]uint32, len(batch.Input))
		counter := 0

		for _, input := range batch.Input {
			length := len(input.TokenIDs)
			for j := 0; j < batch.MaxSequenceLength; j++ {
				if j+1 <= length {
					switch inputMeta.Name {
					case "input_ids":
						backingSlice[counter] = input.TokenIDs[j]
					case "token_type_ids":
						backingSlice[counter] = input.TypeIDs[j]
					case "attention_mask":
						backingSlice[counter] = input.AttentionMask[j]
					default:
						return fmt.Errorf("input %s not recognized", inputMeta.Name)
					}
				} else {
					backingSlice[counter] = 0 // pad with zero
				}
				counter++
			}
		}
		inputMap[inputMeta.Name] = tensor.New(
			tensor.Of(tensor.Uint32),
			tensor.WithShape(len(batch.Input), batch.MaxSequenceLength),
			tensor.WithBacking(backingSlice),
		)
	}
	batch.InputValuesGo = inputMap
	return nil
}

func runGoSessionOnBatch(batch *PipelineBatch, session *gonnx.Model) error {

	tensors, err := session.Run(batch.InputValuesGo)
	if err != nil {
		return err
	}
	// store resulting tensors
	batch.OutputValuesGo = tensors
	return nil
}
