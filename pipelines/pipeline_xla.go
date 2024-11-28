//go:build XLA || ALL

package pipelines

import (
	"fmt"

	"github.com/knights-analytics/hugot/options"

	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/xla/cpu/static"
	"github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/onnx-gomlx/onnx"
)

type XLASession struct {
	Backend backends.Backend

	// ONNX model
	Onnx *onnx.Model

	// ctx with the model's weights.
	Ctx *context.Context

	// exec is used to execute the model with a context.
	Exec *context.Exec

	Destroy func()
}

func createXLAPipeline(pipeline *BasePipeline, onnxBytes []byte, _ *options.Options) error {

	model, err := onnx.Parse(onnxBytes)
	if err != nil {
		return err
	}

	inputs, outputs := loadInputOutputMetaXLA(model)

	var outputNames []string
	for _, v := range outputs {
		outputNames = append(outputNames, v.Name)
	}

	ctx := context.New()
	ctx = ctx.Reuse() // Mark it to reuse variables: it will be an error to create a new variable – for safety.
	// Read variables from ONNX model.
	err = model.VariablesToContext(ctx)
	if err != nil {
		return err
	}

	backend := backends.New()

	// Create model executor.
	exec := context.NewExec(
		backend, ctx,
		func(ctx *context.Context, inputs []*graph.Node) (choice *graph.Node) {
			inputsMap := map[string]*graph.Node{
				"input_ids":      inputs[0],
				"attention_mask": inputs[1]}
			if model.NumInputs() == 3 {
				inputsMap["token_type_ids"] = inputs[2]
			}
			results := model.CallGraph(ctx, inputs[0].Graph(), inputsMap, outputNames...)
			return results[0]
		})

	pipeline.XLASession = &XLASession{
		Backend: backend,
		Onnx:    model,
		Ctx:     ctx,
		Exec:    exec,
		Destroy: func() {
			exec.Finalize()
			backend.Finalize()
		},
	}
	pipeline.InputsMeta = inputs
	pipeline.OutputsMeta = outputs

	return err
}

func loadInputOutputMetaXLA(model *onnx.Model) ([]InputOutputInfo, []InputOutputInfo) {

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

func createInputTensorsXLA(batch *PipelineBatch, inputsMeta []InputOutputInfo) error {
	batchSize := len(batch.Input)
	tensorSize := batchSize * batch.MaxSequenceLength

	inputTensors := make([]*tensors.Tensor, len(inputsMeta))
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
		inputTensors[i] = tensors.FromFlatDataAndDimensions(backingSlice, batchSize, batch.MaxSequenceLength)
	}
	batch.InputValues = inputTensors
	batch.DestroyInputs = func() error {
		for _, input := range inputTensors {
			input.FinalizeAll()
		}
		return nil
	}
	return nil
}

func runXLASessionOnBatch(batch *PipelineBatch, p *BasePipeline) error {

	var outputs []*tensors.Tensor
	defer func() {
		for _, output := range outputs {
			output.FinalizeAll()
		}
	}()

	err := exceptions.TryCatch[error](func() {
		outputs = p.XLASession.Exec.Call(batch.InputValues.([]*tensors.Tensor))
	})
	if err != nil {
		return err
	}

	convertedOutput := make([][]float32, len(outputs))
	for i, t := range outputs {
		convertedOutput[i] = tensors.CopyFlatData[float32](t)
	}

	// store resulting tensors
	batch.OutputValues = convertedOutput

	return nil
}
