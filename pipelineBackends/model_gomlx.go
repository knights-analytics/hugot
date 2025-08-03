package pipelineBackends

import (
	"errors"
	"fmt"
	"io"

	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/onnx-gomlx/onnx"

	"github.com/knights-analytics/hugot/options"

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

func createGoMLXModelBackend(model *Model, options *options.Options) error {
	var insideError error
	var recoverErr error

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

func createInputTensorsGoMLX(batch *PipelineBatch, model *Model, padBatchDimension bool) error {
	batchSize := len(batch.Input)
	batchSizePadded := batchSize
	if padBatchDimension {
		batchSizePadded = nextPowerOf2(batchSize)
	}
	maxSequenceLengthPadded := nextPowerOf2(batch.MaxSequenceLength)
	tensorSize := batchSizePadded * maxSequenceLengthPadded

	inputTensors := make([]*tensors.Tensor, len(model.InputsMeta))
	paddingMasks := make([][]bool, batchSize)
	for i, inputMeta := range model.InputsMeta {
		backingSlice := make([]int64, tensorSize)
		counter := 0
		for j, input := range batch.Input {
			paddingMask := make([]bool, maxSequenceLengthPadded)
			length := len(input.TokenIDs)
			for k := 0; k < maxSequenceLengthPadded; k++ {
				if k+1 <= length {
					switch inputMeta.Name {
					case "input_ids":
						backingSlice[counter] = int64(input.TokenIDs[k])
						paddingMask[k] = true
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
				paddingMasks[j] = paddingMask
			}
		}
		inputTensors[i] = tensors.FromFlatDataAndDimensions(backingSlice, batchSizePadded, maxSequenceLengthPadded)
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

func runGoMLXSessionOnBatch(batch *PipelineBatch, p *BasePipeline) error {

	var outputTensors []*tensors.Tensor
	defer func() {
		for _, output := range outputTensors {
			output.FinalizeAll()
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
		switch t.DType() {
		case dtypes.Float32:
			tensors.ConstFlatData(t, func(flat []float32) {
				convertedOutput[i] = ReshapeOutput(flat, p.Model.OutputsMeta[i], batch.PaddingMask, batch.MaxSequenceLength)
			})
		case dtypes.Int64:
			tensors.ConstFlatData(t, func(flat []int64) {
				convertedOutput[i] = ReshapeOutput(flat, p.Model.OutputsMeta[i], batch.PaddingMask, batch.MaxSequenceLength)
			})
		}
	}

	// store resulting tensors
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
