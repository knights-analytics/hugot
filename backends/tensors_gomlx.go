package backends

import (
	"context"
	"errors"
	"fmt"
	"slices"

	"github.com/gomlx/gomlx/core/tensors"
)

func runNamedTensorsGoMLX(ctx context.Context, model *Model, inputs map[string]Tensor) (result map[string]Tensor, err error) {
	m := model.GoMLXModel
	if m.Exec == nil || m.OnnxModel == nil {
		return nil, errors.New("GoMLX executor is not initialized")
	}
	var allocated []*tensors.Tensor
	defer func() {
		for _, tensor := range allocated {
			if tensor != nil {
				err = errors.Join(err, tensor.FinalizeAll())
			}
		}
		if recovered := recover(); recovered != nil {
			err = errors.Join(err, fmt.Errorf("GoMLX tensor inference: %v", recovered))
		}
		if err != nil {
			result = nil
		}
	}()
	names, shapes := m.OnnxModel.Inputs()
	args := make([]any, len(model.InputsMeta))
	for i, meta := range model.InputsMeta {
		input := inputs[meta.Name]
		dims := make([]int, len(input.Shape))
		for j, dim := range input.Shape {
			dims[j] = int(dim)
		}
		var tensor *tensors.Tensor
		switch data := input.Data.(type) {
		case []float32:
			tensor = tensors.FromFlatDataAndDimensions(data, dims...)
		case []int64:
			tensor = tensors.FromFlatDataAndDimensions(data, dims...)
		case []bool:
			tensor = tensors.FromFlatDataAndDimensions(data, dims...)
		}
		allocated = append(allocated, tensor)
		index := slices.Index(names, meta.Name)
		if index < 0 || tensor.DType() != shapes[index].DType {
			return nil, fmt.Errorf("input %q data type %s does not match graph metadata", meta.Name, tensor.DType())
		}
		args[i] = tensor
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	// Exec is synchronous: cancellation must not free buffers while it is running.
	outputs, execErr := m.Exec.Call(args...)
	allocated = append(allocated, outputs...)
	if execErr != nil {
		return nil, execErr
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	if len(outputs) != len(model.OutputsMeta) {
		return nil, fmt.Errorf("GoMLX returned %d outputs, expected %d", len(outputs), len(model.OutputsMeta))
	}
	result = make(map[string]Tensor, len(outputs))
	for i, output := range outputs {
		shape := make([]int64, output.Rank())
		for j, dim := range output.Shape().Dimensions {
			shape[j] = int64(dim)
		}
		var data any
		switch output.DType().String() {
		case "Float32":
			err = tensors.ConstFlatData(output, func(flat []float32) { data = slices.Clone(flat) })
		case "Int64":
			err = tensors.ConstFlatData(output, func(flat []int64) { data = slices.Clone(flat) })
		case "Bool":
			err = tensors.ConstFlatData(output, func(flat []bool) { data = slices.Clone(flat) })
		default:
			return nil, fmt.Errorf("output %q has unsupported data type %s", model.OutputsMeta[i].Name, output.DType())
		}
		if err != nil {
			return nil, err
		}
		result[model.OutputsMeta[i].Name] = Tensor{Shape: shape, Data: data}
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	return result, nil
}
