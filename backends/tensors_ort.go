//go:build cgo && (ORT || ALL)

package backends

import (
	"context"
	"errors"
	"fmt"
	"slices"

	ort "github.com/microsoft/onnxruntime/go/onnxruntime"
)

func runNamedTensorsORT(ctx context.Context, model *Model, inputs map[string]Tensor) (result map[string]Tensor, err error) {
	m := model.ORTModel
	if m.Session == nil || m.Session.session == nil {
		return nil, errors.New("ORT session is not initialized")
	}
	allocated := make(map[string]*ort.Tensor, len(inputs))
	var outputs map[string]*ort.Tensor
	defer func() {
		for _, tensor := range outputs {
			if tensor != nil {
				err = errors.Join(err, tensor.Close())
			}
		}
		for _, tensor := range allocated {
			err = errors.Join(err, tensor.Close())
		}
		if err != nil {
			result = nil
		}
	}()
	for _, meta := range m.Session.session.Inputs() {
		input := inputs[meta.Name]
		var tensor *ort.Tensor
		var dtype ort.TensorElementDataType
		switch data := input.Data.(type) {
		case []float32:
			dtype = ort.TensorElementDataTypeFloat32
			if dtype == meta.DataType {
				tensor, err = ort.CreateTensor(input.Shape, data)
			}
		case []int64:
			dtype = ort.TensorElementDataTypeInt64
			if dtype == meta.DataType {
				tensor, err = ort.CreateTensor(input.Shape, data)
			}
		case []bool:
			dtype = ort.TensorElementDataTypeBool
			if dtype == meta.DataType {
				tensor, err = ort.CreateTensor(input.Shape, data)
			}
		}
		if dtype != meta.DataType {
			return nil, fmt.Errorf("input %q data type %v does not match graph data type %v", meta.Name, dtype, meta.DataType)
		}
		if err != nil {
			return nil, fmt.Errorf("input %q: %w", meta.Name, err)
		}
		allocated[meta.Name] = tensor
	}
	outputs, err = m.Session.session.Run(ctx, allocated, m.Session.outputNames)
	if err != nil {
		return nil, err
	}
	result = make(map[string]Tensor, len(outputs))
	for name, output := range outputs {
		if output == nil {
			return nil, fmt.Errorf("output %q is nil", name)
		}
		var data any
		switch output.DataType() {
		case ort.TensorElementDataTypeFloat32:
			var flat []float32
			flat, err = ort.TensorData[float32](output)
			data = slices.Clone(flat)
		case ort.TensorElementDataTypeInt64:
			var flat []int64
			flat, err = ort.TensorData[int64](output)
			data = slices.Clone(flat)
		case ort.TensorElementDataTypeBool:
			var flat []bool
			flat, err = ort.TensorData[bool](output)
			data = slices.Clone(flat)
		default:
			return nil, fmt.Errorf("output %q has unsupported data type %v", name, output.DataType())
		}
		if err != nil {
			return nil, fmt.Errorf("output %q: %w", name, err)
		}
		result[name] = Tensor{Shape: output.Shape(), Data: data}
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	return result, nil
}
