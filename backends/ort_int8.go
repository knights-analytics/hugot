//go:build cgo && (ORT || ALL)

package backends

import (
	"errors"
	"fmt"

	ort "github.com/yalue/onnxruntime_go"
)

// convertRankTwoInt8Tensor copies a completed-run ORT INT8 tensor into Go-owned [][]int8
// storage after validating the runtime shape contract.
//
// The tensor must be rank two with shape [batch, dimension], the runtime batch must equal
// expectedBatch, dimension must be positive, and the flattened data length must equal
// batch*dimension. Every coordinate is copied; the returned rows remain valid after the
// source ORT tensor is destroyed. No pooling, normalization, or widening is applied.
//
// tensor: auto-allocated ORT INT8 output from a completed DynamicAdvancedSession.Run.
// outputName: ONNX output name used in validation errors.
// expectedBatch: number of input texts that produced this ORT run.
//
// Returns one independently owned INT8 vector per batch row, or an actionable error.
func convertRankTwoInt8Tensor(tensor *ort.Tensor[int8], outputName string, expectedBatch int) ([][]int8, error) {
	if tensor == nil {
		return nil, fmt.Errorf("ORT output %q is nil; expected *ort.Tensor[int8] with shape [batch, dimension]", outputName)
	}
	shape := tensor.GetShape()
	if len(shape) != 2 {
		return nil, fmt.Errorf(
			"ORT output %q has rank %d and type int8; expected rank 2 shape [batch, dimension]",
			outputName,
			len(shape),
		)
	}
	runtimeBatch := int(shape[0])
	dimension := int(shape[1])
	if runtimeBatch != expectedBatch {
		return nil, fmt.Errorf(
			"ORT output %q batch size %d does not match input batch size %d",
			outputName,
			runtimeBatch,
			expectedBatch,
		)
	}
	if dimension <= 0 {
		return nil, fmt.Errorf("ORT output %q has invalid embedding dimension %d", outputName, dimension)
	}
	data := tensor.GetData()
	expectedLen := runtimeBatch * dimension
	if len(data) != expectedLen {
		return nil, fmt.Errorf(
			"ORT output %q flattened length %d does not equal batch*dimension %d (shape %s)",
			outputName,
			len(data),
			expectedLen,
			shape.String(),
		)
	}

	out := make([][]int8, runtimeBatch)
	for i := range runtimeBatch {
		row := make([]int8, dimension)
		copy(row, data[i*dimension:(i+1)*dimension])
		out[i] = row
	}
	return out, nil
}

// destroyORTOutputs destroys every non-nil auto-allocated ORT output value exactly once.
//
// Call this only after a DynamicAdvancedSession.Run has completed and returned control to
// Hugot. Early cancellation paths must not invoke it while ORT may still write the slots.
//
// outputs: completed-run ORT output slot slice; nil entries are skipped.
//
// Returns the joined Destroy errors, if any.
func destroyORTOutputs(outputs []ort.Value) error {
	var err error
	for i, value := range outputs {
		if value == nil {
			continue
		}
		err = errors.Join(err, value.Destroy())
		outputs[i] = nil
	}
	return err
}
