//go:build cgo && (ORT || ALL)

package backends

import (
	"context"
	"sync"
	"testing"

	"github.com/knights-analytics/hugot/options"
	"github.com/stretchr/testify/require"
)

// ONNX element types used by the test graphs (onnx.TensorProto.DataType).
const (
	onnxFloat  = 1
	onnxString = 8
)

type onnxTestValue struct {
	name     string
	elemType int
	dims     []any // int64 for a fixed size, string for a symbolic one
}

// buildIdentityONNX encodes a model that passes every input through an
// Identity node to the output named "<input>_out". The protobuf is written by
// hand so the tests need neither a model download nor an onnx dependency.
func buildIdentityONNX(values ...onnxTestValue) []byte {
	valueInfo := func(name string, v onnxTestValue) []byte {
		var shape []byte
		for _, d := range v.dims {
			switch d := d.(type) {
			case int64:
				shape = pbBytes(shape, 1, pbVarint(nil, 1, uint64(d)))
			case string:
				shape = pbBytes(shape, 1, pbBytes(nil, 2, []byte(d)))
			}
		}
		tensorType := pbBytes(pbVarint(nil, 1, uint64(v.elemType)), 2, shape)
		return pbBytes(pbBytes(nil, 1, []byte(name)), 2, pbBytes(nil, 1, tensorType))
	}
	graph := pbBytes(nil, 2, []byte("identity"))
	for _, v := range values {
		node := pbBytes(nil, 1, []byte(v.name))
		node = pbBytes(node, 2, []byte(v.name+"_out"))
		node = pbBytes(node, 4, []byte("Identity"))
		graph = pbBytes(graph, 1, node)
	}
	for _, v := range values {
		graph = pbBytes(graph, 11, valueInfo(v.name, v))
	}
	for _, v := range values {
		graph = pbBytes(graph, 12, valueInfo(v.name+"_out", v))
	}
	model := pbVarint(nil, 1, 8) // ir_version
	model = pbBytes(model, 7, graph)
	return pbBytes(model, 8, pbVarint(nil, 2, 17)) // opset_import: default domain, opset 17
}

func pbVarint(b []byte, field int, v uint64) []byte {
	b = appendUvarint(b, uint64(field)<<3)
	return appendUvarint(b, v)
}

func pbBytes(b []byte, field int, data []byte) []byte {
	b = appendUvarint(b, uint64(field)<<3|2)
	b = appendUvarint(b, uint64(len(data)))
	return append(b, data...)
}

func appendUvarint(b []byte, v uint64) []byte {
	for v >= 0x80 {
		b = append(b, byte(v)|0x80)
		v >>= 7
	}
	return append(b, byte(v))
}

var (
	ortTestOnce    sync.Once
	ortTestOptions *options.Options
	ortTestErr     error
)

// newIdentityORTModel loads the identity graph into an ORT session. The ORT
// library is located the usual way (ONNXRUNTIME_DIR or the default path).
func newIdentityORTModel(t *testing.T, values ...onnxTestValue) *Model {
	t.Helper()
	ortTestOnce.Do(func() {
		ortTestOptions = options.Defaults()
		ortTestOptions.Backend = options.BackendORT
		ortTestErr = initializeCoreORT(ortTestOptions)
	})
	require.NoError(t, ortTestErr)

	sessionOptions := ortTestOptions.BackendOptions.(*coreORTOptions)
	session, inputs, outputs, err := newCoreORTSession("", buildIdentityONNX(values...), sessionOptions)
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, session.Close()) })
	return &Model{
		ORTModel:    &ORTModel{Session: session, SessionOptions: sessionOptions},
		InputsMeta:  inputs,
		OutputsMeta: outputs,
	}
}

func TestRunTensorsORTStrings(t *testing.T) {
	model := newIdentityORTModel(t,
		onnxTestValue{name: "text", elemType: onnxString, dims: []any{"batch", int64(2)}},
		onnxTestValue{name: "score", elemType: onnxFloat, dims: []any{"batch"}},
	)
	text := []string{"hello", "", "ünïcode ✓", "with space"}

	outputs, err := model.RunTensors(t.Context(), map[string]Tensor{
		"text":  {Shape: []int64{2, 2}, Data: text},
		"score": {Shape: []int64{2}, Data: []float32{0.5, 1.5}},
	})
	require.NoError(t, err)

	require.Equal(t, Tensor{Shape: []int64{2, 2}, Data: text}, outputs["text_out"])
	require.Equal(t, Tensor{Shape: []int64{2}, Data: []float32{0.5, 1.5}}, outputs["score_out"])
}

func TestRunTensorsORTRejectsStringTypeMismatch(t *testing.T) {
	model := newIdentityORTModel(t, onnxTestValue{name: "text", elemType: onnxString, dims: []any{"batch"}})

	_, err := model.RunTensors(t.Context(), map[string]Tensor{
		"text": {Shape: []int64{1}, Data: []float32{1}},
	})
	require.EqualError(t, err, `input "text" expects string data, got float32`)

	model = newIdentityORTModel(t, onnxTestValue{name: "score", elemType: onnxFloat, dims: []any{"batch"}})
	_, err = model.RunTensors(t.Context(), map[string]Tensor{
		"score": {Shape: []int64{1}, Data: []string{"1"}},
	})
	require.EqualError(t, err, `input "score" expects float32 data, got string`)
}

func TestORTMetadataReportsDataTypes(t *testing.T) {
	model := newIdentityORTModel(t,
		onnxTestValue{name: "text", elemType: onnxString, dims: []any{"batch"}},
		onnxTestValue{name: "score", elemType: onnxFloat, dims: []any{"batch", int64(3)}},
	)

	require.Equal(t, []InputOutputInfo{
		{Name: "text", Dimensions: Shape{-1}, DataType: DataTypeString},
		{Name: "score", Dimensions: Shape{-1, 3}, DataType: DataTypeFloat32},
	}, model.InputsMeta)
	require.Equal(t, []InputOutputInfo{
		{Name: "text_out", Dimensions: Shape{-1}, DataType: DataTypeString},
		{Name: "score_out", Dimensions: Shape{-1, 3}, DataType: DataTypeFloat32},
	}, model.OutputsMeta)
}

func TestRunTensorsORTRejectsNULInStrings(t *testing.T) {
	model := newIdentityORTModel(t, onnxTestValue{name: "text", elemType: onnxString, dims: []any{"batch"}})

	_, err := model.RunTensors(t.Context(), map[string]Tensor{
		"text": {Shape: []int64{2}, Data: []string{"fine", "bad\x00byte"}},
	})
	require.ErrorContains(t, err, `input "text"`)
	require.ErrorContains(t, err, "NUL byte")
}

func TestRunTensorsORTEmptyStringTensor(t *testing.T) {
	model := newIdentityORTModel(t, onnxTestValue{name: "text", elemType: onnxString, dims: []any{"batch"}})

	outputs, err := model.RunTensors(t.Context(), map[string]Tensor{
		"text": {Shape: []int64{0}, Data: []string{}},
	})
	require.NoError(t, err)
	require.Equal(t, Tensor{Shape: []int64{0}, Data: []string{}}, outputs["text_out"])
}

func TestRunORTSessionOnBatchStringOutputs(t *testing.T) {
	model := newIdentityORTModel(t,
		onnxTestValue{name: "label", elemType: onnxString, dims: []any{"batch"}},
		onnxTestValue{name: "tokens", elemType: onnxString, dims: []any{"batch", int64(3)}},
	)
	labels, err := newCoreStringTensor([]int64{2}, []string{"setosa", "virginica"})
	require.NoError(t, err)
	tokens, err := newCoreStringTensor([]int64{2, 3}, []string{"a", "b", "c", "d", "", "f"})
	require.NoError(t, err)
	t.Cleanup(func() {
		require.NoError(t, labels.Close())
		require.NoError(t, tokens.Close())
	})

	batch := NewBatch(2)
	batch.InputValues = []*coreORTTensor{labels, tokens}
	pipeline := &BasePipeline{SessionContext: context.Background(), Model: model}
	require.NoError(t, runORTSessionOnBatch(t.Context(), batch, pipeline))

	require.Equal(t, []any{
		[]string{"setosa", "virginica"},
		[][]string{{"a", "b", "c"}, {"d", "", "f"}},
	}, batch.OutputValues)
}

func TestCoreORTTensorStringData(t *testing.T) {
	newIdentityORTModel(t, onnxTestValue{name: "text", elemType: onnxString, dims: []any{"batch"}}) // initialises ORT

	tensor, err := newCoreStringTensor([]int64{2}, []string{"x", "y"})
	require.NoError(t, err)
	defer func() { require.NoError(t, tensor.Close()) }()
	values, err := tensor.StringData()
	require.NoError(t, err)
	require.Equal(t, []string{"x", "y"}, values)
	_, err = tensor.Float32Data()
	require.Error(t, err, "a string tensor must not read as float32")

	numbers, err := newCoreFloat32Tensor([]int64{1}, []float32{1})
	require.NoError(t, err)
	defer func() { require.NoError(t, numbers.Close()) }()
	_, err = numbers.StringData()
	require.Error(t, err, "a float32 tensor must not read as strings")

	_, err = (*coreORTTensor)(nil).StringData()
	require.EqualError(t, err, "ORT tensor is not initialized")
}
