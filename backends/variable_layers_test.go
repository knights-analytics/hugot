package backends

import (
	"testing"

	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/stretchr/testify/require"
)

func TestPlaceVariables(t *testing.T) {
	graph := &protos.GraphProto{
		Initializer: []*protos.TensorProto{
			{Name: "embeddings.word_embeddings.weight"},
			{Name: "onnx::MatMul_1"}, // layer 0 query, anonymous
			{Name: "onnx::MatMul_2"}, // layer 1 query, anonymous
			{Name: "onnx::MatMul_3"}, // tied: read in layers 0 and 1
			{Name: "classifier.weight"},
		},
		Node: []*protos.NodeProto{
			{Name: "/embeddings/word_embeddings/Gather", OpType: "Gather", Input: []string{"embeddings.word_embeddings.weight", "input_ids"}},
			{Name: "/encoder/layer.0/attention/self/query/MatMul", OpType: "MatMul", Input: []string{"/embeddings/LayerNorm/Add_1_output_0", "onnx::MatMul_1"}},
			{Name: "/encoder/layer.1/attention/self/query/MatMul", OpType: "MatMul", Input: []string{"/encoder/layer.0/output/LayerNorm/Add_1_output_0", "onnx::MatMul_2"}},
			{Name: "/encoder/layer.0/shared/MatMul", OpType: "MatMul", Input: []string{"x", "onnx::MatMul_3"}},
			{Name: "/encoder/layer.1/shared/MatMul", OpType: "MatMul", Input: []string{"y", "onnx::MatMul_3"}},
			{Name: "/classifier/MatMul", OpType: "MatMul", Input: []string{"pooled", "classifier.weight"}},
		},
	}
	p := PlaceVariables(graph)
	require.Equal(t, 0, p.Layers["onnx::MatMul_1"])
	require.Equal(t, 1, p.Layers["onnx::MatMul_2"])
	require.NotContains(t, p.Layers, "onnx::MatMul_3", "a tied weight has no single layer")
	require.NotContains(t, p.Layers, "classifier.weight")
	require.True(t, p.Embeddings["embeddings.word_embeddings.weight"])
	require.False(t, p.Embeddings["classifier.weight"])
	// A fused QKV weight is placed by the layer of the projections reading its input — layer 1
	// here, although its input comes out of layer 0.
	require.Equal(t, 1, p.Layers["__fused_wQKV_|encoder|layer.0|output|LayerNorm|Add_1_output_0"])
	require.Equal(t, 0, p.Layers["__fused_wQKV_|embeddings|LayerNorm|Add_1_output_0"])
}
