package backends

import (
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	mlmodel "github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/onnx-gomlx/onnx/parser"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"
)

// The test encoder embeds a vocabulary of 4 tokens in 3 dimensions: its "hidden states" are the
// rows of this table.
var testEmbeddings = [][]float32{
	{1, 0, 0},
	{0, 2, 0},
	{0, 0, 3},
	{1, 1, 1},
}

func floatTensor(name string, dims []int64, values []float32) *protos.TensorProto {
	t := &protos.TensorProto{Name: name}
	setTensorFloats(t, dims, values)
	return t
}

func tensorValue(name string, elemType protos.TensorProto_DataType, dims ...any) *protos.ValueInfoProto {
	shape := &protos.TensorShapeProto{}
	for _, d := range dims {
		switch d := d.(type) {
		case int:
			shape.Dim = append(shape.Dim, &protos.TensorShapeProto_Dimension{Value: &protos.TensorShapeProto_Dimension_DimValue{DimValue: int64(d)}})
		case string:
			shape.Dim = append(shape.Dim, &protos.TensorShapeProto_Dimension{Value: &protos.TensorShapeProto_Dimension_DimParam{DimParam: d}})
		}
	}
	return &protos.ValueInfoProto{Name: name, Type: &protos.TypeProto{Value: &protos.TypeProto_TensorType{
		TensorType: &protos.TypeProto_Tensor{ElemType: int32(elemType), Shape: shape},
	}}}
}

// testEncoder is a model mapping input_ids [batch, sequence] to last_hidden_state
// [batch, sequence, 3] by an embedding lookup; attention_mask is an input it does not use.
func testEncoder(opset int64) *protos.ModelProto {
	var flat []float32
	for _, row := range testEmbeddings {
		flat = append(flat, row...)
	}
	return &protos.ModelProto{
		IrVersion:   7,
		OpsetImport: []*protos.OperatorSetIdProto{{Version: opset}},
		Graph: &protos.GraphProto{
			Name: "encoder",
			Node: []*protos.NodeProto{{
				Name: "embed", OpType: "Gather",
				Input: []string{"embeddings.weight", "input_ids"}, Output: []string{"last_hidden_state"},
			}},
			Initializer: []*protos.TensorProto{floatTensor("embeddings.weight", []int64{4, 3}, flat)},
			Input: []*protos.ValueInfoProto{
				tensorValue("input_ids", protos.TensorProto_INT64, "batch", "sequence"),
				tensorValue("attention_mask", protos.TensorProto_INT64, "batch", "sequence"),
			},
			Output: []*protos.ValueInfoProto{tensorValue("last_hidden_state", protos.TensorProto_FLOAT, "batch", "sequence", 3)},
		},
	}
}

// withTokenHead adds a token classification head the way the Hugging Face exports do (see
// distilbert-NER): a MatMul with a [hidden, labels] weight, then an Add with the bias first.
func withTokenHead(m *protos.ModelProto, weights []float32, biases []float32) *protos.ModelProto {
	g := m.Graph
	labels := len(biases)
	g.Node = append(g.Node,
		&protos.NodeProto{Name: "head/MatMul", OpType: "MatMul", Input: []string{"last_hidden_state", "onnx::MatMul_1"}, Output: []string{"/classifier/MatMul_output_0"}},
		&protos.NodeProto{Name: "head/Add", OpType: "Add", Input: []string{"classifier.bias", "/classifier/MatMul_output_0"}, Output: []string{"logits"}},
	)
	g.Initializer = append(g.Initializer,
		floatTensor("onnx::MatMul_1", []int64{3, int64(labels)}, weights),
		floatTensor("classifier.bias", []int64{int64(labels)}, biases))
	g.Output = []*protos.ValueInfoProto{tensorValue("logits", protos.TensorProto_FLOAT, "batch", "sequence", labels)}
	return m
}

// withSequenceHead adds a sequence classification head on the first token, as the SST-2 export
// does: a Gather of token 0, then a Gemm with a transposed [labels, hidden] weight.
func withSequenceHead(m *protos.ModelProto, weights []float32, biases []float32) *protos.ModelProto {
	g := m.Graph
	labels := len(biases)
	g.Node = append(g.Node,
		&protos.NodeProto{Name: "first", OpType: "Constant", Output: []string{"zero"}, Attribute: []*protos.AttributeProto{{
			Name: "value", Type: protos.AttributeProto_TENSOR, T: &protos.TensorProto{DataType: int32(protos.TensorProto_INT64), Int64Data: []int64{0}},
		}}},
		&protos.NodeProto{Name: "cls", OpType: "Gather", Input: []string{"last_hidden_state", "zero"}, Output: []string{"cls"},
			Attribute: []*protos.AttributeProto{{Name: "axis", Type: protos.AttributeProto_INT, I: 1}}},
		&protos.NodeProto{Name: "head", OpType: "Gemm", Input: []string{"cls", "classifier.weight", "classifier.bias"}, Output: []string{"logits"},
			Attribute: []*protos.AttributeProto{{Name: "transB", Type: protos.AttributeProto_INT, I: 1}}},
	)
	g.Initializer = append(g.Initializer,
		floatTensor("classifier.weight", []int64{int64(labels), 3}, weights),
		floatTensor("classifier.bias", []int64{int64(labels)}, biases))
	g.Output = []*protos.ValueInfoProto{tensorValue("logits", protos.TensorProto_FLOAT, "batch", labels)}
	return m
}

// runTestModel round-trips the model through its serialized form, as loading does, and runs its
// first output on the Go backend. It returns the output's flat values and shape.
func runTestModel(t *testing.T, m *protos.ModelProto, ids, mask [][]int64) ([]float32, []int) {
	t.Helper()
	data, err := proto.Marshal(m)
	require.NoError(t, err)
	parsed, err := parser.Parse(data)
	require.NoError(t, err)
	store := mlmodel.NewStore()
	require.NoError(t, parsed.VariablesToScope(store.RootScope()))
	backend, err := compute.NewWithConfig("go")
	require.NoError(t, err)
	defer backend.Finalize()

	outputNames, _ := parsed.Outputs()
	exec, err := mlmodel.NewExec(backend, store, func(scope *mlmodel.Scope, inputs []*graph.Node) []*graph.Node {
		return parsed.CallGraph(scope, inputs[0].Graph(), map[string]*graph.Node{"input_ids": inputs[0], "attention_mask": inputs[1]}, outputNames[0])
	})
	require.NoError(t, err)
	defer exec.Finalize()

	flatten := func(rows [][]int64) []int64 {
		var flat []int64
		for _, row := range rows {
			flat = append(flat, row...)
		}
		return flat
	}
	outputs, err := exec.Exec([]*tensors.Tensor{
		tensors.FromFlatDataAndDimensions(flatten(ids), len(ids), len(ids[0])),
		tensors.FromFlatDataAndDimensions(flatten(mask), len(mask), len(mask[0])),
	})
	require.NoError(t, err)
	out := outputs[0]
	var flat []float32
	tensors.ConstFlatData(out, func(values []float32) { flat = append(flat, values...) })
	return flat, out.Shape().Dimensions
}

func initializerFloats(t *testing.T, m *protos.ModelProto, name string) ([]float32, []int64) {
	t.Helper()
	for _, init := range m.Graph.Initializer {
		if init.Name == name {
			values, err := tensorFloats(init)
			require.NoError(t, err)
			return values, init.Dims
		}
	}
	t.Fatalf("no initializer %q", name)
	return nil, nil
}

func TestSetClassifierLabelsResizesTokenHead(t *testing.T) {
	// Labels O, B-PER, I-PER: label j scores hidden unit j, plus a bias of 10*j.
	original := withTokenHead(testEncoder(11),
		[]float32{
			1, 0, 0,
			0, 1, 0,
			0, 0, 1,
		}, []float32{0, 10, 20})
	ids, mask := [][]int64{{0, 1, 2}}, [][]int64{{1, 1, 1}}
	before, _ := runTestModel(t, original, ids, mask)

	resized := proto.Clone(original).(*protos.ModelProto)
	change, err := SetClassifierLabels(resized, TokenClassification,
		map[int]string{0: "O", 1: "B-PER", 2: "I-PER"},
		[]string{"O", "B-PER", "I-PER", "B-CITY", "I-CITY"})
	require.NoError(t, err)
	require.Equal(t, ClassifierHeadChange{Kept: []string{"O", "B-PER", "I-PER"}, Added: []string{"B-CITY", "I-CITY"}}, change)

	after, shape := runTestModel(t, resized, ids, mask)
	require.Equal(t, []int{1, 3, 5}, shape)
	// The known labels score exactly as before; the new ones score with their fresh weights.
	weights, dims := initializerFloats(t, resized, "onnx::MatMul_1")
	require.Equal(t, []int64{3, 5}, dims)
	for token := range 3 {
		for label := range 3 {
			require.InDelta(t, before[token*3+label], after[token*5+label], 1e-6)
		}
		for label := 3; label < 5; label++ {
			var want float32
			for h := range 3 {
				want += testEmbeddings[token][h] * weights[h*5+label]
			}
			require.InDelta(t, want, after[token*5+label], 1e-6, "a new label has a zero bias")
			require.NotZero(t, after[token*5+label], "a new label's weights must not start at zero")
		}
	}
}

func TestSetClassifierLabelsReordersAndDropsGemmHead(t *testing.T) {
	original := withSequenceHead(testEncoder(14),
		[]float32{
			1, 0, 0, // NEGATIVE
			0, 1, 0, // POSITIVE
		}, []float32{-1, 1})
	ids, mask := [][]int64{{1, 0}, {2, 0}}, [][]int64{{1, 1}, {1, 1}}
	before, _ := runTestModel(t, original, ids, mask)

	resized := proto.Clone(original).(*protos.ModelProto)
	change, err := SetClassifierLabels(resized, SequenceClassification,
		map[int]string{0: "NEGATIVE", 1: "POSITIVE"},
		[]string{"POSITIVE", "NEUTRAL"})
	require.NoError(t, err)
	require.Equal(t, ClassifierHeadChange{Kept: []string{"POSITIVE"}, Added: []string{"NEUTRAL"}, Removed: []string{"NEGATIVE"}}, change)

	after, shape := runTestModel(t, resized, ids, mask)
	require.Equal(t, []int{2, 2}, shape)
	for row := range 2 {
		require.InDelta(t, before[row*2+1], after[row*2], 1e-6, "POSITIVE moves to index 0 with its weights")
	}
	_, dims := initializerFloats(t, resized, "classifier.weight")
	require.Equal(t, []int64{2, 3}, dims, "the Gemm weight stays transposed")
}

func TestSetClassifierLabelsIgnoresUnrelatedID2Label(t *testing.T) {
	// A base model's config often carries the LABEL_0/LABEL_1 default, which says nothing about a
	// three-label head: no row can be matched, so every label starts fresh.
	m := withTokenHead(testEncoder(11), make([]float32, 9), []float32{0, 0, 0})
	change, err := SetClassifierLabels(m, TokenClassification, map[int]string{0: "LABEL_0", 1: "LABEL_1"}, []string{"O", "B-X"})
	require.NoError(t, err)
	require.False(t, change.Created)
	require.Equal(t, []string{"O", "B-X"}, change.Added)
	require.Empty(t, change.Kept)
}

func TestSetClassifierLabelsAddsTokenHead(t *testing.T) {
	m := testEncoder(14)
	change, err := SetClassifierLabels(m, TokenClassification, nil, []string{"O", "B-X", "I-X"})
	require.NoError(t, err)
	require.Equal(t, ClassifierHeadChange{Created: true, Added: []string{"O", "B-X", "I-X"}}, change)
	require.Len(t, m.Graph.Output, 1)
	require.Equal(t, "logits", m.Graph.Output[0].Name)

	ids, mask := [][]int64{{0, 3}}, [][]int64{{1, 1}}
	logits, shape := runTestModel(t, m, ids, mask)
	require.Equal(t, []int{1, 2, 3}, shape)
	weights, _ := initializerFloats(t, m, "classifier.weight")
	for token, id := range ids[0] {
		for label := range 3 {
			var want float32
			for h := range 3 {
				want += testEmbeddings[id][h] * weights[h*3+label]
			}
			require.InDelta(t, want, logits[token*3+label], 1e-6)
		}
	}
}

func TestSetClassifierLabelsAddsPooledSequenceHead(t *testing.T) {
	// ReduceSum takes its axes as an attribute before opset 13 and as an input after.
	for _, opset := range []int64{11, 14} {
		m := testEncoder(opset)
		change, err := SetClassifierLabels(m, SequenceClassification, nil, []string{"a", "b"})
		require.NoError(t, err)
		require.True(t, change.Created)

		// The second row's last token is padding and must not count in the mean.
		ids, mask := [][]int64{{0, 1, 2}, {3, 0, 2}}, [][]int64{{1, 1, 1}, {1, 1, 0}}
		logits, shape := runTestModel(t, m, ids, mask)
		require.Equal(t, []int{2, 2}, shape, "opset %d", opset)
		weights, _ := initializerFloats(t, m, "classifier.weight")
		for row := range 2 {
			var pooled [3]float32
			var count float32
			for token, id := range ids[row] {
				if mask[row][token] == 0 {
					continue
				}
				count++
				for h := range 3 {
					pooled[h] += testEmbeddings[id][h]
				}
			}
			for label := range 2 {
				var want float32
				for h := range 3 {
					want += pooled[h] / count * weights[h*2+label]
				}
				require.InDelta(t, want, logits[row*2+label], 1e-5, "opset %d, row %d", opset, row)
			}
		}
	}
}

func TestSetClassifierLabelsRejects(t *testing.T) {
	_, err := SetClassifierLabels(testEncoder(14), TokenClassification, nil, []string{"O", "O"})
	require.EqualError(t, err, `label "O" is given more than once`)
	_, err = SetClassifierLabels(testEncoder(14), TokenClassification, nil, nil)
	require.EqualError(t, err, "at least one label is required")

	// A sentence embedding has no tokens to put a token classification head on.
	m := testEncoder(14)
	m.Graph.Output = []*protos.ValueInfoProto{tensorValue("last_hidden_state", protos.TensorProto_FLOAT, "batch", 3)}
	_, err = SetClassifierLabels(m, TokenClassification, nil, []string{"O"})
	require.ErrorContains(t, err, "neither a token classification head nor")

	noMask := testEncoder(14)
	noMask.Graph.Input = noMask.Graph.Input[:1]
	_, err = SetClassifierLabels(noMask, SequenceClassification, nil, []string{"a"})
	require.ErrorContains(t, err, "needs an attention_mask input")
}
