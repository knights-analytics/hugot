package backends

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"

	"github.com/gomlx/compute-onnx/support/protos"
)

// ClassifierTask says what a classification head predicts: one label per text, or one per token.
type ClassifierTask int

const (
	// SequenceClassification heads output [batch, labels] logits.
	SequenceClassification ClassifierTask = iota
	// TokenClassification heads output [batch, sequence, labels] logits.
	TokenClassification
)

// headInitStdDev is the standard deviation of the normal distribution new head weights are drawn
// from: Hugging Face's default initializer_range for BERT-style models.
const headInitStdDev = 0.02

// headInitSeed seeds the initialisation of new head weights, so that a training run is
// reproducible.
const headInitSeed = 42

// ClassifierHeadChange describes what SetClassifierLabels did to a model's graph.
type ClassifierHeadChange struct {
	// Created is true when the model had no classification head and a new one was added on top of
	// its encoder output; false when an existing head was resized.
	Created bool
	// Kept lists the labels whose weights were carried over from the existing head.
	Kept []string
	// Added lists the labels that start from freshly initialised weights.
	Added []string
	// Removed lists the labels of the existing head that are not in the new label set.
	Removed []string
}

// SetClassifierLabels rewrites a model's graph so that it classifies into exactly labels, in that
// order, and returns what it changed.
//
// When the model already ends in a classification head (a MatMul and Add, or a Gemm, producing
// the first output), the head is resized: a label that is also in oldLabels keeps its trained
// weights, a new label gets freshly initialised ones, and an old label not in labels is dropped.
// oldLabels is the model's current id2label map; it may be empty, in which case every label starts
// fresh.
//
// When the model has no head, which is the case for a plain encoder export whose output is the
// hidden states, a new linear head is added on top of them. For token classification it is applied
// to every token; for sequence classification the hidden states are first mean-pooled over the
// attention mask, as sentence-transformers does. The new head's output, named "logits", becomes the
// model's only output.
func SetClassifierLabels(model *protos.ModelProto, task ClassifierTask, oldLabels map[int]string, labels []string) (ClassifierHeadChange, error) {
	if err := validateLabels(labels); err != nil {
		return ClassifierHeadChange{}, err
	}
	if model == nil || model.Graph == nil {
		return ClassifierHeadChange{}, errors.New("the model has no graph")
	}
	if len(model.Graph.Output) == 0 {
		return ClassifierHeadChange{}, errors.New("the model has no outputs")
	}
	head, err := findClassifierHead(model.Graph, task)
	if err != nil {
		return ClassifierHeadChange{}, err
	}
	if head != nil {
		return resizeClassifierHead(model.Graph, head, oldLabels, labels)
	}
	return addClassifierHead(model, task, labels)
}

func validateLabels(labels []string) error {
	if len(labels) == 0 {
		return errors.New("at least one label is required")
	}
	seen := make(map[string]bool, len(labels))
	for _, label := range labels {
		if label == "" {
			return errors.New("labels must not be empty strings")
		}
		if seen[label] {
			return fmt.Errorf("label %q is given more than once", label)
		}
		seen[label] = true
	}
	return nil
}

// classifierHead is the final linear layer of a classification model.
type classifierHead struct {
	weight *protos.TensorProto
	bias   *protos.TensorProto
	// labelAxis is the weight axis indexed by label: 1 for a MatMul weight [hidden, labels], 0 for a
	// Gemm weight stored transposed, [labels, hidden].
	labelAxis int
	output    *protos.ValueInfoProto
	// intermediate names the MatMul output feeding the bias Add, when there is one.
	intermediate string
}

// findClassifierHead returns the linear layer that produces the model's first output, or nil when
// that output is not produced by one (a plain encoder, whose last op is a layer norm).
func findClassifierHead(g *protos.GraphProto, task ClassifierTask) (*classifierHead, error) {
	output := g.Output[0]
	wantRank := 2
	if task == TokenClassification {
		wantRank = 3
	}
	if dims := valueDims(output); dims != nil && len(dims) != wantRank {
		// Not a classifier output for this task, e.g. the hidden states of an encoder.
		return nil, nil
	}

	producers := make(map[string]*protos.NodeProto, len(g.Node))
	for _, node := range g.Node {
		for _, out := range node.Output {
			producers[out] = node
		}
	}
	initializers := make(map[string]*protos.TensorProto, len(g.Initializer))
	for _, t := range g.Initializer {
		initializers[t.Name] = t
	}

	node := producers[output.Name]
	if node == nil {
		return nil, nil
	}
	head := &classifierHead{output: output}
	switch node.OpType {
	case "Gemm":
		if len(node.Input) < 3 || attrInt(node, "transA", 0) != 0 {
			return nil, nil
		}
		head.weight, head.bias = initializers[node.Input[1]], initializers[node.Input[2]]
		head.labelAxis = 1
		if attrInt(node, "transB", 0) != 0 {
			head.labelAxis = 0
		}
	case "Add":
		if len(node.Input) != 2 {
			return nil, nil
		}
		for i, name := range node.Input {
			bias := initializers[name]
			if bias == nil || len(bias.Dims) != 1 {
				continue
			}
			matMul := producers[node.Input[1-i]]
			if matMul == nil || matMul.OpType != "MatMul" || len(matMul.Input) != 2 {
				continue
			}
			head.weight, head.bias = initializers[matMul.Input[1]], bias
			head.labelAxis = 1
			head.intermediate = matMul.Output[0]
			break
		}
	default:
		return nil, nil
	}
	if head.weight == nil || head.bias == nil || len(head.weight.Dims) != 2 || len(head.bias.Dims) != 1 {
		return nil, nil
	}
	if head.weight.Dims[head.labelAxis] != head.bias.Dims[0] {
		return nil, nil
	}
	return head, nil
}

// resizeClassifierHead rebuilds an existing head's weight and bias for the new labels.
func resizeClassifierHead(g *protos.GraphProto, head *classifierHead, oldLabels map[int]string, labels []string) (ClassifierHeadChange, error) {
	weights, err := tensorFloats(head.weight)
	if err != nil {
		return ClassifierHeadChange{}, err
	}
	biases, err := tensorFloats(head.bias)
	if err != nil {
		return ClassifierHeadChange{}, err
	}
	numOld := int(head.bias.Dims[0])
	hidden := int(head.weight.Dims[1-head.labelAxis])

	// An id2label map that does not describe this head (e.g. the LABEL_0/LABEL_1 default of a base
	// model's config) cannot say which row belongs to which label, so no row is carried over.
	oldIndex := map[string]int{}
	if len(oldLabels) == numOld {
		for i, label := range oldLabels {
			oldIndex[label] = i
		}
	}

	rng := rand.New(rand.NewPCG(headInitSeed, 0))
	numNew := len(labels)
	newWeights := make([]float32, hidden*numNew)
	newBiases := make([]float32, numNew)
	// weightAt returns the index of (label, hidden unit h) in a flat weight for numLabels labels.
	weightAt := func(label, h, numLabels int) int {
		if head.labelAxis == 0 {
			return label*hidden + h
		}
		return h*numLabels + label
	}
	var change ClassifierHeadChange
	for j, label := range labels {
		if i, ok := oldIndex[label]; ok {
			for h := range hidden {
				newWeights[weightAt(j, h, numNew)] = weights[weightAt(i, h, numOld)]
			}
			newBiases[j] = biases[i]
			change.Kept = append(change.Kept, label)
			continue
		}
		for h := range hidden {
			newWeights[weightAt(j, h, numNew)] = float32(rng.NormFloat64() * headInitStdDev)
		}
		change.Added = append(change.Added, label)
	}
	for i := range numOld {
		if label, ok := oldLabels[i]; ok && !slices.Contains(labels, label) {
			change.Removed = append(change.Removed, label)
		}
	}

	weightDims := slices.Clone(head.weight.Dims)
	weightDims[head.labelAxis] = int64(numNew)
	setTensorFloats(head.weight, weightDims, newWeights)
	setTensorFloats(head.bias, []int64{int64(numNew)}, newBiases)

	setLastDim(head.output, int64(numNew))
	// Shape hints for the old size would contradict the new weights; they are optional, so drop them.
	g.ValueInfo = slices.DeleteFunc(g.ValueInfo, func(v *protos.ValueInfoProto) bool {
		return v.Name == head.intermediate || v.Name == head.output.Name
	})
	return change, nil
}

// addClassifierHead adds a linear head on top of a plain encoder's hidden states.
func addClassifierHead(model *protos.ModelProto, task ClassifierTask, labels []string) (ClassifierHeadChange, error) {
	g := model.Graph
	source := encoderOutput(g, task)
	if source == nil {
		if task == TokenClassification {
			return ClassifierHeadChange{}, errors.New("the model has neither a token classification head nor a [batch, sequence, hidden] output to put one on")
		}
		return ClassifierHeadChange{}, errors.New("the model has neither a sequence classification head nor a hidden-state or embedding output to put one on")
	}
	sourceDims := valueDims(source)
	hidden := sourceDims[len(sourceDims)-1].GetDimValue()
	if hidden <= 0 {
		return ClassifierHeadChange{}, fmt.Errorf("the hidden size of output %q is not fixed in the graph", source.Name)
	}

	names := newNamer(g)
	features := source.Name
	if task == SequenceClassification && len(sourceDims) == 3 {
		pooled, err := addMeanPooling(model, names, features)
		if err != nil {
			return ClassifierHeadChange{}, err
		}
		features = pooled
	}

	numLabels := len(labels)
	rng := rand.New(rand.NewPCG(headInitSeed, 0))
	weights := make([]float32, int(hidden)*numLabels)
	for i := range weights {
		weights[i] = float32(rng.NormFloat64() * headInitStdDev)
	}
	weight := &protos.TensorProto{Name: names.unique("classifier.weight")}
	setTensorFloats(weight, []int64{hidden, int64(numLabels)}, weights)
	bias := &protos.TensorProto{Name: names.unique("classifier.bias")}
	setTensorFloats(bias, []int64{int64(numLabels)}, make([]float32, numLabels))
	g.Initializer = append(g.Initializer, weight, bias)

	projected := names.unique("/classifier/MatMul_output_0")
	logits := names.unique("logits")
	g.Node = append(g.Node,
		&protos.NodeProto{Name: names.unique("/classifier/MatMul"), OpType: "MatMul", Input: []string{features, weight.Name}, Output: []string{projected}},
		&protos.NodeProto{Name: names.unique("/classifier/Add"), OpType: "Add", Input: []string{projected, bias.Name}, Output: []string{logits}},
	)

	// The batch (and sequence) dimensions keep the encoder output's names.
	outDims := []*protos.TensorShapeProto_Dimension{sourceDims[0]}
	if task == TokenClassification {
		outDims = append(outDims, sourceDims[1])
	}
	outDims = append(outDims, &protos.TensorShapeProto_Dimension{Value: &protos.TensorShapeProto_Dimension_DimValue{DimValue: int64(numLabels)}})
	g.Output = []*protos.ValueInfoProto{{
		Name: logits,
		Type: &protos.TypeProto{Value: &protos.TypeProto_TensorType{TensorType: &protos.TypeProto_Tensor{
			ElemType: int32(protos.TensorProto_FLOAT),
			Shape:    &protos.TensorShapeProto{Dim: outDims},
		}}},
	}}
	return ClassifierHeadChange{Created: true, Added: slices.Clone(labels)}, nil
}

// encoderOutput picks the output a new head goes on: the token hidden states, or for sequence
// classification a sentence embedding when the model has no per-token output.
func encoderOutput(g *protos.GraphProto, task ClassifierTask) *protos.ValueInfoProto {
	isFloat := func(v *protos.ValueInfoProto) bool {
		return v.GetType().GetTensorType().GetElemType() == int32(protos.TensorProto_FLOAT)
	}
	byRank := func(rank int) *protos.ValueInfoProto {
		for _, preferred := range []string{"last_hidden_state", "token_embeddings", "sentence_embedding"} {
			for _, out := range g.Output {
				if out.Name == preferred && len(valueDims(out)) == rank && isFloat(out) {
					return out
				}
			}
		}
		for _, out := range g.Output {
			if len(valueDims(out)) == rank && isFloat(out) {
				return out
			}
		}
		return nil
	}
	if out := byRank(3); out != nil {
		return out
	}
	if task == SequenceClassification {
		return byRank(2)
	}
	return nil
}

// addMeanPooling appends nodes averaging hiddenStates [batch, sequence, hidden] over the
// attention mask, and returns the name of the pooled [batch, hidden] output.
func addMeanPooling(model *protos.ModelProto, names *namer, hiddenStates string) (string, error) {
	g := model.Graph
	mask := ""
	for _, in := range g.Input {
		if in.Name == "attention_mask" {
			mask = in.Name
		}
	}
	if mask == "" {
		return "", errors.New("mean pooling for a new sequence classification head needs an attention_mask input, and the model has none")
	}
	// ReduceSum takes its axes as an attribute before opset 13, and as an input from 13 on.
	axesAsInput := defaultOpset(model) >= 13

	node := func(op string, inputs []string, attrs ...*protos.AttributeProto) string {
		out := names.unique("/pooling/" + op + "_output_0")
		g.Node = append(g.Node, &protos.NodeProto{Name: names.unique("/pooling/" + op), OpType: op, Input: inputs, Output: []string{out}, Attribute: attrs})
		return out
	}
	constant := func(t *protos.TensorProto) string {
		return node("Constant", nil, &protos.AttributeProto{Name: "value", Type: protos.AttributeProto_TENSOR, T: t})
	}
	reduceSumSequence := func(x string) string {
		keepDims := &protos.AttributeProto{Name: "keepdims", Type: protos.AttributeProto_INT, I: 0}
		if axesAsInput {
			axes := constant(&protos.TensorProto{DataType: int32(protos.TensorProto_INT64), Dims: []int64{1}, Int64Data: []int64{1}})
			return node("ReduceSum", []string{x, axes}, keepDims)
		}
		return node("ReduceSum", []string{x}, keepDims, &protos.AttributeProto{Name: "axes", Type: protos.AttributeProto_INTS, Ints: []int64{1}})
	}

	// mask [batch, sequence] -> float [batch, sequence, 1]
	floatMask := node("Cast", []string{mask}, &protos.AttributeProto{Name: "to", Type: protos.AttributeProto_INT, I: int64(protos.TensorProto_FLOAT)})
	shape := constant(&protos.TensorProto{DataType: int32(protos.TensorProto_INT64), Dims: []int64{3}, Int64Data: []int64{0, -1, 1}})
	mask3 := node("Reshape", []string{floatMask, shape})
	summed := reduceSumSequence(node("Mul", []string{hiddenStates, mask3})) // [batch, hidden]
	counts := reduceSumSequence(mask3)                                      // [batch, 1]
	minCount := constant(&protos.TensorProto{DataType: int32(protos.TensorProto_FLOAT), Dims: []int64{}, FloatData: []float32{1e-9}})
	return node("Div", []string{summed, node("Max", []string{counts, minCount})}), nil
}

func defaultOpset(model *protos.ModelProto) int64 {
	for _, opset := range model.OpsetImport {
		if opset.Domain == "" || opset.Domain == "ai.onnx" {
			return opset.Version
		}
	}
	return 0
}

// namer hands out value and node names that do not clash with any already in the graph.
type namer struct {
	used map[string]bool
}

func newNamer(g *protos.GraphProto) *namer {
	n := &namer{used: map[string]bool{}}
	for _, node := range g.Node {
		n.used[node.Name] = true
		for _, out := range node.Output {
			n.used[out] = true
		}
	}
	for _, t := range g.Initializer {
		n.used[t.Name] = true
	}
	for _, v := range slices.Concat(g.Input, g.Output, g.ValueInfo) {
		n.used[v.Name] = true
	}
	return n
}

func (n *namer) unique(name string) string {
	candidate := name
	for i := 1; n.used[candidate]; i++ {
		candidate = fmt.Sprintf("%s_%d", name, i)
	}
	n.used[candidate] = true
	return candidate
}

func valueDims(v *protos.ValueInfoProto) []*protos.TensorShapeProto_Dimension {
	shape := v.GetType().GetTensorType().GetShape()
	if shape == nil {
		return nil
	}
	return shape.Dim
}

func setLastDim(v *protos.ValueInfoProto, size int64) {
	dims := valueDims(v)
	if len(dims) == 0 {
		return
	}
	dims[len(dims)-1] = &protos.TensorShapeProto_Dimension{Value: &protos.TensorShapeProto_Dimension_DimValue{DimValue: size}}
}

func attrInt(node *protos.NodeProto, name string, defaultValue int64) int64 {
	for _, attr := range node.Attribute {
		if attr.Name == name {
			return attr.I
		}
	}
	return defaultValue
}

// tensorFloats returns the values of a float32 initializer stored in the model file.
func tensorFloats(t *protos.TensorProto) ([]float32, error) {
	if t.DataLocation == protos.TensorProto_EXTERNAL {
		return nil, fmt.Errorf("initializer %q is stored as external data, which classifier head changes do not support", t.Name)
	}
	if t.DataType != int32(protos.TensorProto_FLOAT) {
		return nil, fmt.Errorf("initializer %q has ONNX data type %d; classifier head changes support float32 only", t.Name, t.DataType)
	}
	size := 1
	for _, d := range t.Dims {
		size *= int(d)
	}
	if t.RawData == nil {
		if len(t.FloatData) != size {
			return nil, fmt.Errorf("initializer %q holds %d values, expected %d", t.Name, len(t.FloatData), size)
		}
		return slices.Clone(t.FloatData), nil
	}
	if len(t.RawData) != 4*size {
		return nil, fmt.Errorf("initializer %q holds %d bytes, expected %d", t.Name, len(t.RawData), 4*size)
	}
	values := make([]float32, size)
	for i := range values {
		values[i] = math.Float32frombits(binary.LittleEndian.Uint32(t.RawData[4*i:]))
	}
	return values, nil
}

// setTensorFloats replaces a tensor's contents with float32 values, stored as raw data.
func setTensorFloats(t *protos.TensorProto, dims []int64, values []float32) {
	raw := make([]byte, 4*len(values))
	for i, v := range values {
		binary.LittleEndian.PutUint32(raw[4*i:], math.Float32bits(v))
	}
	t.Dims = dims
	t.DataType = int32(protos.TensorProto_FLOAT)
	t.RawData = raw
	t.FloatData = nil
}
