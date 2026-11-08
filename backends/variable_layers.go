package backends

import (
	"regexp"
	"strconv"
	"strings"

	"github.com/gomlx/compute-onnx/support/protos"
)

// VariablePlacement says where in the transformer each weight of an ONNX graph sits. Keys are
// GoMLX variable names, i.e. ONNX initializer names with "/" replaced by "|".
//
// Exporters name many weights anonymously: PyTorch stores a transposed linear weight as
// "onnx::MatMul_977", which says nothing about its layer. The node that consumes it does
// ("/encoder/layer.0/attention/self/query/MatMul"), so the placement is read from the consumers.
type VariablePlacement struct {
	Layers     map[string]int  // transformer layer index of each weight
	Embeddings map[string]bool // weights of the embeddings block
}

var layerPattern = regexp.MustCompile(`layer\.(\d+)`)

// PlaceVariables reads the layer and embeddings placement of every initializer from the nodes
// that consume it. A weight used in more than one layer (tied weights) gets no layer. It also
// places the fused QKV weights onnx-gomlx creates at load time ("__fused_wQKV_<input>"), which
// are not in the graph yet, by the layer of the projections that read <input>.
func PlaceVariables(graph *protos.GraphProto) VariablePlacement {
	placement := VariablePlacement{Layers: map[string]int{}, Embeddings: map[string]bool{}}
	initializers := make(map[string]bool, len(graph.Initializer))
	for _, init := range graph.Initializer {
		initializers[init.Name] = true
	}
	ambiguous := map[string]bool{}
	setLayer := func(name string, layer int) {
		if ambiguous[name] {
			return
		}
		if previous, ok := placement.Layers[name]; ok && previous != layer {
			delete(placement.Layers, name)
			ambiguous[name] = true
			return
		}
		placement.Layers[name] = layer
	}
	for _, node := range graph.Node {
		layer, hasLayer := nodeLayer(node)
		embedding := strings.Contains(node.Name, "/embeddings/") || strings.HasPrefix(node.Name, "embeddings/")
		for i, input := range node.Input {
			if initializers[input] {
				name := gomlxVariableName(input)
				if hasLayer {
					setLayer(name, layer)
				}
				if embedding {
					placement.Embeddings[name] = true
				}
			} else if i == 0 && hasLayer && node.OpType == "MatMul" {
				setLayer(gomlxVariableName("__fused_wQKV_"+input), layer)
			}
		}
	}
	return placement
}

// nodeLayer reads the transformer layer index from a node's name.
func nodeLayer(node *protos.NodeProto) (int, bool) {
	matches := layerPattern.FindStringSubmatch(node.Name)
	if matches == nil {
		return 0, false
	}
	layer, err := strconv.Atoi(matches[1])
	if err != nil {
		return 0, false
	}
	return layer, true
}

// gomlxVariableName mirrors onnx-gomlx's SafeVarName, which is internal to that module.
func gomlxVariableName(onnxName string) string {
	return strings.ReplaceAll(onnxName, "/", "|")
}
