package hugot

import (
	"testing"

	"github.com/gomlx/gomlx/core/tensors"
	mlmodel "github.com/gomlx/gomlx/ml/model"
	"github.com/stretchr/testify/require"

	"github.com/knights-analytics/hugot/backends"
)

// newFreezeTestStore builds a store holding one scalar variable per name.
func newFreezeTestStore(t *testing.T, names []string) *mlmodel.Store {
	t.Helper()
	store := mlmodel.NewStore()
	scope := store.RootScope().At("onnx")
	for _, name := range names {
		_, err := scope.CreateVariable(name, tensors.FromValue([]float32{1}))
		require.NoError(t, err)
	}
	return store
}

// Embedding variables are frozen whether or not the export prefixes them with the architecture name.
func TestFreezeVariablesEmbeddings(t *testing.T) {
	names := map[string]bool{ // name -> trainable after freezing with [-1]
		"embeddings.LayerNorm.bias":                        false,
		"distilbert.embeddings.word_embeddings.weight":     false,
		"distilbert.embeddings.position_embeddings.weight": false,
		"distilbert.transformer.layer.0.ffn.lin1.weight":   false,
		"encoder.layer.5.output.dense.weight":              true, // the last layer stays trainable
		"classifier.weight":                                true,
		"pre_classifier.bias":                              true,
	}
	var list []string
	for name := range names {
		list = append(list, name)
	}
	store := newFreezeTestStore(t, list)
	summary, err := freezeVariables(store, trainerSettings{freezeLayers: []int{-1}}, backends.VariablePlacement{})
	require.NoError(t, err)
	for v := range store.IterVariables() {
		require.Equal(t, names[v.Name()], v.Trainable, v.Name())
	}
	require.Equal(t, 4, summary.frozen)
	require.ElementsMatch(t, []string{"encoder.layer.5.output.dense.weight", "classifier.weight", "pre_classifier.bias"}, summary.trainable)
}

// A layer index past the model's last layer is an error, not a silent no-op.
func TestFreezeVariablesLayerOutOfRange(t *testing.T) {
	store := newFreezeTestStore(t, []string{
		"encoder.layer.0.output.dense.weight",
		"encoder.layer.1.output.dense.weight",
		"classifier.weight",
	})
	_, err := freezeVariables(store, trainerSettings{freezeLayers: []int{0, 2}}, backends.VariablePlacement{})
	require.ErrorContains(t, err, "cannot freeze layer 2")

	_, err = freezeVariables(store, trainerSettings{freezeLayers: []int{1}}, backends.VariablePlacement{})
	require.NoError(t, err)
}

func TestWithFreezeLayersValidation(t *testing.T) {
	for _, layers := range [][]int{{-2}, {-1, 0}, {0, -1}} {
		require.Error(t, WithFreezeLayers(layers)(&trainerSettings{}), "%v", layers)
	}
	for _, layers := range [][]int{{-1}, {0, 3}, {}} {
		require.NoError(t, WithFreezeLayers(layers)(&trainerSettings{}), "%v", layers)
	}
}

func TestFreezeSummaryString(t *testing.T) {
	s := freezeSummary{frozen: 3, trainable: []string{"classifier.weight", "classifier.bias"}}
	require.Equal(t, "frozen 3 variables, training 2: classifier.weight, classifier.bias", s.String())
}

// Anonymously named weights are frozen through the placement read off the graph.
func TestFreezeVariablesAnonymousWeights(t *testing.T) {
	store := newFreezeTestStore(t, []string{
		"onnx::MatMul_977",                  // layer 0, by placement
		"onnx::MatMul_1000",                 // layer 1, by placement
		"embeddings.LayerNorm.weight",       // embeddings, by name
		"onnx::Gather_5",                    // embeddings, by placement
		"encoder.layer.1.output.dense.bias", // layer 1, by name
		"classifier.weight",
	})
	placement := backends.VariablePlacement{
		Layers:     map[string]int{"onnx::MatMul_977": 0, "onnx::MatMul_1000": 1},
		Embeddings: map[string]bool{"onnx::Gather_5": true},
	}

	summary, err := freezeVariables(store, trainerSettings{freezeLayers: []int{-1}}, placement)
	require.NoError(t, err)
	require.ElementsMatch(t, []string{"onnx::MatMul_1000", "encoder.layer.1.output.dense.bias", "classifier.weight"}, summary.trainable,
		"the last layer (1) stays trainable, found by name and by placement")

	store = newFreezeTestStore(t, []string{"onnx::MatMul_977", "onnx::MatMul_1000", "classifier.weight"})
	summary, err = freezeVariables(store, trainerSettings{freezeLayers: []int{1}}, placement)
	require.NoError(t, err)
	require.ElementsMatch(t, []string{"onnx::MatMul_977", "classifier.weight"}, summary.trainable)
}

// Freezing that leaves nothing to train is refused rather than running a no-op training.
func TestFreezeVariablesNothingLeft(t *testing.T) {
	store := newFreezeTestStore(t, []string{"encoder.layer.0.output.dense.weight", "encoder.layer.1.output.dense.weight"})
	_, err := freezeVariables(store, trainerSettings{freezeLayers: []int{0, 1}}, backends.VariablePlacement{})
	require.ErrorContains(t, err, "no variable to train")
}
