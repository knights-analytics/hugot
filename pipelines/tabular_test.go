package pipelines

import (
	"testing"

	"github.com/knights-analytics/hugot/backends"
	"github.com/stretchr/testify/require"
)

// newTwoOutputTabularClassifier mirrors a scikit-learn export with zipmap=False:
// a label output of shape (batch) and a probability output of shape (batch, classes).
func newTwoOutputTabularClassifier(labels map[int]string) *TabularPipeline {
	return &TabularPipeline{
		BasePipeline: &backends.BasePipeline{Model: &backends.Model{
			OutputsMeta: []backends.InputOutputInfo{
				{Name: "label", Dimensions: backends.NewShape(-1)},
				{Name: "probabilities", Dimensions: backends.NewShape(-1, 2)},
			},
		}},
		ProblemType: "classification",
		IDLabelMap:  labels,
	}
}

func TestTabularPostprocessStringClassLabels(t *testing.T) {
	p := newTwoOutputTabularClassifier(map[int]string{0: "class_0", 1: "class_1"})
	batch := backends.NewBatch(2)
	batch.OutputValues = []any{
		[]string{"cat", "dog"},
		[][]float32{{0.9, 0.1}, {0.2, 0.8}},
	}

	out, err := p.postprocess(batch)
	require.NoError(t, err)
	require.Equal(t, "cat", out.ClassificationResults[0].PredictedClass)
	require.Equal(t, "dog", out.ClassificationResults[1].PredictedClass)
	require.Equal(t, []ClassificationOutput{{Label: "class_0", Score: 0.2}, {Label: "class_1", Score: 0.8}},
		out.ClassificationResults[1].Probabilities)
}

func TestTabularPostprocessIntClassLabels(t *testing.T) {
	p := newTwoOutputTabularClassifier(map[int]string{0: "setosa", 1: "versicolor"})
	batch := backends.NewBatch(2)
	// A label outside the map (classes need not be 0..n-1) keeps its number
	// instead of becoming an empty string.
	batch.OutputValues = []any{
		[]int64{1, 5},
		[][]float32{{0.3, 0.7}, {0.6, 0.4}},
	}

	out, err := p.postprocess(batch)
	require.NoError(t, err)
	require.Equal(t, "versicolor", out.ClassificationResults[0].PredictedClass)
	require.Equal(t, "5", out.ClassificationResults[1].PredictedClass)
}

func TestTabularPostprocessRejectsBadClassLabels(t *testing.T) {
	p := newTwoOutputTabularClassifier(nil)
	batch := backends.NewBatch(2)
	batch.OutputValues = []any{[]float32{0, 1}, [][]float32{{1, 0}, {0, 1}}}
	_, err := p.postprocess(batch)
	require.EqualError(t, err, "expected []int64 or []string class labels, got []float32")

	batch.OutputValues = []any{[]string{"only one"}, [][]float32{{1, 0}, {0, 1}}}
	_, err = p.postprocess(batch)
	require.EqualError(t, err, "expected 2 class labels, got 1")
}
