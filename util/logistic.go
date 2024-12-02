package util

import (
	"slices"

	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/xla/cpu/static"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/data"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/regularizers"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/losses"
	"github.com/gomlx/gomlx/ml/train/optimizers"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/janpfeifer/must"
)

// modelGraph builds graph that returns predictions for inputs.
func modelGraph(ctx *context.Context, spec any, inputs []*context.Node) []*context.Node {
	_ = spec
	logits := layers.Dense(ctx.In("layer_0"), inputs[0], true, 1)
	return []*context.Node{logits}
}

// LogisticRegression is a utility function that implements the binary logistic regression algorithm using gomlx xla
// and returns the weights of the model where the last weight is for the bias term.
func LogisticRegressionWeights(X [][]float64, y []float64) ([]float64, error) {
	backend := backends.New()

	XT := tensors.FromValue(X)
	yT := tensors.FromFlatDataAndDimensions(y, len(y), 1) // requires the same rank as X

	dataset := must.M1(data.InMemoryFromData(backend, "linear dataset", []any{XT}, []any{yT})).
		Infinite(true).Shuffle().BatchSize(len(X), false)

	// Creates Context with learned weights and bias.
	ctx := context.New()
	ctx.In("layer_0").SetParams(map[string]any{
		regularizers.ParamL2: 0.001,
	})

	// train.Trainer executes a training step.
	trainer := train.NewTrainer(backend,
		ctx,
		modelGraph,
		losses.BinaryCrossentropyLogits,
		optimizers.StochasticGradientDescent(),
		nil, nil)

	loop := train.NewLoop(trainer)

	// Loop for given number of steps.
	_, err := loop.RunSteps(dataset, 1000)
	if err != nil {
		return nil, err
	}

	// return coefficients
	coefVar, biasVar := ctx.GetVariableByScopeAndName("/layer_0/dense", "weights"), ctx.GetVariableByScopeAndName("/layer_0/dense", "biases")
	return slices.Concat(tensors.CopyFlatData[float64](coefVar.Value()), tensors.CopyFlatData[float64](biasVar.Value())), nil
}
