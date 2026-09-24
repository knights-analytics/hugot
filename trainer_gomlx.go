package hugot

import (
	"context"
	"errors"
	"fmt"
	"math"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	mlmodel "github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/loss"
	"github.com/gomlx/gomlx/ml/train/optimizer"

	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/pipelines"
)

// GOMLXTrainingOptions overrides the optimizer and loss function used for training.
type GOMLXTrainingOptions struct {
	Optimizer optimizer.Interface
	Loss      loss.LossFn
}

type stoppingError struct{}

func (e stoppingError) Error() string {
	return "stopping error"
}

// applyTrainingDefaults fills in the optimizer and loss function for pipeline types that have
// sensible defaults. A pipeline type that cannot be trained is rejected here, at construction,
// rather than when Train is called.
func applyTrainingDefaults[T backends.Pipeline](config *TrainerConfig[T]) error {
	var zero T
	switch any(zero).(type) {
	case *pipelines.FeatureExtractionPipeline:
		if config.GOMLXOptions == nil {
			config.GOMLXOptions = &GOMLXTrainingOptions{}
		}
		if config.GOMLXOptions.Optimizer == nil {
			config.GOMLXOptions.Optimizer = optimizer.StochasticGradientDescent()
		}
		if config.GOMLXOptions.Loss == nil {
			config.GOMLXOptions.Loss = loss.MeanSquaredError
		}
		return nil
	default:
		return fmt.Errorf("training is not supported for pipeline type %T", zero)
	}
}

// freezeVariables applies the freezeEmbeddings and freezeLayers settings to the model's variables.
func freezeVariables(store *mlmodel.Store, settings trainerSettings) error {
	freezeAllButLast := slices.Contains(settings.freezeLayers, -1)
	if !settings.freezeEmbeddings && !freezeAllButLast && len(settings.freezeLayers) == 0 {
		return nil
	}
	re := regexp.MustCompile(`layer\.(\d+)`) // identify the layer number in the variable name

	for v := range store.IterVariables() {
		name := v.Name()
		if (settings.freezeEmbeddings || freezeAllButLast) && strings.HasPrefix(name, "embeddings") {
			v.SetTrainable(false)
			continue
		}
		matches := re.FindStringSubmatch(name)
		if matches == nil {
			continue
		}
		layerNum, err := strconv.Atoi(matches[1])
		if err != nil {
			return fmt.Errorf("failed to parse layer number from variable name %s: %w", name, err)
		}
		if freezeAllButLast || slices.Contains(settings.freezeLayers, layerNum) {
			v.SetTrainable(false)
		}
	}
	return nil
}

// trainGoMLX runs the training loop. Every backend trains through GoMLX — an ORT session uses it
// via options.WithGoMLX — so there is no backend dispatch here.
func (t *Trainer[T]) trainGoMLX(ctx context.Context) error {
	p, ok := any(t.pipeline).(*pipelines.FeatureExtractionPipeline)
	if !ok {
		// Unreachable via NewTrainer, which rejects unsupported types at construction. Kept as a
		// real error rather than a silent no-op, which is what this switch used to do.
		return fmt.Errorf("training is not supported for pipeline type %T", t.pipeline)
	}

	goMLXModel := p.Model.GoMLXModel
	backend := goMLXModel.Backend
	store := goMLXModel.Store

	if err := freezeVariables(store, t.settings); err != nil {
		return err
	}

	modelFn := func(scope *mlmodel.Scope, _ any, inputs []*mlmodel.Node) []*mlmodel.Node {
		inputsLHS := inputs[:3] // inputIDs, attentionMask, tokenTypeIDs if present
		inputsRHS := inputs[3:]

		embeddingLHS := goMLXModel.Call(scope, inputsLHS)[0]
		embeddingRHS := goMLXModel.Call(scope, inputsRHS)[0]

		// we mean pool the results if needed e.g. if dimensions are [batch, seq, hidden]
		if len(embeddingLHS.Shape().Dimensions) > 2 {
			batchSize := embeddingLHS.Shape().Dim(0)
			embeddingSize := embeddingLHS.Shape().Dim(-1)
			embeddingLHS = graph.Reshape(embeddingLHS, batchSize, -1, embeddingSize)
			embeddingRHS = graph.Reshape(embeddingRHS, batchSize, -1, embeddingSize)

			maskLHS := graph.ConvertDType(graph.BroadcastToShape(graph.Reshape(inputsLHS[1], batchSize, -1, 1), embeddingLHS.Shape()), dtypes.Bool)
			maskRHS := graph.ConvertDType(graph.BroadcastToShape(graph.Reshape(inputsRHS[1], batchSize, -1, 1), embeddingRHS.Shape()), dtypes.Bool)

			embeddingLHS = graph.MaskedReduceMean(embeddingLHS, maskLHS, 1)
			embeddingRHS = graph.MaskedReduceMean(embeddingRHS, maskRHS, 1)
		}
		cosineSimilarity := graph.CosineSimilarity(embeddingLHS, embeddingRHS, -1)
		return []*mlmodel.Node{cosineSimilarity}
	}

	gomlxTrainer := train.NewTrainer(backend,
		store,
		modelFn,
		t.config.GOMLXOptions.Loss,
		t.config.GOMLXOptions.Optimizer,
		nil,
		nil)

	loop := train.NewLoop(gomlxTrainer)

	if t.config.Verbose {
		if t.settings.earlyStopping != nil {
			fmt.Printf("Training for %d epochs with early stopping\n", t.settings.maxEpochs)
		} else {
			fmt.Printf("Training for %d epochs\n", t.settings.maxEpochs)
		}
	}

	var currentEpoch int
	var trainLosses []float32
	var evalLosses []float32

	var bestLoss float32 = math.MaxFloat32
	epochsWithoutImprovement := 0

	var evaluateEpoch train.OnStepFn = func(loop *train.Loop, _ []*tensors.Tensor) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		if loop.Epoch == currentEpoch {
			return nil
		}
		if t.config.TrainEvalDataset != nil {
			if t.config.Verbose {
				fmt.Printf("Running evaluation for epoch %d on trainEvalDataset\n", loop.Epoch)
			}
			lossAndMetrics, err := gomlxTrainer.Eval(t.config.TrainEvalDataset)
			if err != nil {
				return err
			}
			trainLosses = append(trainLosses, lossAndMetrics[0].Value().(float32))
		}

		if t.settings.earlyStopping != nil {
			if t.config.Verbose {
				fmt.Printf("Running evaluation for epoch %d on evalDataset\n", loop.Epoch)
			}
			lossAndMetrics, err := gomlxTrainer.Eval(t.config.EvalDataset)
			if err != nil {
				return err
			}
			meanLoss := lossAndMetrics[0].Value().(float32)
			evalLosses = append(evalLosses, meanLoss)

			if bestLoss-meanLoss > t.settings.earlyStopping.tolerance {
				bestLoss = meanLoss
				epochsWithoutImprovement = 0
				if t.config.Verbose {
					fmt.Printf("New best loss: %.4f at epoch %d\n", bestLoss, loop.Epoch)
				}
			} else {
				epochsWithoutImprovement++
				if t.config.Verbose {
					fmt.Printf("No improvement in loss, epochs without improvement: %d\n", epochsWithoutImprovement)
				}
				if epochsWithoutImprovement >= t.settings.earlyStopping.patience {
					if t.config.Verbose {
						fmt.Printf("Early stopping triggered after %d epochs without improvement\n", t.settings.earlyStopping.patience)
					}
					return stoppingError{} // trigger stopping
				}
			}
		}
		currentEpoch = loop.Epoch
		return nil
	}
	loop.OnStep("evaluateAfterEpoch", train.Priority(1), evaluateEpoch)

	// we rely on try catch because an error is returned if there is an initialization error but
	// a panic will be thrown if e.g. dataset reset fails.
	if _, err := loop.RunEpochs(t.config.TrainDataset, t.settings.maxEpochs); err != nil {
		if errors.Is(err, stoppingError{}) {
			if t.config.Verbose {
				fmt.Printf("Training stopped after epoch %d\n", currentEpoch)
			}
		} else {
			return err
		}
	}

	if t.config.Verbose {
		fmt.Println("Training complete")
	}
	t.statistics = TrainingStatistics{
		EpochTrainLosses: trainLosses,
		EpochEvalLosses:  evalLosses,
	}
	return nil
}
