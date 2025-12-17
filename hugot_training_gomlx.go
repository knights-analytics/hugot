package hugot

import (
	"errors"
	"fmt"
	"math"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/train"
	"github.com/gomlx/gomlx/pkg/ml/train/losses"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers"
	"github.com/gomlx/gomlx/pkg/core/dtypes"

	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/pipelines"
)

type GOMLXTrainingOptions struct {
	Optimizer optimizers.Interface
	Loss      losses.LossFn
}

type stoppingError struct{}

func (e stoppingError) Error() string {
	return "stopping error"
}

func NewGoTrainingSession[T backends.Pipeline](config TrainingConfig) (*TrainingSession, error) {
	s, err := newTrainingSession[T]("GO", config)
	if err != nil {
		return nil, err
	}

	return newGoMLXTrainingSession(s)
}

func NewXLATrainingSession[T backends.Pipeline](config TrainingConfig) (*TrainingSession, error) {
	s, err := newTrainingSession[T]("XLA", config)
	if err != nil {
		return nil, err
	}

	return newGoMLXTrainingSession(s)
}

func newGoMLXTrainingSession(s *TrainingSession) (*TrainingSession, error) {
	// set defaults
	switch any(s.pipeline).(type) {
	case *pipelines.FeatureExtractionPipeline:
		if s.config.GOMLXTrainingOptions == nil {
			s.config.GOMLXTrainingOptions = &GOMLXTrainingOptions{}
		}
		if s.config.GOMLXTrainingOptions.Optimizer == nil {
			s.config.GOMLXTrainingOptions.Optimizer = optimizers.StochasticGradientDescent()
		}
		if s.config.GOMLXTrainingOptions.Loss == nil {
			s.config.GOMLXTrainingOptions.Loss = losses.MeanSquaredError
		}
	default:
		return nil, fmt.Errorf("loss function is required")
	}
	return s, nil
}

func TrainGoMLX(s *TrainingSession) error {
	switch p := s.pipeline.(type) {
	case *pipelines.FeatureExtractionPipeline:
		GoMLXModel := p.Model.GoMLXModel
		backend := GoMLXModel.Backend
		ctx := GoMLXModel.Ctx

		// freeze the layers if requested
		freezeAllButLast := slices.Contains(s.freezeLayers, -1)
		re := regexp.MustCompile(`layer\.(\d+)`) // identify the layer number in the variable name

		for v := range ctx.IterVariables() {
			name := v.Name()
			if (s.freezeEmbeddings || freezeAllButLast) && strings.HasPrefix(name, "embeddings") {
				v.SetTrainable(false)
				continue
			}

			if matches := re.FindStringSubmatch(name); matches != nil {
				layerNumStr := matches[1]
				layerNum, err := strconv.Atoi(layerNumStr)
				if err != nil {
					return fmt.Errorf("failed to parse layer number from variable name %s: %w", name, err)
				}
				if freezeAllButLast || slices.Contains(s.freezeLayers, layerNum) {
					v.SetTrainable(false)
					continue
				}
			}
		}

		modelFn := func(ctx *context.Context, _ any, inputs []*context.Node) []*context.Node {
			inputsLHS := inputs[:3] // inputIDs, attentionMask, tokenTypeIDs if present
			inputsRHS := inputs[3:]

			embeddingLHS := GoMLXModel.Call(ctx.Reuse(), inputsLHS)[0]
			embeddingRHS := GoMLXModel.Call(ctx.Reuse(), inputsRHS)[0]

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
			return []*context.Node{cosineSimilarity}
		}

		gomlxTrainer := train.NewTrainer(backend,
			ctx,
			modelFn,
			s.config.GOMLXTrainingOptions.Loss,
			s.config.GOMLXTrainingOptions.Optimizer,
			nil,
			nil)

		loop := train.NewLoop(gomlxTrainer)

		// Loop for given number of epochs.
		if s.config.Verbose {
			if s.earlyStopping != nil {
				fmt.Printf("Training for %d epochs with early stopping\n", s.maxEpochs)
			} else {
				fmt.Printf("Training for %d epochs\n", s.maxEpochs)
			}
		}

		var currentEpoch int
		var trainLosses []float32
		var evalLosses []float32

		var bestLoss float32 = math.MaxFloat32
		epochsWithoutImprovement := 0

		var evaluateEpoch train.OnStepFn = func(loop *train.Loop, _ []*tensors.Tensor) error {
			if loop.Epoch != currentEpoch {
				if s.config.TrainEvalDataset != nil {
					if s.config.Verbose {
						fmt.Printf("Running evaluation for epoch %d\n on trainEvalDataset", loop.Epoch)
					}
					lossAndMetrics, err := gomlxTrainer.Eval(s.config.TrainEvalDataset)
					if err != nil {
						return err
					}
					meanTrainLoss := lossAndMetrics[1].Value().(float32)
					trainLosses = append(trainLosses, meanTrainLoss)
				}

				if s.earlyStopping != nil {
					if s.config.Verbose {
						fmt.Printf("Running evaluation for epoch %d\n on evalDataset", loop.Epoch)
					}
					lossAndMetrics, err := gomlxTrainer.Eval(s.config.EvalDataset)
					if err != nil {
						return err
					}
					meanLoss := lossAndMetrics[1].Value().(float32)
					evalLosses = append(evalLosses, meanLoss)

					if bestLoss-meanLoss > s.earlyStopping.tolerance {
						bestLoss = meanLoss
						epochsWithoutImprovement = 0
						if s.config.Verbose {
							fmt.Printf("New best loss: %.4f at epoch %d\n", bestLoss, loop.Epoch)
						}
					} else {
						epochsWithoutImprovement++
						if s.config.Verbose {
							fmt.Printf("No improvement in loss, epochs without improvement: %d\n", epochsWithoutImprovement)
						}
						if epochsWithoutImprovement >= s.earlyStopping.patience {
							if s.config.Verbose {
								fmt.Printf("Early stopping triggered after %d epochs without improvement\n", s.earlyStopping.patience)
							}
							return stoppingError{} // trigger stopping
						}
					}
				}
				currentEpoch = loop.Epoch
			}
			return nil
		}
		loop.OnStep("evaluateAfterEpoch", train.Priority(1), evaluateEpoch)

		// we rely on try catch because an error is returned if there is an initialization error but
		// a panic will be thrown if e.g. dataset reset fails.
		err := exceptions.TryCatch[error](func() {
			if _, err := loop.RunEpochs(s.config.TrainDataset, s.maxEpochs); err != nil {
				if errors.Is(err, stoppingError{}) {
					if s.config.Verbose {
						fmt.Printf("Training stopped after epoch %d\n", currentEpoch)
					}
				} else {
					panic(err)
				}
			}
		})
		if err != nil {
			return err
		}
		if s.config.Verbose {
			fmt.Println("Training complete")
		}
		s.statistics = TrainingStatistics{
			EpochTrainLosses: trainLosses,
			EpochEvalLosses:  evalLosses,
		}
	}
	return nil
}
