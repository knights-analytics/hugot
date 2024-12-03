//go:build XLA || ALL

package hugot

import (
	"fmt"

	"github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/losses"
	"github.com/gomlx/gomlx/ml/train/optimizers"
	"github.com/knights-analytics/hugot/pipelineBackends"
	"github.com/knights-analytics/hugot/pipelines"
)

type XLATrainingOptions struct {
	optimizer optimizers.Interface
	loss      losses.LossFn
}

func NewXLATrainingSession[T pipelineBackends.Pipeline](config TrainingConfig) (*TrainingSession, error) {
	s, err := newTrainingSession[T]("XLA", config)
	if err != nil {
		return nil, err
	}

	s.xlaOptions = &XLATrainingOptions{
		optimizer: optimizers.StochasticGradientDescent(),
	}
	s.runtime = "XLA"

	switch any(s.pipeline).(type) {
	case *pipelines.FeatureExtractionPipeline:
		s.xlaOptions.loss = CosineSimilarityLoss
	default:
		return nil, fmt.Errorf("loss function is required")
	}
	return s, nil
}

func CosineSimilarityLoss(labels, predictions []*context.Node) *context.Node {
	predictionsLeft := predictions[0]
	predictionsRight := predictions[1]
	scores := labels[0]

	if predictionsLeft.Shape().Rank() != 2 {
		panic(fmt.Errorf("expected rank 2, got %d", predictionsLeft.Shape().Rank()))
	}
	if predictionsRight.Shape().Rank() != 2 {
		panic(fmt.Errorf("expected rank 2, got %d", predictionsLeft.Shape().Rank()))
	}
	if scores.Shape().Rank() != 2 {
		panic(fmt.Errorf("expected rank 2, got %d", predictionsLeft.Shape().Rank()))
	}
	m := graph.Mul(predictionsLeft, predictionsRight)
	dotProduct := graph.ReduceAndKeep(m, graph.ReduceSum, 1)
	normLeft := graph.L2Norm(predictionsLeft, 1)
	normRight := graph.L2Norm(predictionsRight, 1)
	denom := graph.Mul(normLeft, normRight)
	similarity := graph.Div(dotProduct, denom)
	residuals := graph.L2NormSquare(graph.Sub(scores, similarity), 1)
	loss := graph.ReduceAllMean(residuals)
	return loss
}

func TrainXLA(s *TrainingSession) error {
	switch p := s.pipeline.(type) {
	case *pipelines.FeatureExtractionPipeline:
		XLAModel := p.Model.XLAModel
		backend := XLAModel.Backend
		ctx := XLAModel.Ctx

		modelFn := func(ctx *context.Context, spec any, inputs []*context.Node) []*context.Node {
			l := len(inputs) / 2
			embeddingLeft := XLAModel.Call(ctx, inputs[:l])
			embeddingRight := XLAModel.Call(ctx.Reuse(), inputs[l:])

			// we mean pool the results if needed e.g. if dimensions are [batch, seq, hidden]
			if len(embeddingLeft.Shape().Dimensions) > 2 {
				var axisToReduce []int
				axisToReduce = append(axisToReduce, 1)
				for i := range embeddingLeft.Shape().Dimensions {
					if i >= 3 {
						axisToReduce = append(axisToReduce, i)
					}
				}
				// TODO: check how to use the mask here
				embeddingLeft = graph.ReduceMean(embeddingLeft, axisToReduce...)
				embeddingRight = graph.ReduceMean(embeddingRight, axisToReduce...)
			}

			return []*context.Node{embeddingLeft, embeddingRight}
		}

		gomlxTrainer := train.NewTrainer(backend,
			ctx,
			modelFn,
			s.xlaOptions.loss,
			s.xlaOptions.optimizer,
			nil,
			nil)

		loop := train.NewLoop(gomlxTrainer)

		// Loop for given number of steps.
		_, err := loop.RunSteps(s.config.Dataset, 1)
		if err != nil {
			return err
		}
	}
	return nil
}
