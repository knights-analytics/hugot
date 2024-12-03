//go:build XLA || ALL

package hugot

import (
	"fmt"

	. "github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/graph"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/losses"
	"github.com/gomlx/gomlx/ml/train/optimizers"
	"github.com/knights-analytics/hugot/pipelineBackends"
	"github.com/knights-analytics/hugot/pipelines"
)

type XLATrainingOptions struct {
	Optimizer optimizers.Interface
	Loss      losses.LossFn
}

func NewXLATrainingSession[T pipelineBackends.Pipeline](config TrainingConfig) (*TrainingSession, error) {
	s, err := newTrainingSession[T]("XLA", config)
	if err != nil {
		return nil, err
	}

	// set defaults
	switch any(s.pipeline).(type) {
	case *pipelines.FeatureExtractionPipeline:
		if s.config.XlaTrainingOptions == nil {
			s.config.XlaTrainingOptions = &XLATrainingOptions{}
		}
		if s.config.XlaTrainingOptions.Optimizer == nil {
			s.config.XlaTrainingOptions.Optimizer = optimizers.StochasticGradientDescent()
		}
		if s.config.XlaTrainingOptions.Loss == nil {
			s.config.XlaTrainingOptions.Loss = CosineSimilarityLoss
		}
	default:
		return nil, fmt.Errorf("loss function is required")
	}
	return s, nil
}

func TrainXLA(s *TrainingSession) error {
	switch p := s.pipeline.(type) {
	case *pipelines.FeatureExtractionPipeline:
		XLAModel := p.Model.XLAModel
		backend := XLAModel.Backend
		ctx := XLAModel.Ctx

		modelFn := func(ctx *context.Context, spec any, inputs []*context.Node) []*context.Node {
			l := len(inputs) / 2
			embeddingLeft := XLAModel.Call(ctx, inputs[:l])[0]
			embeddingRight := XLAModel.Call(ctx.Reuse(), inputs[l:])[0]

			// we mean pool the results if needed e.g. if dimensions are [batch, seq, hidden]
			if len(embeddingLeft.Shape().Dimensions) > 2 {
				var axisToReduce []int
				axisToReduce = append(axisToReduce, 1)
				for i := range embeddingLeft.Shape().Dimensions {
					if i >= 3 {
						axisToReduce = append(axisToReduce, i)
					}
				}
				// TODO: check how to use the mask here to reduce
				embeddingLeft = graph.ReduceMean(embeddingLeft, axisToReduce...)
				embeddingRight = graph.ReduceMean(embeddingRight, axisToReduce...)
			}

			return []*context.Node{embeddingLeft, embeddingRight}
		}

		gomlxTrainer := train.NewTrainer(backend,
			ctx,
			modelFn,
			s.config.XlaTrainingOptions.Loss,
			s.config.XlaTrainingOptions.Optimizer,
			nil,
			nil)

		loop := train.NewLoop(gomlxTrainer)

		// Loop for given number of steps.
		if s.config.Verbose {
			fmt.Printf("Training for %d epochs\n", s.config.Epochs)
		}
		_, err := loop.RunEpochs(s.config.Dataset, s.config.Epochs)
		if err != nil {
			return err
		}
		if s.config.Verbose {
			fmt.Println("Training complete")
		}
	}
	return nil
}

// CosineSimilarityLoss computes the cosine similarity loss between two predictions.
//
// It assumes two predictions are provided which should be tensors of rank 2 with the same dimensions (N, D).
// It also assumes there is one label tensor with rank 2 with dimensions (N, 1) that contains the target cosine similarity.
// The loss will then calculate the residuals between the cosine similarity of the two prediction tensors row-wise
// and the target cosine similarity, and return the mean of the squared residuals.
func CosineSimilarityLoss(labels, predictions []*context.Node) *context.Node {
	predictionsLeft := predictions[0]
	predictionsRight := predictions[1]
	scores := labels[0]

	if predictionsLeft.Shape().Rank() != 2 {
		Panicf("expected rank 2, got %d (shape=%s)", predictionsLeft.Shape().Rank(), predictionsLeft.Shape())
	}
	if predictionsRight.Shape().Rank() != 2 {
		Panicf("expected rank 2, got %d (shape=%s)", predictionsRight.Shape().Rank(), predictionsRight.Shape())
	}
	if scores.Shape().Rank() != 2 {
		Panicf("expected rank 2, got %d (shape=%s)", scores.Shape().Rank(), scores.Shape())
	}
	dotProduct := InsertAxes(Einsum("ij,ij->i", predictionsLeft, predictionsRight), -1)
	normLeft := L2Norm(predictionsLeft, 1)
	normRight := L2Norm(predictionsRight, 1)
	similarity := Div(dotProduct, Mul(normLeft, normRight))
	residuals := L2NormSquare(Sub(scores, similarity), 1)
	loss := ReduceAllMean(residuals)
	return loss
}
