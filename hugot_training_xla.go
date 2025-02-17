//go:build XLA || ALL

package hugot

import (
	"fmt"

	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/losses"
	"github.com/gomlx/gomlx/ml/train/optimizers"
	"github.com/gomlx/gopjrt/dtypes"

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
			s.config.XlaTrainingOptions.Loss = losses.MeanSquaredError
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
			inputsLeft := inputs[:3] // inputIDs, attentionMask, tokenTypeIDs if present
			inputsRight := inputs[3:]

			embeddingLeft := XLAModel.Call(ctx.Reuse(), inputsLeft)[0]
			embeddingRight := XLAModel.Call(ctx.Reuse(), inputsRight)[0]

			// we mean pool the results if needed e.g. if dimensions are [batch, seq, hidden]
			if len(embeddingLeft.Shape().Dimensions) > 2 {
				batchSize := embeddingLeft.Shape().Dim(0)
				embeddingSize := embeddingLeft.Shape().Dim(-1)
				embeddingLeft = graph.Reshape(embeddingLeft, batchSize, -1, embeddingSize)
				embeddingRight = graph.Reshape(embeddingRight, batchSize, -1, embeddingSize)

				maskLeft := graph.ConvertDType(graph.BroadcastToShape(graph.Reshape(inputsLeft[1], batchSize, -1, 1), embeddingLeft.Shape()), dtypes.Bool)
				maskRight := graph.ConvertDType(graph.BroadcastToShape(graph.Reshape(inputsRight[1], batchSize, -1, 1), embeddingRight.Shape()), dtypes.Bool)

				embeddingLeft = graph.MaskedReduceMean(embeddingLeft, maskLeft, 1)
				embeddingRight = graph.MaskedReduceMean(embeddingRight, maskRight, 1)
			}

			cosineSimilarity := CosineSimilarity(embeddingLeft, embeddingRight)
			return []*context.Node{cosineSimilarity}
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

		// we rely on try catch because an error is returned if there is an initialization error but
		// a panic will be thrown if e.g. dataset reset fails.
		err := exceptions.TryCatch[error](func() {
			if _, err := loop.RunEpochs(s.config.Dataset, s.config.Epochs); err != nil {
				panic(err)
			}
		})
		if err != nil {
			return err
		}
		if s.config.Verbose {
			fmt.Println("Training complete")
		}
	}
	return nil
}

func CosineSimilarity(left *context.Node, right *context.Node) *context.Node {
	if left.Shape().Rank() != 2 {
		exceptions.Panicf("expected rank 2, got %d (shape=%s)", left.Shape().Rank(), left.Shape())
	}
	if right.Shape().Rank() != 2 {
		exceptions.Panicf("expected rank 2, got %d (shape=%s)", right.Shape().Rank(), right.Shape())
	}
	left = graph.Where(graph.Equal(left, graph.ZerosLike(left)), graph.OnePlus(left), left)
	right = graph.Where(graph.Equal(right, graph.ZerosLike(right)), graph.OnePlus(right), right)
	dotProduct := graph.InsertAxes(graph.Einsum("ij,ij->i", left, right), -1)
	normalisationDenominator := graph.Mul(graph.L2Norm(left, 1), graph.L2Norm(right, 1))
	return graph.Div(dotProduct, normalisationDenominator)
}
