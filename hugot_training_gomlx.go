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

type GOMLXTrainingOptions struct {
	Optimizer optimizers.Interface
	Loss      losses.LossFn
}

func NewGoTrainingSession[T pipelineBackends.Pipeline](config TrainingConfig) (*TrainingSession, error) {
	s, err := newTrainingSession[T]("GO", config)
	if err != nil {
		return nil, err
	}

	return newGoMLXTrainingSession(s)
}

func NewXLATrainingSession[T pipelineBackends.Pipeline](config TrainingConfig) (*TrainingSession, error) {
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

		modelFn := func(ctx *context.Context, spec any, inputs []*context.Node) []*context.Node {
			inputsLhs := inputs[:3] // inputIDs, attentionMask, tokenTypeIDs if present
			inputsRhs := inputs[3:]

			embeddingLhs := GoMLXModel.Call(ctx.Reuse(), inputsLhs)[0]
			embeddingRhs := GoMLXModel.Call(ctx.Reuse(), inputsRhs)[0]

			// we mean pool the results if needed e.g. if dimensions are [batch, seq, hidden]
			if len(embeddingLhs.Shape().Dimensions) > 2 {
				batchSize := embeddingLhs.Shape().Dim(0)
				embeddingSize := embeddingLhs.Shape().Dim(-1)
				embeddingLhs = graph.Reshape(embeddingLhs, batchSize, -1, embeddingSize)
				embeddingRhs = graph.Reshape(embeddingRhs, batchSize, -1, embeddingSize)

				maskLhs := graph.ConvertDType(graph.BroadcastToShape(graph.Reshape(inputsLhs[1], batchSize, -1, 1), embeddingLhs.Shape()), dtypes.Bool)
				maskRhs := graph.ConvertDType(graph.BroadcastToShape(graph.Reshape(inputsRhs[1], batchSize, -1, 1), embeddingRhs.Shape()), dtypes.Bool)

				embeddingLhs = graph.MaskedReduceMean(embeddingLhs, maskLhs, 1)
				embeddingRhs = graph.MaskedReduceMean(embeddingRhs, maskRhs, 1)
			}

			cosineSimilarity := graph.CosineSimilarity(embeddingLhs, embeddingRhs, -1)
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
