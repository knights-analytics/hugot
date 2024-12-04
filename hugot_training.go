package hugot

import (
	"fmt"

	"github.com/knights-analytics/hugot/datasets"
	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/pipelineBackends"
	"github.com/knights-analytics/hugot/pipelines"
)

type TrainingSession struct {
	runtime    string
	pipeline   pipelineBackends.Pipeline
	config     TrainingConfig
	xlaOptions *XLATrainingOptions
}

type TrainingOption func(eo *TrainingSession) error

type TrainingConfig struct {
	ModelPath       string
	OnnxFilename    string
	TrainingOptions []TrainingOption
	Dataset         datasets.Dataset
}

func newTrainingSession[T pipelineBackends.Pipeline](runtime string, config TrainingConfig) (*TrainingSession, error) {
	session := &TrainingSession{
		config: config,
	}

	for _, option := range config.TrainingOptions {
		err := option(session)
		if err != nil {
			return nil, err
		}
	}

	var trainingPipeline T
	var model *pipelineBackends.Model
	var err error

	options := options.Defaults()
	options.Runtime = runtime
	model, err = pipelineBackends.LoadModel(config.ModelPath, config.OnnxFilename, options)
	if err != nil {
		return nil, err
	}

	switch any(trainingPipeline).(type) {
	case *pipelines.FeatureExtractionPipeline:
		pipelineConfig := FeatureExtractionConfig{}
		pipeline := any(trainingPipeline).(*pipelines.FeatureExtractionPipeline)
		pipeline, _, err = InitializePipeline(pipeline, pipelineConfig, options, model)
		if err != nil {
			return nil, err
		}
		session.pipeline = pipeline

		// hook the dataset up with the pipeline for tokenization
		if d, ok := session.config.Dataset.(*datasets.SemanticSimilarityDataset); !ok {
			return nil, fmt.Errorf("expected SemanticSimilarityDataset, got %T", d)
		} else {
			if e := d.SetTokenizationPipeline(pipeline); e != nil {
				return nil, e
			}
		}
	default:
		return nil, fmt.Errorf("training for pipeline type is not supported")
	}

	return session, nil
}

func (s *TrainingSession) Train() error {
	switch s.runtime {
	case "XLA":
		return TrainXLA(s)
	default:
		return fmt.Errorf("training runtime %s is not supported", s.runtime)
	}
}

// XLeft := tensors.FromFlatDataAndDimensions()

// // LogisticRegressionDataset is a dataset for training a logistic regression model.
// type LogisticRegressionDataset struct {
// 	X [][]float64
// 	Y []float64
// }

// func (d *LogisticRegressionDataset) Validate() error {
// 	if len(d.X) == 0 {
// 		return fmt.Errorf("input X must not be empty")
// 	}
// 	if len(d.X) != len(d.Y) {
// 		return fmt.Errorf("inputs X and y must have the same length")
// 	}

// 	nFeatures := len(d.X[0])

// 	for i := range len(d.X) {
// 		if d.Y[i] != 0 && d.Y[i] != 1 {
// 			return fmt.Errorf("labels must be 0 or 1, got %f for example %d", d.Y[i], i)
// 		}

// 		if len(d.X[i]) != nFeatures {
// 			return fmt.Errorf("all examples must have the same number of features, got %d for example %d", len(d.X[i]), i)
// 		}
// 	}

// 	return nil
// }

// func (d *LogisticRegressionDataset) toGomlx(backend backends.Backend) (train.Dataset, error) {
// 	XT := tensors.FromValue(d.X)
// 	yT := tensors.FromFlatDataAndDimensions(d.Y, len(d.Y), 1) // requires the same rank as X

// 	dataset, err := data.InMemoryFromData(backend, "linear dataset", []any{XT}, []any{yT})
// 	if err != nil {
// 		return nil, err
// 	}
// 	dataset = dataset.Infinite(true).Shuffle().BatchSize(len(d.X), false)
// 	return dataset, nil
// }

// func (t *Trainer) Train() error {
// 	var gomlxDataset train.Dataset
// 	var err error

// 	switch t.Pipeline.(type) {
// 	case *taskPipelines.LogisticRegressionPipeline:
// 		if _, ok := t.Data.(*LogisticRegressionDataset); !ok {
// 			return fmt.Errorf("expected LogisticRegressionDataset, got %T", t.Data)
// 		}
// 		if err := t.Data.Validate(); err != nil {
// 			return err
// 		}
// 		gomlxDataset, err = t.Data.toGomlx(t.backend)
// 		if err != nil {
// 			return err
// 		}
// 	}

// 	gomlxTrainer := train.NewTrainer(t.backend,
// 		t.ctx,
// 		newLogisticGraph,
// 		t.loss,
// 		t.optimizer,
// 		nil,
// 		nil)

// 	loop := train.NewLoop(gomlxTrainer)

// 	// Loop for given number of steps.
// 	_, err = loop.RunSteps(gomlxDataset, 1000)
// 	if err != nil {
// 		return err
// 	}
// 	return nil
// }

// // modelGraph builds graph that returns predictions for inputs.
// func newLogisticGraph(ctx *context.Context, spec any, inputs []*context.Node) []*context.Node {
// 	_ = spec
// 	logits := layers.Dense(ctx.In("layer_0"), inputs[0], true, 1)
// 	return []*context.Node{logits}
// }
