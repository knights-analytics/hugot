//go:build XLA || ALL

package hugot

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"slices"

	"github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/losses"
	"github.com/gomlx/gomlx/ml/train/optimizers"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/pipelineBackends"
	"github.com/knights-analytics/hugot/pipelines"
	"github.com/knights-analytics/hugot/util"
)

type TrainingSession struct {
	pipeline  pipelineBackends.Pipeline
	config    TrainingConfig
	optimizer optimizers.Interface
	loss      losses.LossFn
}

type TrainingOption func(eo *TrainingSession) error

type TrainingConfig struct {
	ModelPath       string
	OnnxFilename    string
	TrainingOptions []TrainingOption
	Dataset         Dataset
}

func CosineSimilarityLoss(labels, predictions []*context.Node) *context.Node {
	predictionsLeft := predictions[0]
	predictionsRight := predictions[1]
	scores := labels[0]
	s := predictionsLeft.Shape()
	if len(s.Dimensions) > 2 {
		predictionsLeft = graph.ReduceMean(predictionsLeft, s.Dimensions[2:]...)
		predictionsRight = graph.ReduceMean(predictionsRight, s.Dimensions[2:]...)
	}
	cosineSimilarity := graph.Div(graph.Mul(predictionsLeft, predictionsRight), graph.Mul(graph.L2Norm(predictionsLeft), graph.L2Norm(predictionsRight)))
	loss := graph.L2NormSquare(graph.Sub(scores, cosineSimilarity))
	return loss
}

func NewTrainingSession[T pipelineBackends.Pipeline](config TrainingConfig) (*TrainingSession, error) {
	session := &TrainingSession{
		config: config,
	}

	for _, option := range config.TrainingOptions {
		err := option(session)
		if err != nil {
			return nil, err
		}
	}

	if session.optimizer == nil {
		session.optimizer = optimizers.StochasticGradientDescent()
	}

	var trainingPipeline T
	var model *pipelineBackends.Model
	var err error

	options := options.Defaults()
	options.Runtime = "XLA"
	model, err = pipelineBackends.LoadModel(config.ModelPath, config.OnnxFilename, options)

	switch any(trainingPipeline).(type) {
	case *pipelines.FeatureExtractionPipeline:
		pipelineConfig := FeatureExtractionConfig{}
		pipeline := any(trainingPipeline).(*pipelines.FeatureExtractionPipeline)
		pipeline, _, err = InitializePipeline(pipeline, pipelineConfig, options, model)
		if err != nil {
			return nil, err
		}
		session.pipeline = pipeline

		// hook the data up with the pipeline
		if d, ok := session.config.Dataset.(*SemanticSimilarityDataset); !ok {
			return nil, fmt.Errorf("expected SemanticSimilarityDataset, got %T", d)
		} else {
			d.pipeline = pipeline
		}
	default:
		return nil, fmt.Errorf("training for pipeline type is not supported")
	}

	if session.loss == nil {
		switch any(session.pipeline).(type) {
		case *pipelines.FeatureExtractionPipeline:
			session.loss = CosineSimilarityLoss
		default:
			return nil, fmt.Errorf("loss function is required")
		}
	}

	return session, nil
}

func (s *TrainingSession) Train() error {
	switch p := s.pipeline.(type) {
	case *pipelines.FeatureExtractionPipeline:
		XLAModel := p.Model.XLAModel
		backend := XLAModel.Backend
		ctx := XLAModel.Ctx

		modelFn := func(ctx *context.Context, spec any, inputs []*context.Node) []*context.Node {
			l := len(inputs) / 2
			embeddingLeft := XLAModel.Call(ctx, inputs[:l])
			embeddingRight := XLAModel.Call(ctx.Reuse(), inputs[l:])
			return []*context.Node{embeddingLeft, embeddingRight}
		}

		gomlxTrainer := train.NewTrainer(backend,
			ctx,
			modelFn,
			s.loss,
			s.optimizer,
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

// DATASETS

type Dataset interface {
	train.Dataset
	Validate() error
}

type SemanticSimilarityDataset struct {
	train.Dataset
	TrainingPath string
	pipeline     *pipelines.FeatureExtractionPipeline
}

func (s *SemanticSimilarityDataset) Validate() error {
	if s.TrainingPath == "" {
		return fmt.Errorf("training path is required")
	}
	if filepath.Ext(s.TrainingPath) != ".jsonl" {
		return fmt.Errorf("training path must be a .jsonl file")
	}
	return nil
}

type SemanticSimilarityExample struct {
	Sentence1 string  `json:"sentence1"`
	Sentence2 string  `json:"sentence2"`
	Score     float64 `json:"label"`
}

func NewSemanticSimilarityDataset(trainingPath string) (*SemanticSimilarityDataset, error) {
	d := &SemanticSimilarityDataset{
		TrainingPath: trainingPath,
	}
	if err := d.Validate(); err != nil {
		return nil, err
	}
	return d, nil
}

func (s *SemanticSimilarityDataset) Yield() (spec any, inputs []*tensors.Tensor, labels []*tensors.Tensor, err error) {
	jsonBytes, err := util.ReadFileBytes(s.TrainingPath)
	if err != nil {
		return nil, nil, nil, err
	}

	scanner := bufio.NewScanner(bytes.NewReader(jsonBytes))

	inputsLeft := []string{}
	inputsRight := []string{}
	scores := []float64{}

	for scanner.Scan() {
		var lineData SemanticSimilarityExample
		if err := json.Unmarshal(scanner.Bytes(), &lineData); err != nil {
			return nil, nil, nil, fmt.Errorf("failed to parse JSON line: %w", err)
		} else {
			inputsLeft = append(inputsLeft, lineData.Sentence1)
			inputsRight = append(inputsRight, lineData.Sentence2)
			scores = append(scores, lineData.Score)
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, nil, nil, fmt.Errorf("error reading file: %w", err)
	}

	var runErrors []error
	batchLeft := pipelineBackends.NewBatch()
	batchRight := pipelineBackends.NewBatch()
	defer func() {
		runErrors = append(runErrors, batchLeft.Destroy(), batchRight.Destroy())
	}()
	runErrors = append(runErrors, s.pipeline.Preprocess(batchLeft, inputsLeft))
	runErrors = append(runErrors, s.pipeline.Preprocess(batchRight, inputsRight))
	if e := errors.Join(runErrors...); e != nil {
		return nil, nil, nil, e
	}
	inputLeft := batchLeft.InputValues.([]*tensors.Tensor)
	inputRight := batchRight.InputValues.([]*tensors.Tensor)
	labelTensor := tensors.FromFlatDataAndDimensions(scores, len(scores), 1)
	return nil, slices.Concat(inputLeft, inputRight), []*tensors.Tensor{labelTensor}, nil
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
