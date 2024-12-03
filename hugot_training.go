package hugot

import (
	"fmt"

	"github.com/knights-analytics/hugot/datasets"
	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/pipelineBackends"
	"github.com/knights-analytics/hugot/pipelines"
)

type TrainingSession struct {
	runtime  string
	pipeline pipelineBackends.Pipeline
	config   TrainingConfig
}

type TrainingOption func(eo *TrainingSession) error

type TrainingConfig struct {
	ModelPath          string
	OnnxFilename       string
	Cuda               bool
	Epochs             int
	XlaTrainingOptions *XLATrainingOptions
	Dataset            datasets.Dataset
	Verbose            bool
}

func newTrainingSession[T pipelineBackends.Pipeline](runtime string, config TrainingConfig) (*TrainingSession, error) {
	session := &TrainingSession{
		config:  config,
		runtime: runtime,
	}

	var trainingPipeline T
	var model *pipelineBackends.Model
	var err error

	options := options.Defaults()
	options.Runtime = runtime

	switch runtime {
	case "XLA":
		options.XLAOptions.Cuda = config.Cuda
	default:
		return nil, fmt.Errorf("runtime %s is not supported", runtime)
	}

	if config.Epochs <= 0 {
		config.Epochs = 1
	}

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

	if session.config.Verbose {
		session.config.Dataset.SetVerbose(true)
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

func (s *TrainingSession) Save(path string) error {
	model := s.pipeline.GetModel()
	if model != nil {
		xlaModel := model.XLAModel
		if xlaModel != nil {
			if err := xlaModel.OnnxModel.ContextToONNX(xlaModel.Ctx); err != nil {
				return err
			}
			if err := xlaModel.OnnxModel.SaveToFile(path); err != nil {
				return err
			}
			return nil
		} else {
			return fmt.Errorf("xla model is nil")
		}
	} else {
		return fmt.Errorf("pipeline model is nil")
	}
}
