package hugot

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"

	"github.com/knights-analytics/hugot/datasets"
	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/pipelineBackends"
	"github.com/knights-analytics/hugot/pipelines"
	"github.com/knights-analytics/hugot/util"
)

type TrainingSession struct {
	backend          string
	pipeline         pipelineBackends.Pipeline
	config           TrainingConfig
	maxEpochs        int
	freezeEmbeddings bool  // freeze the embedding layers of the transfomer model
	freezeLayers     []int // freeze the layers of the transformer model, 0 is the first layer etc. Set [-1] to freeze all layers apart from the last one
}

// GetPipeline returns the pipeline used in the training session.
func (s *TrainingSession) GetPipeline() pipelineBackends.Pipeline {
	return s.pipeline
}

func (s *TrainingSession) Destroy() error {
	err := s.pipeline.GetModel().Destroy()
	if err != nil {
		return err
	}
	s.pipeline = nil
	return nil
}

type TrainingOption func(eo *TrainingSession) error

func WithEpochs(epochs int) TrainingOption {
	return func(eo *TrainingSession) error {
		if epochs <= 0 {
			return fmt.Errorf("epochs must be greater than 0")
		}
		eo.maxEpochs = epochs
		return nil
	}
}

func WithFreezeEmbeddings() TrainingOption {
	return func(eo *TrainingSession) error {
		eo.freezeEmbeddings = true
		return nil
	}
}

func WithFreezeLayers(layers []int) TrainingOption {
	return func(eo *TrainingSession) error {
		eo.freezeLayers = layers
		return nil
	}
}

type TrainingConfig struct {
	ModelPath            string
	OnnxFilename         string
	Cuda                 bool
	Dataset              datasets.Dataset
	Options              []TrainingOption
	Verbose              bool
	GOMLXTrainingOptions *GOMLXTrainingOptions
}

func newTrainingSession[T pipelineBackends.Pipeline](backend string, config TrainingConfig) (*TrainingSession, error) {
	session := &TrainingSession{
		config:  config,
		backend: backend,
	}

	var trainingPipeline T
	var model *pipelineBackends.Model
	var err error

	opts := options.Defaults()
	opts.Backend = backend

	switch backend {
	case "XLA":
		opts.GoMLXOptions.XLA = true
		opts.GoMLXOptions.Cuda = config.Cuda
	case "GO":
	default:
		return nil, fmt.Errorf("runtime %s is not supported", backend)
	}

	for _, opt := range config.Options {
		if err = opt(session); err != nil {
			return nil, err
		}
	}

	if session.maxEpochs <= 0 {
		session.maxEpochs = 100
	}

	model, err = pipelineBackends.LoadModel(config.ModelPath, config.OnnxFilename, opts)
	if err != nil {
		return nil, err
	}

	switch any(trainingPipeline).(type) {
	case *pipelines.FeatureExtractionPipeline:
		pipelineConfig := FeatureExtractionConfig{}
		pipeline := any(trainingPipeline).(*pipelines.FeatureExtractionPipeline)
		pipeline, _, err = InitializePipeline(pipeline, pipelineConfig, opts, model)
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
	switch s.backend {
	case "GO", "XLA":
		return TrainGoMLX(s)
	default:
		return fmt.Errorf("training runtime %s is not supported", s.backend)
	}
}

// Save serializes the trained model as an onnx model.
// If a tokenizer is present, the tokenizer files are copied from the untrained model directory to the trained model.
// Path is the full path to the directory where the model will be saved.
func (s *TrainingSession) Save(path string) error {
	if path == "" {
		return fmt.Errorf("path is required")
	}

	var writeErr error

	model := s.pipeline.GetModel()
	if model != nil {
		if s.backend == "GO" || s.backend == "XLA" {
			goMLXModel := model.GoMLXModel

			if goMLXModel != nil {
				w, err := util.NewFileWriter(util.PathJoinSafe(path, "model.onnx"), "")
				if err != nil {
					return err
				}
				defer func() {
					writeErr = errors.Join(writeErr, w.Close())
				}()

				if writeErr = goMLXModel.Save(w); writeErr != nil {
					return writeErr
				}
				if model.Tokenizer != nil {
					// copy tokenizer files from original model
					if writeErr = copyTokenizer(model.Path, path); writeErr != nil {
						return writeErr
					}
				}
				return writeErr
			}
			return fmt.Errorf("model is nil")
		} else {
			return fmt.Errorf("go or XLA backends are required for saving a training model")
		}
	} else {
		return fmt.Errorf("pipeline model is nil")
	}
}

func copyTokenizer(from, to string) error {
	toCopy := map[string]bool{
		"special_tokens_map.json": true,
		"tokenizer_config.json":   true,
		"tokenizer.json":          true,
		"vocab.txt":               true,
	}

	walker := func(_ context.Context, _ string, parent string, info os.FileInfo, _ io.Reader) (toContinue bool, err error) {
		if toCopy[info.Name()] {
			if err = util.CopyFile(util.PathJoinSafe(from, parent, info.Name()), to); err != nil {
				return false, err
			}
		}
		return true, nil
	}
	return util.WalkDir()(context.Background(), from, walker)
}
