package hugot

import (
	"context"
	"encoding/json"
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

type earlyStopping struct {
	patience  int     // number of epochs to wait for improvement before stopping
	tolerance float32 // tolerance for loss comparison
}

type TrainingStatistics struct {
	EpochTrainLosses []float32 `json:"epochTrainLosses"` // stores the training loss for each epoch
	EpochEvalLosses  []float32 `json:"epochEvalLosses"`  // stores the evaluation loss for each epoch
}

type TrainingSession struct {
	backend          string
	pipeline         pipelineBackends.Pipeline
	config           TrainingConfig
	cuda             bool
	maxEpochs        int
	earlyStopping    *earlyStopping
	statistics       TrainingStatistics
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

func WithCuda() TrainingOption {
	return func(eo *TrainingSession) error {
		eo.cuda = true
		return nil
	}
}

func WithEarlyStopping() TrainingOption {
	return WithEarlyStoppingParams(3, 1e-4) // default patience and tolerance
}

func WithEarlyStoppingParams(patience int, tolerance float32) TrainingOption {
	return func(eo *TrainingSession) error {
		if eo.config.EvalDataset == nil {
			return fmt.Errorf("early stopping requires an evaluation dataset")
		}
		if patience <= 0 {
			return fmt.Errorf("patience must be greater than 0")
		}
		if tolerance <= 0 {
			return fmt.Errorf("tolerance must be greater than 0")
		}
		eo.earlyStopping = &earlyStopping{
			patience:  patience,
			tolerance: tolerance,
		}
		return nil
	}
}

type TrainingConfig struct {
	ModelPath            string
	OnnxFilename         string
	Dataset              datasets.Dataset
	EvalDataset          datasets.Dataset // optional, used for early stopping and eval metrics
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

	for _, opt := range config.Options {
		if err = opt(session); err != nil {
			return nil, err
		}
	}

	switch backend {
	case "XLA":
		opts.GoMLXOptions.XLA = true
		opts.GoMLXOptions.Cuda = session.cuda
	case "GO":
	default:
		return nil, fmt.Errorf("runtime %s is not supported", backend)
	}

	if session.maxEpochs <= 0 {
		session.maxEpochs = 100 // default to 100 epochs if not set
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

		// hook the datasets up with the pipeline for tokenization
		var trainDS *datasets.SemanticSimilarityDataset
		var evalDS *datasets.SemanticSimilarityDataset
		var ok bool

		trainDS, ok = session.config.Dataset.(*datasets.SemanticSimilarityDataset)
		if !ok {
			return nil, fmt.Errorf("expected SemanticSimilarityDataset for train dataset, got %T", trainDS)
		}
		if session.config.EvalDataset != nil {
			evalDS, ok = session.config.EvalDataset.(*datasets.SemanticSimilarityDataset)
			if !ok {
				return nil, fmt.Errorf("expected SemanticSimilarityDataset for eval dataset, got %T", session.config.EvalDataset)
			}
		}
		if setErr := trainDS.SetTokenizationPipeline(pipeline); setErr != nil {
			return nil, fmt.Errorf("failed to set tokenization pipeline for training dataset: %w", setErr)
		}
		if evalDS != nil {
			if setErr := evalDS.SetTokenizationPipeline(pipeline); setErr != nil {
				return nil, fmt.Errorf("failed to set tokenization pipeline for evaluation dataset: %w", setErr)
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

	statisticsWriter, err := util.NewFileWriter(util.PathJoinSafe(path, "statistics.txt"), "")
	if err != nil {
		return err
	}
	defer func() {
		writeErr = errors.Join(writeErr, statisticsWriter.Close())
	}()

	statisticsBytes, err := json.Marshal(s.statistics)
	if err != nil {
		return fmt.Errorf("failed to marshal training statistics: %w", err)
	}
	if _, err = statisticsWriter.Write(statisticsBytes); err != nil {
		return fmt.Errorf("failed to write training statistics: %w", err)
	}

	model := s.pipeline.GetModel()
	if model != nil {
		if s.backend == "GO" || s.backend == "XLA" {
			goMLXModel := model.GoMLXModel

			if goMLXModel != nil {
				modelWriter, err := util.NewFileWriter(util.PathJoinSafe(path, "model.onnx"), "")
				if err != nil {
					return err
				}
				defer func() {
					writeErr = errors.Join(writeErr, modelWriter.Close())
				}()
				writeErr = errors.Join(writeErr, goMLXModel.Save(modelWriter))
				if model.Tokenizer != nil {
					// copy tokenizer files from original model
					writeErr = errors.Join(writeErr, copyTokenizer(model.Path, path))
				}
			} else {
				return fmt.Errorf("gomlx model is nil")
			}
		} else {
			return fmt.Errorf("go or XLA backends are required for saving a training model")
		}
	} else {
		return fmt.Errorf("pipeline model is nil")
	}
	return writeErr
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
