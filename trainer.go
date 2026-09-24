package hugot

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"

	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/datasets"
	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/util/fileutil"
)

// TrainingStatistics holds the per-epoch losses recorded during a training run.
type TrainingStatistics struct {
	EpochTrainLosses []float32 `json:"epochTrainLosses"` // training loss for each epoch
	EpochEvalLosses  []float32 `json:"epochEvalLosses"`  // evaluation loss for each epoch
}

type earlyStopping struct {
	patience  int     // number of epochs to wait for improvement before stopping
	tolerance float32 // tolerance for loss comparison
}

// TrainerConfig describes WHAT to train: a model and the data to train it on. How to train it is
// supplied separately as [TrainerOption] values, and where to run it is a property of the
// [Session] the trainer is created from.
type TrainerConfig[T backends.Pipeline] struct {
	// TrainDataset is the dataset to train on. Required.
	TrainDataset datasets.Dataset
	// TrainEvalDataset is evaluated after each epoch to record the training loss. Optional.
	TrainEvalDataset datasets.Dataset
	// EvalDataset is evaluated after each epoch for early stopping. Required by
	// [WithEarlyStopping].
	EvalDataset datasets.Dataset
	// GOMLXOptions overrides the optimizer and loss function. Defaults are filled in per pipeline
	// type when nil.
	GOMLXOptions *GOMLXTrainingOptions
	// ModelPath is the directory holding the ONNX model to fine-tune. Required.
	ModelPath string
	// OnnxFilename names the ONNX file within ModelPath, when it cannot be inferred.
	OnnxFilename string
	// Verbose prints progress to stdout during training.
	Verbose bool
}

// trainerSettings holds everything a [TrainerOption] can set.
//
// It is deliberately NOT generic. If options targeted Trainer[T] directly, every option would need
// a type argument at the call site — hugot.WithEpochs[*pipelines.FeatureExtractionPipeline](2) —
// which is unusable. Keeping the option target concrete means options stay plain values.
type trainerSettings struct {
	earlyStopping    *earlyStopping
	freezeLayers     []int // 0 is the first layer; [-1] freezes all layers but the last
	maxEpochs        int
	freezeEmbeddings bool
}

// TrainerOption configures how a training run behaves.
type TrainerOption func(*trainerSettings) error

// WithEpochs sets the maximum number of epochs to train for. Defaults to 100.
func WithEpochs(epochs int) TrainerOption {
	return func(t *trainerSettings) error {
		if epochs <= 0 {
			return fmt.Errorf("epochs must be greater than 0, got %d", epochs)
		}
		t.maxEpochs = epochs
		return nil
	}
}

// WithFreezeEmbeddings freezes the embedding layers of the transformer model.
func WithFreezeEmbeddings() TrainerOption {
	return func(t *trainerSettings) error {
		t.freezeEmbeddings = true
		return nil
	}
}

// WithFreezeLayers freezes the given transformer layers. 0 is the first layer; the single value -1
// freezes every layer apart from the last.
func WithFreezeLayers(layers []int) TrainerOption {
	return func(t *trainerSettings) error {
		t.freezeLayers = layers
		return nil
	}
}

// WithEarlyStopping enables early stopping with a patience of 3 epochs and a tolerance of 1e-4.
// It requires TrainerConfig.EvalDataset to be set.
func WithEarlyStopping() TrainerOption {
	return WithEarlyStoppingParams(3, 1e-4)
}

// WithEarlyStoppingParams enables early stopping with an explicit patience and tolerance.
// It requires TrainerConfig.EvalDataset to be set.
func WithEarlyStoppingParams(patience int, tolerance float32) TrainerOption {
	return func(t *trainerSettings) error {
		if patience <= 0 {
			return fmt.Errorf("patience must be greater than 0, got %d", patience)
		}
		if tolerance <= 0 {
			return fmt.Errorf("tolerance must be greater than 0, got %v", tolerance)
		}
		t.earlyStopping = &earlyStopping{patience: patience, tolerance: tolerance}
		return nil
	}
}

// Trainer fine-tunes a pipeline of type T. It is created from a [Session] with
// [Session.NewTrainer], and runs on that session's backend.
//
// A Trainer is a single run: Train, then optionally Save, then Close. Close releases the trainer's
// private copy of the model; a trainer that is never closed is released by [Session.Destroy].
// A Trainer is not safe for concurrent use.
type Trainer[T backends.Pipeline] struct {
	session    *Session
	model      *backends.Model
	config     TrainerConfig[T]
	pipeline   T
	settings   trainerSettings
	statistics TrainingStatistics
	closed     bool
}

// Pipeline returns the pipeline being trained, typed as T.
func (t *Trainer[T]) Pipeline() T {
	return t.pipeline
}

// Statistics returns the per-epoch losses recorded by the last call to Train.
func (t *Trainer[T]) Statistics() TrainingStatistics {
	return t.statistics
}

// Close releases the trainer's copy of the model and removes it from the session. Save first if
// the fine-tuned model is wanted: after Close, Train and Save return an error. Close is safe to call
// more than once, and before or after [Session.Destroy].
func (t *Trainer[T]) Close() error {
	if t.closed {
		return nil
	}
	t.closed = true
	t.session.registryMu.Lock()
	delete(t.session.models, t.model.ID)
	t.session.registryMu.Unlock()
	return t.model.Close()
}

// checkUsable reports why the trainer can no longer run, if it cannot.
func (t *Trainer[T]) checkUsable() error {
	if t.closed {
		return errors.New("trainer has been closed")
	}
	if t.session.options == nil {
		return errors.New("session has been destroyed")
	}
	return nil
}

// NewTrainer creates a trainer for a pipeline of type T, fine-tuning the model at
// config.ModelPath on this session's backend.
//
// T is inferred from config, so no explicit type argument is needed:
//
//	trainer, err := session.NewTrainer(hugot.TrainerConfig[*pipelines.FeatureExtractionPipeline]{
//	    ModelPath:    modelPath,
//	    TrainDataset: dataset,
//	}, hugot.WithEpochs(2))
//
// Training runs on GoMLX for every backend. An ORT session must therefore be created with
// [options.WithGoMLX]; Go and XLA sessions need nothing extra.
//
// The trainer always loads its own copy of the model, even when the session already has one for
// the same path: training writes to the model's weights, so sharing it would silently retrain any
// inference pipeline built on it. The copy is registered with the session under a private key, so no
// pipeline can ever reach it. [Trainer.Close] releases it; [Session.Destroy] releases any trainer
// left open. To serve the fine-tuned model, [Trainer.Save] it and load the saved path as usual.
func (s *Session) NewTrainer[T backends.Pipeline](config TrainerConfig[T], opts ...TrainerOption) (*Trainer[T], error) {
	if s.options == nil {
		return nil, errors.New("session has been destroyed")
	}
	if config.ModelPath == "" {
		return nil, errors.New("a model path is required")
	}
	if config.TrainDataset == nil {
		return nil, errors.New("a training dataset is required")
	}
	if err := checkTrainable[T](); err != nil {
		return nil, err
	}
	// Checked before loading: without WithGoMLX an ORT session would build a native ORT model,
	// only for it to be rejected afterwards for having no GoMLX graph.
	if s.options.Backend == options.BackendORT && !s.options.UseGoMLX {
		return nil, errors.New("training on an ORT session requires creating it with options.WithGoMLX()")
	}

	settings := trainerSettings{maxEpochs: 100}
	for _, opt := range opts {
		if err := opt(&settings); err != nil {
			return nil, err
		}
	}
	// Checked here rather than inside the option, because an option cannot see the config.
	if settings.earlyStopping != nil && config.EvalDataset == nil {
		return nil, errors.New("early stopping requires TrainerConfig.EvalDataset to be set")
	}

	// The model is not registered with the session until the trainer is fully built, so every
	// failure below closes it here rather than leaving it in the registry until Session.Destroy.
	model, err := backends.LoadModel(s.sessionContext, config.ModelPath, config.OnnxFilename, s.options, false)
	if err != nil {
		return nil, err
	}
	trainer, err := newTrainer(s, config, settings, model)
	if err != nil {
		return nil, errors.Join(err, model.Close())
	}
	s.registerTrainingModel(model)
	return trainer, nil
}

// newTrainer builds the trainer around an already loaded model. It does not register or close the
// model; NewTrainer owns both.
func newTrainer[T backends.Pipeline](s *Session, config TrainerConfig[T], settings trainerSettings, model *backends.Model) (*Trainer[T], error) {
	if model.GoMLXModel == nil {
		return nil, fmt.Errorf("training requires a GoMLX model, but the model loaded from %q does not have one", config.ModelPath)
	}

	pipeline, _, err := initializePipeline(s.sessionContext, backends.PipelineConfig[T]{}, model)
	if err != nil {
		return nil, err
	}

	trainer := &Trainer[T]{
		session:  s,
		model:    model,
		config:   config,
		pipeline: pipeline,
		settings: settings,
	}

	if err = trainer.bindDatasets(); err != nil {
		return nil, err
	}
	applyTrainingDefaults(&trainer.config)
	return trainer, nil
}

// registerTrainingModel registers a trainer's private copy of the model with the session, under a
// key unique to this load, so no inference pipeline and no other trainer can share its weights.
// Registering it is what lets Session.Destroy release a trainer that was never closed.
func (s *Session) registerTrainingModel(model *backends.Model) {
	// The pointer makes the key unique while the model is alive; the path keeps it readable.
	model.ID = fmt.Sprintf("trainer:%p:%s", model, model.ID)
	s.registryMu.Lock()
	s.models[model.ID] = model
	s.registryMu.Unlock()
}

// bindDatasets hands the tokenization pipeline to each configured dataset and validates it.
//
// The datasets check that the pipeline is one they can work with, so there is no type switch here:
// a dataset that cannot tokenize with this pipeline says so itself, with a better message than a
// cast in this package could produce.
func (t *Trainer[T]) bindDatasets() error {
	for _, d := range []struct {
		dataset datasets.Dataset
		name    string
	}{
		{t.config.TrainDataset, "train"},
		{t.config.TrainEvalDataset, "train eval"},
		{t.config.EvalDataset, "eval"},
	} {
		if d.dataset == nil {
			continue
		}
		if err := d.dataset.SetTokenizationPipeline(t.pipeline); err != nil {
			return fmt.Errorf("failed to set tokenization pipeline for the %s dataset: %w", d.name, err)
		}
		if err := d.dataset.Validate(); err != nil {
			return fmt.Errorf("invalid %s dataset: %w", d.name, err)
		}
		if t.config.Verbose {
			d.dataset.SetVerbose(true)
		}
	}
	return nil
}

// Train runs the training loop. The context cancels the run.
func (t *Trainer[T]) Train(ctx context.Context) error {
	if err := t.checkUsable(); err != nil {
		return err
	}
	return t.trainGoMLX(ctx)
}

// Save writes the fine-tuned model to path as an ONNX model, alongside a statistics.txt holding
// the per-epoch losses. Tokenizer files are copied from the original model directory.
func (t *Trainer[T]) Save(ctx context.Context, path string) (writeErr error) {
	if path == "" {
		return errors.New("path is required")
	}
	if err := t.checkUsable(); err != nil {
		return err
	}
	ctx = fileutil.WithFileSystem(ctx, t.session.options.FileSystem)

	model := t.pipeline.GetModel()
	if model == nil {
		return errors.New("pipeline model is nil")
	}
	if model.GoMLXModel == nil {
		return errors.New("gomlx model is nil")
	}

	// writeErr is the named result, so the deferred Close errors below reach the caller. For a
	// remote filesystem such as S3, Close is where the upload happens.
	statisticsWriter, err := fileutil.NewFileWriter(ctx, fileutil.PathJoinSafe(path, "statistics.txt"), "")
	if err != nil {
		return err
	}
	defer func() {
		writeErr = errors.Join(writeErr, statisticsWriter.Close())
	}()
	statisticsBytes, err := json.Marshal(t.statistics)
	if err != nil {
		return fmt.Errorf("failed to marshal training statistics: %w", err)
	}
	if _, err = statisticsWriter.Write(statisticsBytes); err != nil {
		return fmt.Errorf("failed to write training statistics: %w", err)
	}

	modelWriter, err := fileutil.NewFileWriter(ctx, fileutil.PathJoinSafe(path, "model.onnx"), "")
	if err != nil {
		return err
	}
	defer func() {
		writeErr = errors.Join(writeErr, modelWriter.Close())
	}()
	writeErr = errors.Join(writeErr, model.GoMLXModel.Save(modelWriter))

	if model.Tokenizer != nil {
		writeErr = errors.Join(writeErr, copyTokenizer(ctx, model.Path, path))
	}
	return writeErr
}

func copyTokenizer(ctx context.Context, from, to string) error {
	toCopy := map[string]bool{
		"special_tokens_map.json": true,
		"tokenizer_config.json":   true,
		"tokenizer.json":          true,
		"vocab.txt":               true,
	}
	walker := func(ctx context.Context, _ string, parent string, info os.FileInfo, _ io.Reader) (toContinue bool, err error) {
		if ctx.Err() != nil {
			return false, ctx.Err()
		}
		if toCopy[info.Name()] {
			if err = fileutil.CopyFile(ctx, fileutil.PathJoinSafe(from, parent, info.Name()), fileutil.PathJoinSafe(to, info.Name())); err != nil {
				return false, err
			}
		}
		return true, nil
	}
	return fileutil.WalkDir(ctx, from, walker)
}
