package hugot

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"reflect"
	"sync"

	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/pipelines"
	"github.com/knights-analytics/hugot/util/fileutil"
)

// Session allows for the creation of new pipelines and holds the pipeline already created.
type Session struct {
	sessionContext       context.Context
	pipelines            map[string]backends.Pipeline
	models               map[string]*backends.Model
	modelLocks           map[string]*sync.Mutex
	pipelineLocks        map[string]*sync.Mutex
	options              *options.Options
	environmentDestroy   func() error
	cancelSessionContext context.CancelFunc
	modelLocksMu         sync.Mutex
	pipelineLocksMu      sync.Mutex
	destroyMu            sync.Mutex
	registryMu           sync.RWMutex
}

func (s *Session) GetModels() map[string]*backends.Model {
	s.registryMu.RLock()
	defer s.registryMu.RUnlock()
	models := make(map[string]*backends.Model, len(s.models))
	maps.Copy(models, s.models)
	return models
}

func (s *Session) getModelLock(modelID string) *sync.Mutex {
	s.modelLocksMu.Lock()
	defer s.modelLocksMu.Unlock()

	if lock, ok := s.modelLocks[modelID]; ok {
		return lock
	}

	lock := &sync.Mutex{}
	s.modelLocks[modelID] = lock
	return lock
}

func (s *Session) removeModelLock(modelID string) {
	s.modelLocksMu.Lock()
	delete(s.modelLocks, modelID)
	s.modelLocksMu.Unlock()
}

func (s *Session) getPipelineLock(name string) *sync.Mutex {
	s.pipelineLocksMu.Lock()
	defer s.pipelineLocksMu.Unlock()

	if lock, ok := s.pipelineLocks[name]; ok {
		return lock
	}

	lock := &sync.Mutex{}
	s.pipelineLocks[name] = lock
	return lock
}

func newSession(ctx context.Context, backend options.Backend, opts ...options.WithOption) (*Session, error) {
	parsedOptions := options.Defaults()
	parsedOptions.Backend = backend
	if !backend.Valid() {
		return nil, fmt.Errorf("runtime %q is not supported", backend)
	}
	// Collect options into a struct, so they can be applied in the correct order later
	if backend == "XLA" {
		parsedOptions.GoMLXOptions.XLA = true
	}
	for _, option := range opts {
		err := option(parsedOptions)
		if err != nil {
			return nil, err
		}
	}

	ctx = fileutil.WithFileSystem(ctx, parsedOptions.FileSystem)
	sessionContext, cancelSessionContext := context.WithCancel(ctx)

	session := &Session{
		pipelines:     map[string]backends.Pipeline{},
		models:        map[string]*backends.Model{},
		modelLocks:    map[string]*sync.Mutex{},
		pipelineLocks: map[string]*sync.Mutex{},
		options:       parsedOptions,
		environmentDestroy: func() error {
			return nil
		},
		sessionContext:       sessionContext,
		cancelSessionContext: cancelSessionContext,
	}

	return session, nil
}

// pipelineConstructor builds a pipeline of some concrete type from an untyped config.
type pipelineConstructor func(ctx context.Context, config any, model *backends.Model) (backends.Pipeline, error)

// pipelineConstructors maps each concrete pipeline type to the constructor that builds it.
// To support a new pipeline type, register it once in the init() below instead of extending
// a set of parallel type switches.
var pipelineConstructors = map[reflect.Type]pipelineConstructor{}

// registerPipeline adapts a typed pipeline constructor into the untyped registry entry, keyed
// by the pipeline's concrete type.
func registerPipeline[T backends.Pipeline](construct func(context.Context, backends.PipelineConfig[T], *backends.Model) (T, error)) {
	var zero T
	pipelineConstructors[reflect.TypeOf(zero)] = func(ctx context.Context, config any, model *backends.Model) (backends.Pipeline, error) {
		typedConfig, ok := config.(backends.PipelineConfig[T])
		if !ok {
			return nil, fmt.Errorf("invalid config type %T for pipeline %T", config, zero)
		}
		return construct(ctx, typedConfig, model)
	}
}

func init() {
	registerPipeline(pipelines.NewTokenClassificationPipeline)
	registerPipeline(pipelines.NewTextClassificationPipeline)
	registerPipeline(pipelines.NewFeatureExtractionPipeline)
	registerPipeline(pipelines.NewZeroShotClassificationPipeline)
	registerPipeline(pipelines.NewCrossEncoderPipeline)
	registerPipeline(pipelines.NewImageClassificationPipeline)
	registerPipeline(pipelines.NewObjectDetectionPipeline)
	registerPipeline(pipelines.NewTextGenerationPipeline)
	registerPipeline(pipelines.NewTabularPipeline)
	registerPipeline(pipelines.NewQuestionAnsweringPipeline)
}

// FeatureExtractionConfig is the configuration for a feature extraction pipeline.
type FeatureExtractionConfig = backends.PipelineConfig[*pipelines.FeatureExtractionPipeline]

// FeatureExtractionOption is an option for a feature extraction pipeline.
type FeatureExtractionOption = backends.PipelineOption[*pipelines.FeatureExtractionPipeline]

// TextClassificationConfig is the configuration for a text classification pipeline.
type TextClassificationConfig = backends.PipelineConfig[*pipelines.TextClassificationPipeline]

// TextClassificationOption is an option for a text classification pipeline.
type TextClassificationOption = backends.PipelineOption[*pipelines.TextClassificationPipeline]

// ZeroShotClassificationConfig is the configuration for a zero shot classification pipeline.
type ZeroShotClassificationConfig = backends.PipelineConfig[*pipelines.ZeroShotClassificationPipeline]

// ZeroShotClassificationOption is an option for a zero shot classification pipeline.
type ZeroShotClassificationOption = backends.PipelineOption[*pipelines.ZeroShotClassificationPipeline]

// TokenClassificationConfig is the configuration for a token classification pipeline.
type TokenClassificationConfig = backends.PipelineConfig[*pipelines.TokenClassificationPipeline]

// TokenClassificationOption is an option for a token classification pipeline.
type TokenClassificationOption = backends.PipelineOption[*pipelines.TokenClassificationPipeline]

// CrossEncoderConfig is the configuration for a cross encoder pipeline.
type CrossEncoderConfig = backends.PipelineConfig[*pipelines.CrossEncoderPipeline]

// CrossEncoderOption is an option for a cross encoder pipeline.
type CrossEncoderOption = backends.PipelineOption[*pipelines.CrossEncoderPipeline]

// ImageClassificationConfig is the configuration for an image classification pipeline.
type ImageClassificationConfig = backends.PipelineConfig[*pipelines.ImageClassificationPipeline]

// ImageClassificationOption is an option for an image classification pipeline.
type ImageClassificationOption = backends.PipelineOption[*pipelines.ImageClassificationPipeline]

// ObjectDetectionConfig is the configuration for an object detection pipeline.
type ObjectDetectionConfig = backends.PipelineConfig[*pipelines.ObjectDetectionPipeline]

// ObjectDetectionOption is an option for an object detection pipeline.
type ObjectDetectionOption = backends.PipelineOption[*pipelines.ObjectDetectionPipeline]

// TextGenerationConfig is the configuration for a text generation pipeline.
type TextGenerationConfig = backends.PipelineConfig[*pipelines.TextGenerationPipeline]

// TextGenerationOption is an option for a text generation pipeline.
type TextGenerationOption = backends.PipelineOption[*pipelines.TextGenerationPipeline]

// TabularConfig is the configuration for a tabular pipeline.
type TabularConfig = backends.PipelineConfig[*pipelines.TabularPipeline]

// TabularOption is an option for a tabular pipeline.
type TabularOption = backends.PipelineOption[*pipelines.TabularPipeline]

// QuestionAnsweringConfig is the configuration for a question answering pipeline.
type QuestionAnsweringConfig = backends.PipelineConfig[*pipelines.QuestionAnsweringPipeline]

// QuestionAnsweringOption is an option for a question answering pipeline.
type QuestionAnsweringOption = backends.PipelineOption[*pipelines.QuestionAnsweringPipeline]

// NewPipeline can be used to create a new pipeline of type T. The initialised pipeline will be returned and it
// will also be stored in the session object so that all created pipelines can be destroyed with session.Destroy()
// at once.
func NewPipeline[T backends.Pipeline](s *Session, pipelineConfig backends.PipelineConfig[T]) (T, error) {
	var pipeline T
	if pipelineConfig.Name == "" {
		return pipeline, errors.New("a name for the pipeline is required")
	}

	pipelineLock := s.getPipelineLock(pipelineConfig.Name)
	pipelineLock.Lock()
	defer pipelineLock.Unlock()

	s.registryMu.RLock()
	_, exists := s.pipelines[pipelineConfig.Name]
	s.registryMu.RUnlock()
	if exists {
		return pipeline, fmt.Errorf("pipeline %s has already been initialised", pipelineConfig.Name)
	}

	constructor, ok := pipelineConstructors[reflect.TypeOf(pipeline)]
	if !ok {
		return pipeline, fmt.Errorf("pipeline type not supported: %T", pipeline)
	}

	// Load model if it has not been loaded already
	modelID := pipelineConfig.ModelPath + ":" + pipelineConfig.OnnxFilename
	modelLock := s.getModelLock(modelID)
	modelLock.Lock()
	defer modelLock.Unlock()

	s.registryMu.RLock()
	model, ok := s.models[modelID]
	s.registryMu.RUnlock()
	if !ok {
		var err error
		model, err = backends.LoadModel(s.sessionContext, pipelineConfig.ModelPath, pipelineConfig.OnnxFilename, s.options, pipeline.IsGenerative())
		if err != nil {
			return pipeline, err
		}
		s.registryMu.Lock()
		s.models[modelID] = model
		s.registryMu.Unlock()
	}

	created, err := constructor(s.sessionContext, pipelineConfig, model)
	if err != nil {
		return pipeline, err
	}

	name := pipelineConfig.Name
	s.registryMu.Lock()
	model.Pipelines[name] = created
	s.pipelines[name] = created
	s.registryMu.Unlock()

	return created.(T), nil
}

// initializePipeline constructs a pipeline of type T from its config using the registered
// constructor, without storing it in a Session. Used by flows (e.g. training) that manage the
// pipeline lifecycle themselves.
func initializePipeline[T backends.Pipeline](sessionContext context.Context, config backends.PipelineConfig[T], model *backends.Model) (T, string, error) {
	var zero T
	constructor, ok := pipelineConstructors[reflect.TypeOf(zero)]
	if !ok {
		return zero, "", fmt.Errorf("pipeline type not supported: %T", zero)
	}
	created, err := constructor(sessionContext, config, model)
	if err != nil {
		return zero, "", err
	}
	return created.(T), config.Name, nil
}

// GetPipeline can be used to retrieve a pipeline of type T with the given name from the session.
func GetPipeline[T backends.Pipeline](s *Session, name string) (T, error) {
	var zero T
	s.registryMu.RLock()
	defer s.registryMu.RUnlock()
	p, ok := s.pipelines[name]
	if !ok {
		return zero, &pipelineNotFoundError{pipelineName: name}
	}
	typed, ok := p.(T)
	if !ok {
		return zero, fmt.Errorf("pipeline %s is not of the requested type %T", name, zero)
	}
	return typed, nil
}

// GetPipelines returns all pipelines of type T currently held by the session, keyed by name.
func GetPipelines[T backends.Pipeline](s *Session) (map[string]T, error) {
	s.registryMu.RLock()
	defer s.registryMu.RUnlock()
	result := map[string]T{}
	for name, p := range s.pipelines {
		if typed, ok := p.(T); ok {
			result[name] = typed
		}
	}
	return result, nil
}

// ClosePipeline removes the pipeline of type T with the given name from the session, tearing down
// the underlying model when no other pipeline depends on it.
func ClosePipeline[T backends.Pipeline](s *Session, name string) error {
	s.registryMu.Lock()
	defer s.registryMu.Unlock()
	p, ok := s.pipelines[name]
	if !ok {
		return nil
	}
	if _, ok := p.(T); !ok {
		return nil
	}

	model := p.GetModel()
	delete(s.pipelines, name)
	delete(model.Pipelines, name)
	if len(model.Pipelines) == 0 {
		delete(s.models, model.ID)
		s.removeModelLock(model.ID)
		return model.Close()
	}
	return nil
}

type pipelineNotFoundError struct {
	pipelineName string
}

func (e *pipelineNotFoundError) Error() string {
	return fmt.Sprintf("Pipeline with name %s not found", e.pipelineName)
}

// GetStatistics returns runtime statistics for all initialized pipelines for profiling purposes. We currently record for each pipeline:
// the total runtime of the tokenization step
// the number of batch calls to the tokenization step
// the average time per tokenization batch call
// the total runtime of the inference (i.e. onnxruntime) step
// the number of batch calls to the onnxruntime inference
// the average time per onnxruntime inference batch call.
func (s *Session) GetStatistics() map[string]backends.PipelineStatistics {
	s.registryMu.RLock()
	defer s.registryMu.RUnlock()
	statistics := map[string]backends.PipelineStatistics{}
	for name, p := range s.pipelines {
		statistics[name] = p.GetStatistics()
	}
	return statistics
}

// PrintStatistics prints runtime statistics for all initialized pipelines to stdout.
func (s *Session) PrintStatistics() {
	statistics := s.GetStatistics()
	for pipelineName, v := range statistics {
		fmt.Printf("Statistics for pipeline %s:\n", pipelineName)
		v.Print()
	}
}

// Destroy deletes the hugot session and onnxruntime environment and all initialized pipelines, freeing memory.
// A hugot session should be destroyed when not neeeded any more, preferably with a defer() call.
func (s *Session) Destroy() error {
	s.destroyMu.Lock()
	defer s.destroyMu.Unlock()
	if s.sessionContext == nil && s.options == nil {
		return nil
	}
	var err error
	s.registryMu.Lock()
	for _, model := range s.models {
		err = errors.Join(err, model.Close())
	}
	s.models = nil
	s.pipelines = nil
	s.registryMu.Unlock()

	if s.options != nil {
		if s.options.BackendOptions != nil {
			err = errors.Join(err, s.options.BackendOptions.Destroy())
		}
		s.options.BackendOptions = nil
		s.options = nil
	}

	if s.environmentDestroy != nil {
		err = errors.Join(err, s.environmentDestroy())
		s.environmentDestroy = nil
	}

	if s.cancelSessionContext != nil {
		s.cancelSessionContext()
		s.cancelSessionContext = nil
		s.sessionContext = nil
	}

	return err
}
