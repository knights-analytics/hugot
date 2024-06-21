package hugot

import (
	"context"
	"errors"
	"fmt"

	util "github.com/knights-analytics/hugot/utils"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/knights-analytics/hugot/pipelines"
)

// Session allows for the creation of new pipelines and holds the pipeline already created.
type Session struct {
	featureExtractionPipelines   pipelineMap[*pipelines.FeatureExtractionPipeline]
	tokenClassificationPipelines pipelineMap[*pipelines.TokenClassificationPipeline]
	textClassificationPipelines  pipelineMap[*pipelines.TextClassificationPipeline]
	ortOptions                   *ort.SessionOptions
}

type pipelineMap[T pipelines.Pipeline] map[string]T

func (m pipelineMap[T]) Destroy() error {
	var err error
	for _, p := range m {
		err = p.Destroy()
	}
	return err
}

func (m pipelineMap[T]) GetStats() []string {
	var stats []string
	for _, p := range m {
		stats = append(stats, p.GetStats()...)
	}
	return stats
}

// FeatureExtractionConfig is the configuration for a feature extraction pipeline
type FeatureExtractionConfig = pipelines.PipelineConfig[*pipelines.FeatureExtractionPipeline]

// TextClassificationConfig is the configuration for a text classification pipeline
type TextClassificationConfig = pipelines.PipelineConfig[*pipelines.TextClassificationPipeline]

// TextClassificationOption is an option for a text classification pipeline
type TextClassificationOption = pipelines.PipelineOption[*pipelines.TextClassificationPipeline]

// TokenClassificationConfig is the configuration for a token classification pipeline
type TokenClassificationConfig = pipelines.PipelineConfig[*pipelines.TokenClassificationPipeline]

// // TokenClassificationOption is an option for a token classification pipeline
type TokenClassificationOption = pipelines.PipelineOption[*pipelines.TokenClassificationPipeline]

// FeatureExtractionOption is an option for a feature extraction pipeline
type FeatureExtractionOption = pipelines.PipelineOption[*pipelines.FeatureExtractionPipeline]

// NewSession is the main entrypoint to hugot and is used to create a new hugot session object.
// ortLibraryPath should be the path to onnxruntime.so. If it's the empty string, hugot will try
// to load the library from the default location (/usr/lib/onnxruntime.so).
// A new session must be destroyed when it's not needed anymore to avoid memory leaks. See the Destroy method.
// Note moreover that there can be at most one hugot session active (i.e., the Session object is a singleton),
// otherwise NewSession will return an error.
func NewSession(options ...WithOption) (*Session, error) {

	if ort.IsInitialized() {
		return nil, errors.New("another session is currently active, and only one session can be active at one time")
	}

	session := &Session{
		featureExtractionPipelines:   map[string]*pipelines.FeatureExtractionPipeline{},
		textClassificationPipelines:  map[string]*pipelines.TextClassificationPipeline{},
		tokenClassificationPipelines: map[string]*pipelines.TokenClassificationPipeline{},
	}

	// set session options and initialise
	if initialised, err := session.initialiseORT(options...); err != nil {
		if initialised {
			destroyErr := session.Destroy()
			return nil, errors.Join(err, destroyErr)
		}
		return nil, err
	}

	return session, nil
}

func (s *Session) initialiseORT(options ...WithOption) (bool, error) {

	// Collect options into a struct, so they can be applied in the correct order later
	o := &ortOptions{}
	for _, option := range options {
		option(o)
	}

	// Set pre-initialisation options
	if o.libraryPath != "" {
		ortPathExists, err := util.FileSystem.Exists(context.Background(), o.libraryPath)
		if err != nil {
			return false, err
		}
		if !ortPathExists {
			return false, fmt.Errorf("cannot find the ort library at: %s", o.libraryPath)
		}
		ort.SetSharedLibraryPath(o.libraryPath)
	}

	// Start OnnxRuntime
	if err := ort.InitializeEnvironment(); err != nil {
		return false, err
	}

	if o.telemetry {
		if err := ort.EnableTelemetry(); err != nil {
			return true, err
		}
	} else {
		if err := ort.DisableTelemetry(); err != nil {
			return true, err
		}
	}

	// Create session options for use in all pipelines
	sessionOptions, optionsError := ort.NewSessionOptions()
	if optionsError != nil {
		return true, optionsError
	}
	s.ortOptions = sessionOptions

	if o.intraOpNumThreads != 0 {
		if err := sessionOptions.SetIntraOpNumThreads(o.intraOpNumThreads); err != nil {
			return true, err
		}
	}
	if o.interOpNumThreads != 0 {
		if err := sessionOptions.SetInterOpNumThreads(o.interOpNumThreads); err != nil {
			return true, err
		}
	}
	if o.cpuMemArenaSet {
		if err := sessionOptions.SetCpuMemArena(o.cpuMemArena); err != nil {
			return true, err
		}
	}
	if o.memPatternSet {
		if err := sessionOptions.SetMemPattern(o.memPattern); err != nil {
			return true, err
		}
	}
	if o.cudaOptionsSet {
		cudaOptions, optErr := ort.NewCUDAProviderOptions()
		if optErr != nil {
			return true, optErr
		}
		if len(o.cudaOptions) > 0 {
			optErr = cudaOptions.Update(o.cudaOptions)
			if optErr != nil {
				return true, optErr
			}
		}
		if err := sessionOptions.AppendExecutionProviderCUDA(cudaOptions); err != nil {
			return true, err
		}
	}
	if o.coreMLOptionsSet {
		if err := sessionOptions.AppendExecutionProviderCoreML(o.coreMLOptions); err != nil {
			return true, err
		}
	}
	if o.directMLOptionsSet {
		if err := sessionOptions.AppendExecutionProviderDirectML(o.directMLOptions); err != nil {
			return true, err
		}
	}
	if o.openVINOOptionsSet {
		if err := sessionOptions.AppendExecutionProviderOpenVINO(o.openVINOOptions); err != nil {
			return true, err
		}
	}
	if o.tensorRTOptionsSet {
		tensorRTOptions, optErr := ort.NewTensorRTProviderOptions()
		if optErr != nil {
			return true, optErr
		}
		if len(o.cudaOptions) > 0 {
			optErr = tensorRTOptions.Update(o.tensorRTOptions)
			if optErr != nil {
				return true, optErr
			}
		}
		if err := sessionOptions.AppendExecutionProviderTensorRT(tensorRTOptions); err != nil {
			return true, err
		}
	}

	return true, nil
}

type pipelineNotFoundError struct {
	pipelineName string
}

func (e *pipelineNotFoundError) Error() string {
	return fmt.Sprintf("Pipeline with name %s not found", e.pipelineName)
}

// NewPipeline can be used to create a new pipeline of type T. The initialised pipeline will be returned and it
// will also be stored in the session object so that all created pipelines can be destroyed with session.Destroy()
// at once.
func NewPipeline[T pipelines.Pipeline](s *Session, pipelineConfig pipelines.PipelineConfig[T]) (T, error) {
	var pipeline T
	var err error
	if pipelineConfig.Name == "" {
		return pipeline, errors.New("a name for the pipeline is required")
	}

	_, getError := GetPipeline[T](s, pipelineConfig.Name)
	var notFoundError *pipelineNotFoundError
	if getError == nil {
		return pipeline, fmt.Errorf("pipeline %s has already been initialised", pipelineConfig.Name)
	} else if !errors.As(getError, &notFoundError) {
		return pipeline, getError
	}

	switch any(pipeline).(type) {
	case *pipelines.TokenClassificationPipeline:
		config := any(pipelineConfig).(pipelines.PipelineConfig[*pipelines.TokenClassificationPipeline])
		pipelineInitialised, err := pipelines.NewTokenClassificationPipeline(config, s.ortOptions)
		if err != nil {
			return pipeline, err
		}
		s.tokenClassificationPipelines[config.Name] = pipelineInitialised
		pipeline = any(pipelineInitialised).(T)
	case *pipelines.TextClassificationPipeline:
		config := any(pipelineConfig).(pipelines.PipelineConfig[*pipelines.TextClassificationPipeline])
		pipelineInitialised, err := pipelines.NewTextClassificationPipeline(config, s.ortOptions)
		if err != nil {
			return pipeline, err
		}
		s.textClassificationPipelines[config.Name] = pipelineInitialised
		pipeline = any(pipelineInitialised).(T)
	case *pipelines.FeatureExtractionPipeline:
		config := any(pipelineConfig).(pipelines.PipelineConfig[*pipelines.FeatureExtractionPipeline])
		pipelineInitialised, err := pipelines.NewFeatureExtractionPipeline(config, s.ortOptions)
		if err != nil {
			return pipeline, err
		}
		s.featureExtractionPipelines[config.Name] = pipelineInitialised
		pipeline = any(pipelineInitialised).(T)
	default:
		return pipeline, fmt.Errorf("not implemented")
	}
	return pipeline, err
}

// GetPipeline can be used to retrieve a pipeline of type T with the given name from the session
func GetPipeline[T pipelines.Pipeline](s *Session, name string) (T, error) {
	var pipeline T
	switch any(pipeline).(type) {
	case *pipelines.TokenClassificationPipeline:
		p, ok := s.tokenClassificationPipelines[name]
		if !ok {
			return pipeline, &pipelineNotFoundError{pipelineName: name}
		}
		return any(p).(T), nil
	case *pipelines.TextClassificationPipeline:
		p, ok := s.textClassificationPipelines[name]
		if !ok {
			return pipeline, &pipelineNotFoundError{pipelineName: name}
		}
		return any(p).(T), nil
	case *pipelines.FeatureExtractionPipeline:
		p, ok := s.featureExtractionPipelines[name]
		if !ok {
			return pipeline, &pipelineNotFoundError{pipelineName: name}
		}
		return any(p).(T), nil
	default:
		return pipeline, errors.New("pipeline type not supported")
	}
}

// Destroy deletes the hugot session and onnxruntime environment and all initialized pipelines, freeing memory.
// A hugot session should be destroyed when not neeeded anymore, preferably with a defer() call.
func (s *Session) Destroy() error {
	return errors.Join(
		s.featureExtractionPipelines.Destroy(),
		// s.tokenClassificationPipelines.Destroy(),
		s.textClassificationPipelines.Destroy(),
		s.ortOptions.Destroy(),
		ort.DestroyEnvironment(),
	)
}

// GetStats returns runtime statistics for all initialized pipelines for profiling purposes. We currently record for each pipeline:
// the total runtime of the tokenization step
// the number of batch calls to the tokenization step
// the average time per tokenization batch call
// the total runtime of the inference (i.e. onnxruntime) step
// the number of batch calls to the onnxruntime inference
// the average time per onnxruntime inference batch call
func (s *Session) GetStats() []string {
	// slices.Concat() is not implemented in experimental x/exp/slices package
	return append(append(
		s.tokenClassificationPipelines.GetStats(),
		s.textClassificationPipelines.GetStats()...),
		s.featureExtractionPipelines.GetStats()...,
	)
}

// deprecated methods

// NewTokenClassificationPipeline creates and returns a new token classification pipeline object.
// modelPath should be the path to a folder with the onnx exported transformer model. Name is an identifier
// for the pipeline (see GetTokenClassificationPipeline).
// Deprecated: use NewPipeline
func (s *Session) NewTokenClassificationPipeline(modelPath string, name string, opts ...TokenClassificationOption) (*pipelines.TokenClassificationPipeline, error) {
	config := pipelines.PipelineConfig[*pipelines.TokenClassificationPipeline]{
		ModelPath: modelPath,
		Name:      name,
		Options:   opts,
	}
	return NewPipeline(s, config)
}

// NewTextClassificationPipeline creates and returns a new text classification pipeline object.
// modelPath should be the path to a folder with the onnx exported transformer model. Name is an identifier
// for the pipeline (see GetTextClassificationPipeline).
// Deprecated: use NewPipeline
func (s *Session) NewTextClassificationPipeline(modelPath string, name string, opts ...TextClassificationOption) (*pipelines.TextClassificationPipeline, error) {
	config := pipelines.PipelineConfig[*pipelines.TextClassificationPipeline]{
		ModelPath: modelPath,
		Name:      name,
		Options:   opts,
	}
	return NewPipeline(s, config)
}

// NewFeatureExtractionPipeline creates and returns a new feature extraction pipeline object.
// modelPath should be the path to a folder with the onnx exported transformer model. Name is an identifier
// for the pipeline (see GetFeatureExtractionPipeline).
// Deprecated: use NewPipeline
func (s *Session) NewFeatureExtractionPipeline(modelPath string, name string, opts ...FeatureExtractionOption) (*pipelines.FeatureExtractionPipeline, error) {
	config := pipelines.PipelineConfig[*pipelines.FeatureExtractionPipeline]{
		ModelPath: modelPath,
		Name:      name,
		Options:   opts,
	}
	return NewPipeline(s, config)
}

// GetFeatureExtractionPipeline returns a feature extraction pipeline by name. If the name does not exist, it will return an error.
// Deprecated: use GetPipeline.
func (s *Session) GetFeatureExtractionPipeline(name string) (*pipelines.FeatureExtractionPipeline, error) {
	return GetPipeline[*pipelines.FeatureExtractionPipeline](s, name)
}

// GetTextClassificationPipeline returns a text classification pipeline by name. If the name does not exist, it will return an error.
// Deprecated: use GetPipeline.
func (s *Session) GetTextClassificationPipeline(name string) (*pipelines.TextClassificationPipeline, error) {
	return GetPipeline[*pipelines.TextClassificationPipeline](s, name)
}

// GetTokenClassificationPipeline returns a token classification pipeline by name. If the name does not exist, it will return an error.
// Deprecated: use GetPipeline.
func (s *Session) GetTokenClassificationPipeline(name string) (*pipelines.TokenClassificationPipeline, error) {
	return GetPipeline[*pipelines.TokenClassificationPipeline](s, name)
}
