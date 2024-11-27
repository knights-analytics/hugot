package hugot

import (
	"errors"
	"fmt"
	"slices"

	"github.com/knights-analytics/hugot/hfPipelines"
	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/pipelines"
)

// Session allows for the creation of new pipelines and holds the pipeline already created.
type Session struct {
	featureExtractionPipelines      pipelineMap[*hfPipelines.FeatureExtractionPipeline]
	tokenClassificationPipelines    pipelineMap[*hfPipelines.TokenClassificationPipeline]
	textClassificationPipelines     pipelineMap[*hfPipelines.TextClassificationPipeline]
	zeroShotClassificationPipelines pipelineMap[*hfPipelines.ZeroShotClassificationPipeline]
	options                         *options.Options
	environmentDestroy              func() error
}

func newSession(runtime string, additionalSetup func(*Session) (*Session, error), opts ...options.WithOption) (*Session, error) {

	parsedOptions := options.Defaults()
	parsedOptions.Runtime = runtime
	// Collect options into a struct, so they can be applied in the correct order later
	for _, option := range opts {
		err := option(parsedOptions)
		if err != nil {
			return nil, err
		}
	}

	session := &Session{
		featureExtractionPipelines:      map[string]*hfPipelines.FeatureExtractionPipeline{},
		textClassificationPipelines:     map[string]*hfPipelines.TextClassificationPipeline{},
		tokenClassificationPipelines:    map[string]*hfPipelines.TokenClassificationPipeline{},
		zeroShotClassificationPipelines: map[string]*hfPipelines.ZeroShotClassificationPipeline{},
		options:                         parsedOptions,
		environmentDestroy: func() error {
			return nil
		},
	}

	switch parsedOptions.Runtime {
	case "GO", "XLA":
		// No session specific initialisation for these currently
		return session, nil
	case "ORT":
		return additionalSetup(session)
	default:
		return nil, errors.New("unsupported runtime: " + parsedOptions.Runtime)
	}
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
type FeatureExtractionConfig = pipelines.PipelineConfig[*hfPipelines.FeatureExtractionPipeline]

// FeatureExtractionOption is an option for a feature extraction pipeline
type FeatureExtractionOption = pipelines.PipelineOption[*hfPipelines.FeatureExtractionPipeline]

// TextClassificationConfig is the configuration for a text classification pipeline
type TextClassificationConfig = pipelines.PipelineConfig[*hfPipelines.TextClassificationPipeline]

// type ZSCConfig = pipelines.PipelineConfig[*pipelines.ZeroShotClassificationPipeline]

type ZeroShotClassificationConfig = pipelines.PipelineConfig[*hfPipelines.ZeroShotClassificationPipeline]

// TextClassificationOption is an option for a text classification pipeline
type TextClassificationOption = pipelines.PipelineOption[*hfPipelines.TextClassificationPipeline]

// TokenClassificationConfig is the configuration for a token classification pipeline
type TokenClassificationConfig = pipelines.PipelineConfig[*hfPipelines.TokenClassificationPipeline]

// TokenClassificationOption is an option for a token classification pipeline
type TokenClassificationOption = pipelines.PipelineOption[*hfPipelines.TokenClassificationPipeline]

// GetPipeline can be used to retrieve a pipeline of type T with the given name from the session
func GetPipeline[T pipelines.Pipeline](s *Session, name string) (T, error) {
	var pipeline T
	switch any(pipeline).(type) {
	case *hfPipelines.TokenClassificationPipeline:
		p, ok := s.tokenClassificationPipelines[name]
		if !ok {
			return pipeline, &pipelineNotFoundError{pipelineName: name}
		}
		return any(p).(T), nil
	case *hfPipelines.TextClassificationPipeline:
		p, ok := s.textClassificationPipelines[name]
		if !ok {
			return pipeline, &pipelineNotFoundError{pipelineName: name}
		}
		return any(p).(T), nil
	case *hfPipelines.FeatureExtractionPipeline:
		p, ok := s.featureExtractionPipelines[name]
		if !ok {
			return pipeline, &pipelineNotFoundError{pipelineName: name}
		}
		return any(p).(T), nil
	case *hfPipelines.ZeroShotClassificationPipeline:
		p, ok := s.zeroShotClassificationPipelines[name]
		if !ok {
			return pipeline, &pipelineNotFoundError{pipelineName: name}
		}
		return any(p).(T), nil
	default:
		return pipeline, errors.New("pipeline type not supported")
	}
}

// NewPipeline can be used to create a new pipeline of type T. The initialised pipeline will be returned and it
// will also be stored in the session object so that all created pipelines can be destroyed with session.Destroy()
// at once.
func NewPipeline[T pipelines.Pipeline](s *Session, pipelineConfig pipelines.PipelineConfig[T]) (T, error) {
	var pipeline T
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
	case *hfPipelines.TokenClassificationPipeline:
		config := any(pipelineConfig).(pipelines.PipelineConfig[*hfPipelines.TokenClassificationPipeline])
		pipelineInitialised, err := hfPipelines.NewTokenClassificationPipeline(config, s.options)
		if err != nil {
			return pipeline, err
		}
		s.tokenClassificationPipelines[config.Name] = pipelineInitialised
		pipeline = any(pipelineInitialised).(T)
	case *hfPipelines.TextClassificationPipeline:
		config := any(pipelineConfig).(pipelines.PipelineConfig[*hfPipelines.TextClassificationPipeline])
		pipelineInitialised, err := hfPipelines.NewTextClassificationPipeline(config, s.options)
		if err != nil {
			return pipeline, err
		}
		s.textClassificationPipelines[config.Name] = pipelineInitialised
		pipeline = any(pipelineInitialised).(T)
	case *hfPipelines.FeatureExtractionPipeline:
		config := any(pipelineConfig).(pipelines.PipelineConfig[*hfPipelines.FeatureExtractionPipeline])
		pipelineInitialised, err := hfPipelines.NewFeatureExtractionPipeline(config, s.options)
		if err != nil {
			return pipeline, err
		}
		s.featureExtractionPipelines[config.Name] = pipelineInitialised
		pipeline = any(pipelineInitialised).(T)
	case *hfPipelines.ZeroShotClassificationPipeline:
		config := any(pipelineConfig).(pipelines.PipelineConfig[*hfPipelines.ZeroShotClassificationPipeline])
		pipelineInitialised, err := hfPipelines.NewZeroShotClassificationPipeline(config, s.options)
		if err != nil {
			return pipeline, err
		}
		s.zeroShotClassificationPipelines[config.Name] = pipelineInitialised
		pipeline = any(pipelineInitialised).(T)
	default:
		return pipeline, fmt.Errorf("not implemented")
	}
	return pipeline, nil
}

type pipelineNotFoundError struct {
	pipelineName string
}

func (e *pipelineNotFoundError) Error() string {
	return fmt.Sprintf("Pipeline with name %s not found", e.pipelineName)
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
	return slices.Concat(
		s.tokenClassificationPipelines.GetStats(),
		s.textClassificationPipelines.GetStats(),
		s.featureExtractionPipelines.GetStats(),
		s.zeroShotClassificationPipelines.GetStats(),
	)
}

// Destroy deletes the hugot session and onnxruntime environment and all initialized pipelines, freeing memory.
// A hugot session should be destroyed when not neeeded any more, preferably with a defer() call.
func (s *Session) Destroy() error {
	err := errors.Join(
		s.featureExtractionPipelines.Destroy(),
		s.tokenClassificationPipelines.Destroy(),
		s.textClassificationPipelines.Destroy(),
		s.zeroShotClassificationPipelines.Destroy(),
		s.options.Destroy(),
		s.environmentDestroy(),
	)
	return err
}
