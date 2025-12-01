package hugot

import (
	"errors"
	"fmt"
	"slices"

	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/pipelines"
)

// Session allows for the creation of new pipelines and holds the pipeline already created.
type Session struct {
	featureExtractionPipelines      pipelineMap[*pipelines.FeatureExtractionPipeline]
	tokenClassificationPipelines    pipelineMap[*pipelines.TokenClassificationPipeline]
	textClassificationPipelines     pipelineMap[*pipelines.TextClassificationPipeline]
	zeroShotClassificationPipelines pipelineMap[*pipelines.ZeroShotClassificationPipeline]
	crossEncoderPipelines           pipelineMap[*pipelines.CrossEncoderPipeline]
	imageClassificationPipelines    pipelineMap[*pipelines.ImageClassificationPipeline]
	textGenerationPipelines         pipelineMap[*pipelines.TextGenerationPipeline]
	models                          map[string]*backends.Model
	options                         *options.Options
	environmentDestroy              func() error
}

func newSession(backend string, opts ...options.WithOption) (*Session, error) {
	parsedOptions := options.Defaults()
	parsedOptions.Backend = backend
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

	session := &Session{
		featureExtractionPipelines:      map[string]*pipelines.FeatureExtractionPipeline{},
		textClassificationPipelines:     map[string]*pipelines.TextClassificationPipeline{},
		tokenClassificationPipelines:    map[string]*pipelines.TokenClassificationPipeline{},
		zeroShotClassificationPipelines: map[string]*pipelines.ZeroShotClassificationPipeline{},
		crossEncoderPipelines:           map[string]*pipelines.CrossEncoderPipeline{},
		imageClassificationPipelines:    map[string]*pipelines.ImageClassificationPipeline{},
		textGenerationPipelines:         map[string]*pipelines.TextGenerationPipeline{},
		models:                          map[string]*backends.Model{},
		options:                         parsedOptions,
		environmentDestroy: func() error {
			return nil
		},
	}

	return session, nil
}

type pipelineMap[T backends.Pipeline] map[string]T

func (m pipelineMap[T]) GetStats() []string {
	var stats []string
	for _, p := range m {
		stats = append(stats, p.GetStats()...)
	}
	return stats
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

// TextGenerationConfig is the configuration for a text generation pipeline.
type TextGenerationConfig = backends.PipelineConfig[*pipelines.TextGenerationPipeline]

// TextGenerationOption is an option for a text generation pipeline.
type TextGenerationOption = backends.PipelineOption[*pipelines.TextGenerationPipeline]

// NewPipeline can be used to create a new pipeline of type T. The initialised pipeline will be returned and it
// will also be stored in the session object so that all created pipelines can be destroyed with session.Destroy()
// at once.
func NewPipeline[T backends.Pipeline](s *Session, pipelineConfig backends.PipelineConfig[T]) (T, error) {
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

	// Load model if it has not been loaded already
	model, ok := s.models[pipelineConfig.ModelPath]

	var err error
	var name string

	if !ok {
		model, err = backends.LoadModel(pipelineConfig.ModelPath, pipelineConfig.OnnxFilename, s.options)
		if err != nil {
			return pipeline, err
		}
		s.models[pipelineConfig.ModelPath] = model
	}

	pipeline, name, err = InitializePipeline(pipeline, pipelineConfig, s.options, model)
	if err != nil {
		return pipeline, err
	}

	switch typedPipeline := any(pipeline).(type) {
	case *pipelines.TokenClassificationPipeline:
		s.tokenClassificationPipelines[name] = typedPipeline
	case *pipelines.TextClassificationPipeline:
		s.textClassificationPipelines[name] = typedPipeline
	case *pipelines.FeatureExtractionPipeline:
		s.featureExtractionPipelines[name] = typedPipeline
	case *pipelines.ZeroShotClassificationPipeline:
		s.zeroShotClassificationPipelines[name] = typedPipeline
	case *pipelines.CrossEncoderPipeline:
		s.crossEncoderPipelines[name] = typedPipeline
	case *pipelines.ImageClassificationPipeline:
		s.imageClassificationPipelines[name] = typedPipeline
	case *pipelines.TextGenerationPipeline:
		s.textGenerationPipelines[name] = typedPipeline
	default:
		return pipeline, fmt.Errorf("pipeline type not supported: %T", typedPipeline)
	}
	return pipeline, nil
}

func InitializePipeline[T backends.Pipeline](p T, pipelineConfig backends.PipelineConfig[T], options *options.Options, model *backends.Model) (T, string, error) {
	var pipeline T
	var name string

	switch any(p).(type) {
	case *pipelines.TokenClassificationPipeline:
		config := any(pipelineConfig).(backends.PipelineConfig[*pipelines.TokenClassificationPipeline])
		pipelineInitialised, err := pipelines.NewTokenClassificationPipeline(config, options, model)
		if err != nil {
			return pipeline, name, err
		}
		pipeline = any(pipelineInitialised).(T)
		name = config.Name
	case *pipelines.TextClassificationPipeline:
		config := any(pipelineConfig).(backends.PipelineConfig[*pipelines.TextClassificationPipeline])
		pipelineInitialised, err := pipelines.NewTextClassificationPipeline(config, options, model)
		if err != nil {
			return pipeline, name, err
		}
		pipeline = any(pipelineInitialised).(T)
		name = config.Name
	case *pipelines.FeatureExtractionPipeline:
		config := any(pipelineConfig).(backends.PipelineConfig[*pipelines.FeatureExtractionPipeline])
		pipelineInitialised, err := pipelines.NewFeatureExtractionPipeline(config, options, model)
		if err != nil {
			return pipeline, name, err
		}
		pipeline = any(pipelineInitialised).(T)
		name = config.Name
	case *pipelines.ZeroShotClassificationPipeline:
		config := any(pipelineConfig).(backends.PipelineConfig[*pipelines.ZeroShotClassificationPipeline])
		pipelineInitialised, err := pipelines.NewZeroShotClassificationPipeline(config, options, model)
		if err != nil {
			return pipeline, name, err
		}
		pipeline = any(pipelineInitialised).(T)
		name = config.Name
	case *pipelines.CrossEncoderPipeline:
		config := any(pipelineConfig).(backends.PipelineConfig[*pipelines.CrossEncoderPipeline])
		pipelineInitialised, err := pipelines.NewCrossEncoderPipeline(config, options, model)
		if err != nil {
			return pipeline, name, err
		}
		pipeline = any(pipelineInitialised).(T)
		name = config.Name
	case *pipelines.ImageClassificationPipeline:
		config := any(pipelineConfig).(backends.PipelineConfig[*pipelines.ImageClassificationPipeline])
		pipelineInitialised, err := pipelines.NewImageClassificationPipeline(config, options, model)
		if err != nil {
			return pipeline, name, err
		}
		pipeline = any(pipelineInitialised).(T)
		name = config.Name
	case *pipelines.TextGenerationPipeline:
		config := any(pipelineConfig).(backends.PipelineConfig[*pipelines.TextGenerationPipeline])
		pipelineInitialised, err := pipelines.NewTextGenerationPipeline(config, options, model)
		if err != nil {
			return pipeline, name, err
		}
		pipeline = any(pipelineInitialised).(T)
		name = config.Name
	default:
		return pipeline, name, fmt.Errorf("not implemented")
	}

	model.Pipelines[name] = pipeline
	return pipeline, name, nil
}

// GetPipeline can be used to retrieve a pipeline of type T with the given name from the session.
func GetPipeline[T backends.Pipeline](s *Session, name string) (T, error) {
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
	case *pipelines.ZeroShotClassificationPipeline:
		p, ok := s.zeroShotClassificationPipelines[name]
		if !ok {
			return pipeline, &pipelineNotFoundError{pipelineName: name}
		}
		return any(p).(T), nil
	case *pipelines.CrossEncoderPipeline:
		p, ok := s.crossEncoderPipelines[name]
		if !ok {
			return pipeline, &pipelineNotFoundError{pipelineName: name}
		}
		return any(p).(T), nil
	case *pipelines.ImageClassificationPipeline:
		p, ok := s.imageClassificationPipelines[name]
		if !ok {
			return pipeline, &pipelineNotFoundError{pipelineName: name}
		}
		return any(p).(T), nil
	case *pipelines.TextGenerationPipeline:
		p, ok := s.textGenerationPipelines[name]
		if !ok {
			return pipeline, &pipelineNotFoundError{pipelineName: name}
		}
		return any(p).(T), nil
	default:
		return pipeline, errors.New("pipeline type not supported")
	}
}

func ClosePipeline[T backends.Pipeline](s *Session, name string) error {
	var pipeline T
	switch any(pipeline).(type) {
	case *pipelines.TokenClassificationPipeline:
		p, ok := s.tokenClassificationPipelines[name]
		if ok {
			model := p.Model
			delete(s.tokenClassificationPipelines, name)
			delete(model.Pipelines, name)
			if len(model.Pipelines) == 0 {
				delete(s.models, model.Path)
				return model.Destroy()
			}
		}
	case *pipelines.TextClassificationPipeline:
		p, ok := s.textClassificationPipelines[name]
		if ok {
			model := p.Model
			delete(s.textClassificationPipelines, name)
			delete(model.Pipelines, name)
			if len(model.Pipelines) == 0 {
				delete(s.models, model.Path)
				return model.Destroy()
			}
		}
	case *pipelines.FeatureExtractionPipeline:
		p, ok := s.featureExtractionPipelines[name]
		if ok {
			model := p.Model
			delete(s.featureExtractionPipelines, name)
			delete(model.Pipelines, name)
			if len(model.Pipelines) == 0 {
				delete(s.models, model.Path)
				return model.Destroy()
			}
		}
	case *pipelines.ZeroShotClassificationPipeline:
		p, ok := s.zeroShotClassificationPipelines[name]
		if ok {
			model := p.Model
			delete(s.zeroShotClassificationPipelines, name)
			delete(model.Pipelines, name)
			if len(model.Pipelines) == 0 {
				delete(s.models, model.Path)
				return model.Destroy()
			}
		}
	case *pipelines.CrossEncoderPipeline:
		p, ok := s.crossEncoderPipelines[name]
		if ok {
			model := p.Model
			delete(s.crossEncoderPipelines, name)
			delete(model.Pipelines, name)
			if len(model.Pipelines) == 0 {
				delete(s.models, model.Path)
				return model.Destroy()
			}
		}
	case *pipelines.ImageClassificationPipeline:
		p, ok := s.imageClassificationPipelines[name]
		if ok {
			model := p.Model
			delete(s.imageClassificationPipelines, name)
			delete(model.Pipelines, name)
			if len(model.Pipelines) == 0 {
				delete(s.models, model.Path)
				return model.Destroy()
			}
		}
	case *pipelines.TextGenerationPipeline:
		p, ok := s.textGenerationPipelines[name]
		if ok {
			model := p.Model
			delete(s.textGenerationPipelines, name)
			delete(model.Pipelines, name)
			if len(model.Pipelines) == 0 {
				delete(s.models, model.Path)
				return model.Destroy()
			}
		}
	default:
		return errors.New("pipeline type not supported")
	}
	return nil
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
// the average time per onnxruntime inference batch call.
func (s *Session) GetStats() []string {
	return slices.Concat(
		s.tokenClassificationPipelines.GetStats(),
		s.textClassificationPipelines.GetStats(),
		s.featureExtractionPipelines.GetStats(),
		s.zeroShotClassificationPipelines.GetStats(),
		s.crossEncoderPipelines.GetStats(),
		s.textGenerationPipelines.GetStats(),
	)
}

// Destroy deletes the hugot session and onnxruntime environment and all initialized pipelines, freeing memory.
// A hugot session should be destroyed when not neeeded any more, preferably with a defer() call.
func (s *Session) Destroy() error {
	var err error
	for _, model := range s.models {
		err = errors.Join(err, model.Destroy())
	}
	s.models = nil
	s.featureExtractionPipelines = nil
	s.tokenClassificationPipelines = nil
	s.textClassificationPipelines = nil
	s.zeroShotClassificationPipelines = nil
	s.textGenerationPipelines = nil
	s.crossEncoderPipelines = nil

	if s.options != nil {
		err = errors.Join(err, s.options.Destroy())
		s.options = nil
	}

	err = errors.Join(err, s.environmentDestroy())
	return err
}
