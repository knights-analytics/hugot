package hugot

import (
	"errors"
	"fmt"
	"slices"

	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/pipelines"
	"github.com/knights-analytics/hugot/taskPipelines"
)

// Session allows for the creation of new pipelines and holds the pipeline already created.
type Session struct {
	featureExtractionPipelines      pipelineMap[*taskPipelines.FeatureExtractionPipeline]
	tokenClassificationPipelines    pipelineMap[*taskPipelines.TokenClassificationPipeline]
	textClassificationPipelines     pipelineMap[*taskPipelines.TextClassificationPipeline]
	zeroShotClassificationPipelines pipelineMap[*taskPipelines.ZeroShotClassificationPipeline]
	models                          map[string]*pipelines.Model
	options                         *options.Options
	environmentDestroy              func() error
}

func newSession(runtime string, opts ...options.WithOption) (*Session, error) {

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
		featureExtractionPipelines:      map[string]*taskPipelines.FeatureExtractionPipeline{},
		textClassificationPipelines:     map[string]*taskPipelines.TextClassificationPipeline{},
		tokenClassificationPipelines:    map[string]*taskPipelines.TokenClassificationPipeline{},
		zeroShotClassificationPipelines: map[string]*taskPipelines.ZeroShotClassificationPipeline{},
		models:                          map[string]*pipelines.Model{},
		options:                         parsedOptions,
		environmentDestroy: func() error {
			return nil
		},
	}

	return session, nil
}

type pipelineMap[T pipelines.Pipeline] map[string]T

func (m pipelineMap[T]) GetStats() []string {
	var stats []string
	for _, p := range m {
		stats = append(stats, p.GetStats()...)
	}
	return stats
}

// FeatureExtractionConfig is the configuration for a feature extraction pipeline
type FeatureExtractionConfig = pipelines.PipelineConfig[*taskPipelines.FeatureExtractionPipeline]

// FeatureExtractionOption is an option for a feature extraction pipeline
type FeatureExtractionOption = pipelines.PipelineOption[*taskPipelines.FeatureExtractionPipeline]

// TextClassificationConfig is the configuration for a text classification pipeline
type TextClassificationConfig = pipelines.PipelineConfig[*taskPipelines.TextClassificationPipeline]

// type ZSCConfig = pipelines.PipelineConfig[*pipelines.ZeroShotClassificationPipeline]

type ZeroShotClassificationConfig = pipelines.PipelineConfig[*taskPipelines.ZeroShotClassificationPipeline]

// TextClassificationOption is an option for a text classification pipeline
type TextClassificationOption = pipelines.PipelineOption[*taskPipelines.TextClassificationPipeline]

// TokenClassificationConfig is the configuration for a token classification pipeline
type TokenClassificationConfig = pipelines.PipelineConfig[*taskPipelines.TokenClassificationPipeline]

// TokenClassificationOption is an option for a token classification pipeline
type TokenClassificationOption = pipelines.PipelineOption[*taskPipelines.TokenClassificationPipeline]

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

	// Load model if it has not been loaded already
	model, ok := s.models[pipelineConfig.ModelPath]
	if !ok {
		model = &pipelines.Model{
			Path:         pipelineConfig.ModelPath,
			OnnxFilename: pipelineConfig.OnnxFilename,
		}

		err := pipelines.LoadOnnxModelBytes(model)
		if err != nil {
			return pipeline, err
		}

		err = pipelines.CreateModelBackend(model, s.options)
		if err != nil {
			return pipeline, err
		}

		tkErr := pipelines.LoadTokenizer(model, s.options)
		if tkErr != nil {
			return pipeline, tkErr
		}

		model.Destroy = func() error {
			destroyErr := model.Tokenizer.Destroy()
			switch s.options.Runtime {
			case "ORT":
				destroyErr = errors.Join(destroyErr, model.ORTModel.Destroy())
			case "XLA":
				model.XLAModel.Destroy()
			}
			return destroyErr
		}

		s.models[pipelineConfig.ModelPath] = model
	}

	switch any(pipeline).(type) {
	case *taskPipelines.TokenClassificationPipeline:
		config := any(pipelineConfig).(pipelines.PipelineConfig[*taskPipelines.TokenClassificationPipeline])
		pipelineInitialised, err := taskPipelines.NewTokenClassificationPipeline(config, s.options, model)
		if err != nil {
			return pipeline, err
		}
		s.tokenClassificationPipelines[config.Name] = pipelineInitialised
		pipeline = any(pipelineInitialised).(T)
	case *taskPipelines.TextClassificationPipeline:
		config := any(pipelineConfig).(pipelines.PipelineConfig[*taskPipelines.TextClassificationPipeline])
		pipelineInitialised, err := taskPipelines.NewTextClassificationPipeline(config, s.options, model)
		if err != nil {
			return pipeline, err
		}
		s.textClassificationPipelines[config.Name] = pipelineInitialised
		pipeline = any(pipelineInitialised).(T)
	case *taskPipelines.FeatureExtractionPipeline:
		config := any(pipelineConfig).(pipelines.PipelineConfig[*taskPipelines.FeatureExtractionPipeline])
		pipelineInitialised, err := taskPipelines.NewFeatureExtractionPipeline(config, s.options, model)
		if err != nil {
			return pipeline, err
		}
		s.featureExtractionPipelines[config.Name] = pipelineInitialised
		pipeline = any(pipelineInitialised).(T)
	case *taskPipelines.ZeroShotClassificationPipeline:
		config := any(pipelineConfig).(pipelines.PipelineConfig[*taskPipelines.ZeroShotClassificationPipeline])
		pipelineInitialised, err := taskPipelines.NewZeroShotClassificationPipeline(config, s.options, model)
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

// GetPipeline can be used to retrieve a pipeline of type T with the given name from the session
func GetPipeline[T pipelines.Pipeline](s *Session, name string) (T, error) {
	var pipeline T
	switch any(pipeline).(type) {
	case *taskPipelines.TokenClassificationPipeline:
		p, ok := s.tokenClassificationPipelines[name]
		if !ok {
			return pipeline, &pipelineNotFoundError{pipelineName: name}
		}
		return any(p).(T), nil
	case *taskPipelines.TextClassificationPipeline:
		p, ok := s.textClassificationPipelines[name]
		if !ok {
			return pipeline, &pipelineNotFoundError{pipelineName: name}
		}
		return any(p).(T), nil
	case *taskPipelines.FeatureExtractionPipeline:
		p, ok := s.featureExtractionPipelines[name]
		if !ok {
			return pipeline, &pipelineNotFoundError{pipelineName: name}
		}
		return any(p).(T), nil
	case *taskPipelines.ZeroShotClassificationPipeline:
		p, ok := s.zeroShotClassificationPipelines[name]
		if !ok {
			return pipeline, &pipelineNotFoundError{pipelineName: name}
		}
		return any(p).(T), nil
	default:
		return pipeline, errors.New("pipeline type not supported")
	}
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
	var err error
	for _, model := range s.models {
		err = errors.Join(err, model.Destroy())
	}
	err = errors.Join(
		s.options.Destroy(),
		s.environmentDestroy(),
	)
	return err
}
