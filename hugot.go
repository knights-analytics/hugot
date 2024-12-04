package hugot

import (
	"errors"
	"fmt"
	"slices"

	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/pipelineBackends"
	"github.com/knights-analytics/hugot/pipelines"
)

// Session allows for the creation of new pipelines and holds the pipeline already created.
type Session struct {
	featureExtractionPipelines      pipelineMap[*pipelines.FeatureExtractionPipeline]
	tokenClassificationPipelines    pipelineMap[*pipelines.TokenClassificationPipeline]
	textClassificationPipelines     pipelineMap[*pipelines.TextClassificationPipeline]
	zeroShotClassificationPipelines pipelineMap[*pipelines.ZeroShotClassificationPipeline]
	models                          map[string]*pipelineBackends.Model
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
		featureExtractionPipelines:      map[string]*pipelines.FeatureExtractionPipeline{},
		textClassificationPipelines:     map[string]*pipelines.TextClassificationPipeline{},
		tokenClassificationPipelines:    map[string]*pipelines.TokenClassificationPipeline{},
		zeroShotClassificationPipelines: map[string]*pipelines.ZeroShotClassificationPipeline{},
		models:                          map[string]*pipelineBackends.Model{},
		options:                         parsedOptions,
		environmentDestroy: func() error {
			return nil
		},
	}

	return session, nil
}

type pipelineMap[T pipelineBackends.Pipeline] map[string]T

func (m pipelineMap[T]) GetStats() []string {
	var stats []string
	for _, p := range m {
		stats = append(stats, p.GetStats()...)
	}
	return stats
}

// FeatureExtractionConfig is the configuration for a feature extraction pipeline
type FeatureExtractionConfig = pipelineBackends.PipelineConfig[*pipelines.FeatureExtractionPipeline]

// FeatureExtractionOption is an option for a feature extraction pipeline
type FeatureExtractionOption = pipelineBackends.PipelineOption[*pipelines.FeatureExtractionPipeline]

// TextClassificationConfig is the configuration for a text classification pipeline
type TextClassificationConfig = pipelineBackends.PipelineConfig[*pipelines.TextClassificationPipeline]

// TextClassificationOption is an option for a text classification pipeline
type TextClassificationOption = pipelineBackends.PipelineOption[*pipelines.TextClassificationPipeline]

// ZeroShotClassificationConfig is the configuration for a zero shot classification pipeline
type ZeroShotClassificationConfig = pipelineBackends.PipelineConfig[*pipelines.ZeroShotClassificationPipeline]

// ZeroShotClassificationOption is an option for a zero shot classification pipeline
type ZeroShotClassificationOption = pipelineBackends.PipelineOption[*pipelines.ZeroShotClassificationPipeline]

// TokenClassificationConfig is the configuration for a token classification pipeline
type TokenClassificationConfig = pipelineBackends.PipelineConfig[*pipelines.TokenClassificationPipeline]

// TokenClassificationOption is an option for a token classification pipeline
type TokenClassificationOption = pipelineBackends.PipelineOption[*pipelines.TokenClassificationPipeline]

// NewPipeline can be used to create a new pipeline of type T. The initialised pipeline will be returned and it
// will also be stored in the session object so that all created pipelines can be destroyed with session.Destroy()
// at once.
func NewPipeline[T pipelineBackends.Pipeline](s *Session, pipelineConfig pipelineBackends.PipelineConfig[T]) (T, error) {
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
		model = &pipelineBackends.Model{
			Path:         pipelineConfig.ModelPath,
			OnnxFilename: pipelineConfig.OnnxFilename,
		}

		err := pipelineBackends.LoadOnnxModelBytes(model)
		if err != nil {
			return pipeline, err
		}

		err = pipelineBackends.CreateModelBackend(model, s.options)
		if err != nil {
			return pipeline, err
		}

		tkErr := pipelineBackends.LoadTokenizer(model, s.options)
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
	case *pipelines.TokenClassificationPipeline:
		config := any(pipelineConfig).(pipelineBackends.PipelineConfig[*pipelines.TokenClassificationPipeline])
		pipelineInitialised, err := pipelines.NewTokenClassificationPipeline(config, s.options, model)
		if err != nil {
			return pipeline, err
		}
		s.tokenClassificationPipelines[config.Name] = pipelineInitialised
		pipeline = any(pipelineInitialised).(T)
	case *pipelines.TextClassificationPipeline:
		config := any(pipelineConfig).(pipelineBackends.PipelineConfig[*pipelines.TextClassificationPipeline])
		pipelineInitialised, err := pipelines.NewTextClassificationPipeline(config, s.options, model)
		if err != nil {
			return pipeline, err
		}
		s.textClassificationPipelines[config.Name] = pipelineInitialised
		pipeline = any(pipelineInitialised).(T)
	case *pipelines.FeatureExtractionPipeline:
		config := any(pipelineConfig).(pipelineBackends.PipelineConfig[*pipelines.FeatureExtractionPipeline])
		pipelineInitialised, err := pipelines.NewFeatureExtractionPipeline(config, s.options, model)
		if err != nil {
			return pipeline, err
		}
		s.featureExtractionPipelines[config.Name] = pipelineInitialised
		pipeline = any(pipelineInitialised).(T)
	case *pipelines.ZeroShotClassificationPipeline:
		config := any(pipelineConfig).(pipelineBackends.PipelineConfig[*pipelines.ZeroShotClassificationPipeline])
		pipelineInitialised, err := pipelines.NewZeroShotClassificationPipeline(config, s.options, model)
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
func GetPipeline[T pipelineBackends.Pipeline](s *Session, name string) (T, error) {
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
