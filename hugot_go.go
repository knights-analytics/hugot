package hugot

import (
	"errors"
	"fmt"

	"github.com/knights-analytics/hugot/pipelines"
)

type GoSession struct {
	*Session
}

func NewGoSession() (*GoSession, error) {

	session := &Session{
		featureExtractionPipelines:      map[string]*pipelines.FeatureExtractionPipeline{},
		textClassificationPipelines:     map[string]*pipelines.TextClassificationPipeline{},
		tokenClassificationPipelines:    map[string]*pipelines.TokenClassificationPipeline{},
		zeroShotClassificationPipelines: map[string]*pipelines.ZeroShotClassificationPipeline{},
	}

	return &GoSession{session}, nil
}

// NewGoPipeline can be used to create a new pipeline of type T. The initialised pipeline will be returned and it
// will also be stored in the session object so that all created pipelines can be destroyed with session.Destroy()
// at once.
func NewGoPipeline[T pipelines.Pipeline](s *GoSession, pipelineConfig pipelines.PipelineConfig[T]) (T, error) {
	var pipeline T
	if pipelineConfig.Name == "" {
		return pipeline, errors.New("a name for the pipeline is required")
	}

	_, getError := GetPipeline[T](s.Session, pipelineConfig.Name)
	var notFoundError *pipelineNotFoundError
	if getError == nil {
		return pipeline, fmt.Errorf("pipeline %s has already been initialised", pipelineConfig.Name)
	} else if !errors.As(getError, &notFoundError) {
		return pipeline, getError
	}

	switch any(pipeline).(type) {
	case *pipelines.TokenClassificationPipeline:
		config := any(pipelineConfig).(pipelines.PipelineConfig[*pipelines.TokenClassificationPipeline])
		pipelineInitialised, err := pipelines.NewTokenClassificationPipelineGo(config)
		if err != nil {
			return pipeline, err
		}
		s.tokenClassificationPipelines[config.Name] = pipelineInitialised
		pipeline = any(pipelineInitialised).(T)
	case *pipelines.TextClassificationPipeline:
		config := any(pipelineConfig).(pipelines.PipelineConfig[*pipelines.TextClassificationPipeline])
		pipelineInitialised, err := pipelines.NewTextClassificationPipelineGo(config)
		if err != nil {
			return pipeline, err
		}
		s.textClassificationPipelines[config.Name] = pipelineInitialised
		pipeline = any(pipelineInitialised).(T)
	case *pipelines.FeatureExtractionPipeline:
		config := any(pipelineConfig).(pipelines.PipelineConfig[*pipelines.FeatureExtractionPipeline])
		pipelineInitialised, err := pipelines.NewFeatureExtractionPipelineGo(config)
		if err != nil {
			return pipeline, err
		}
		s.featureExtractionPipelines[config.Name] = pipelineInitialised
		pipeline = any(pipelineInitialised).(T)
	case *pipelines.ZeroShotClassificationPipeline:
		config := any(pipelineConfig).(pipelines.PipelineConfig[*pipelines.ZeroShotClassificationPipeline])
		pipelineInitialised, err := pipelines.NewZeroShotClassificationPipelineGo(config)
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

func (s *GoSession) Destroy() error {
	return errors.Join(
		s.featureExtractionPipelines.Destroy(),
		s.tokenClassificationPipelines.Destroy(),
		s.textClassificationPipelines.Destroy(),
		s.zeroShotClassificationPipelines.Destroy(),
	)
}
