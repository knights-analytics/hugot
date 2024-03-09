package hugot

import (
	"context"
	"errors"
	"fmt"
	"slices"

	"github.com/knights-analytics/hugot/pipelines"
	util "github.com/knights-analytics/hugot/utils"
	ort "github.com/yalue/onnxruntime_go"
)

type Session struct {
	featureExtractionPipelines   pipelineMap[*pipelines.FeatureExtractionPipeline]
	tokenClassificationPipelines pipelineMap[*pipelines.TokenClassificationPipeline]
	textClassificationPipelines  pipelineMap[*pipelines.TextClassificationPipeline]
}

type pipelineMap[T pipelines.Pipeline] map[string]T

func (m pipelineMap[T]) GetPipeline(name string) (T, error) {
	p, ok := m[name]
	if !ok {
		return p, fmt.Errorf("pipeline named %s does not exist", name)
	}
	return p, nil
}

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
		stats = slices.Concat(stats, p.GetStats())
	}
	return stats
}

func NewSession(ortLibraryPath string) (*Session, error) {
	session := &Session{
		featureExtractionPipelines:   map[string]*pipelines.FeatureExtractionPipeline{},
		tokenClassificationPipelines: map[string]*pipelines.TokenClassificationPipeline{},
		textClassificationPipelines:  map[string]*pipelines.TextClassificationPipeline{},
	}

	if ortLibraryPath != "" {
		ortPathExists, err := util.FileSystem.Exists(context.Background(), ortLibraryPath)
		if err != nil {
			return nil, err
		}
		if !ortPathExists {
			return nil, fmt.Errorf("Cannot find the ort library at: %s", ortLibraryPath)
		}
		ort.SetSharedLibraryPath(ortLibraryPath)
	}

	if ort.IsInitialized() {
		return nil, errors.New("another session is currently active and only one session can be active at one time")
	} else {
		err := ort.InitializeEnvironment()
		if err != nil {
			return nil, err
		}
	}
	err := ort.DisableTelemetry()
	if err != nil {
		return nil, err
	}
	return session, nil
}

func (s *Session) NewTokenClassificationPipeline(modelPath string, name string, opts ...pipelines.TokenClassificationOption) (*pipelines.TokenClassificationPipeline, error) {
	pipeline, err := pipelines.NewTokenClassificationPipeline(modelPath, name, opts...)
	if err != nil {
		return nil, err
	}
	s.tokenClassificationPipelines[name] = pipeline
	return pipeline, nil
}

func (s *Session) NewTextClassificationPipeline(modelPath string, name string, opts ...pipelines.TextClassificationOption) (*pipelines.TextClassificationPipeline, error) {
	pipeline, err := pipelines.NewTextClassificationPipeline(modelPath, name, opts...)
	if err != nil {
		return nil, err
	}
	s.textClassificationPipelines[name] = pipeline
	return pipeline, nil
}

func (s *Session) NewFeatureExtractionPipeline(modelPath string, name string) (*pipelines.FeatureExtractionPipeline, error) {
	pipeline, err := pipelines.NewFeatureExtractionPipeline(modelPath, name)
	if err != nil {
		return nil, err
	}
	s.featureExtractionPipelines[name] = pipeline
	return pipeline, nil
}

func (s *Session) GetFeatureExtractionPipeline(name string) (*pipelines.FeatureExtractionPipeline, error) {
	return s.featureExtractionPipelines.GetPipeline(name)
}

func (s *Session) GetTextClassificationPipeline(name string) (*pipelines.TextClassificationPipeline, error) {
	return s.textClassificationPipelines.GetPipeline(name)
}

func (s *Session) GetTokenClassificationPipeline(name string) (*pipelines.TokenClassificationPipeline, error) {
	return s.tokenClassificationPipelines.GetPipeline(name)
}

func (s *Session) Destroy() error {

	return errors.Join(
		s.featureExtractionPipelines.Destroy(),
		s.tokenClassificationPipelines.Destroy(),
		s.textClassificationPipelines.Destroy(),
		ort.DestroyEnvironment(),
	)
}

func (s *Session) GetStats() []string {
	return slices.Concat(s.tokenClassificationPipelines.GetStats(),
		s.textClassificationPipelines.GetStats(),
		s.featureExtractionPipelines.GetStats(),
	)
}
