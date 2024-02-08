package hugo

import (
	"fmt"

	"github.com/knights-analytics/hugo/pipelines"
	"github.com/knights-analytics/hugo/utils/checks"
	"github.com/phuslu/log"
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

func (m pipelineMap[T]) Destroy() {
	for _, p := range m {
		p.Destroy()
	}
}

func (m pipelineMap[T]) GetStats() {
	for _, p := range m {
		p.LogStats()
	}
}

func NewSession() *Session {
	session := &Session{
		featureExtractionPipelines:   map[string]*pipelines.FeatureExtractionPipeline{},
		tokenClassificationPipelines: map[string]*pipelines.TokenClassificationPipeline{},
		textClassificationPipelines:  map[string]*pipelines.TextClassificationPipeline{},
	}

	if ort.IsInitialized() {
		log.Fatal().Msg("Another session is currently active and only one session can be active at one time.")
	} else {
		checks.Check(ort.InitializeEnvironment())
	}
	checks.Check(ort.DisableTelemetry())
	return session
}

func (s *Session) NewTokenClassificationPipeline(modelPath string, name string, opts ...pipelines.TokenClassificationOption) *pipelines.TokenClassificationPipeline {
	pipeline := pipelines.NewTokenClassificationPipeline(modelPath, name, opts...)
	s.tokenClassificationPipelines[name] = pipeline
	return pipeline
}

func (s *Session) NewTextClassificationPipeline(modelPath string, name string, opts ...pipelines.TextClassificationOption) *pipelines.TextClassificationPipeline {
	pipeline := pipelines.NewTextClassificationPipeline(modelPath, name, opts...)
	s.textClassificationPipelines[name] = pipeline
	return pipeline
}

func (s *Session) NewFeatureExtractionPipeline(modelPath string, name string) *pipelines.FeatureExtractionPipeline {
	pipeline := pipelines.NewFeatureExtractionPipeline(modelPath, name)
	s.featureExtractionPipelines[name] = pipeline
	return pipeline
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

func (s *Session) Destroy() {
	log.Info().Msg("Destroying pipelines")
	s.featureExtractionPipelines.Destroy()
	s.tokenClassificationPipelines.Destroy()
	s.textClassificationPipelines.Destroy()

	log.Info().Msg("Destroying Onnx Runtime")
	checks.Check(ort.DestroyEnvironment())
}

func (s *Session) GetStats() {
	s.tokenClassificationPipelines.GetStats()
	s.textClassificationPipelines.GetStats()
	s.featureExtractionPipelines.GetStats()
}
