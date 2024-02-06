package hugo

import (
	"github.com/Knights-Analytics/HuGo/pipelines"
	"github.com/Knights-Analytics/HuGo/utils/checks"
	"github.com/phuslu/log"
	ort "github.com/yalue/onnxruntime_go"
)

type Session struct {
	Pipelines map[string]pipelines.Pipeline
}

func NewSession() *Session {
	session := &Session{
		Pipelines: map[string]pipelines.Pipeline{},
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
	s.Pipelines[name] = pipeline
	return pipeline
}

func (s *Session) NewTextClassificationPipeline(modelPath string, name string, opts ...pipelines.TextClassificationOption) *pipelines.TextClassificationPipeline {
	pipeline := pipelines.NewTextClassificationPipeline(modelPath, name, opts...)
	s.Pipelines[name] = pipeline
	return pipeline
}

func (s *Session) NewFeatureExtractionPipeline(modelPath string, name string) *pipelines.FeatureExtractionPipeline {
	pipeline := pipelines.NewFeatureExtractionPipeline(modelPath, name)
	s.Pipelines[name] = pipeline
	return pipeline
}

func (s *Session) Destroy() {
	log.Info().Msg("Destroying pipelines")
	for _, p := range s.Pipelines {
		p.Destroy()
	}
	log.Info().Msg("Destroying Onnx Runtime")
	checks.Check(ort.DestroyEnvironment())
}

func (s *Session) GetStats() {
	for _, p := range s.Pipelines {
		p.LogStats()
	}
}
