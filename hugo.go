package hugo

import (
	"time"

	"github.com/Knights-Analytics/HuGo/pipelines"
	"github.com/Knights-Analytics/HuGo/utils/checks"
	"github.com/phuslu/log"
	ort "github.com/yalue/onnxruntime_go"
)

type Session struct {
	TokenClassificationPipelines map[string]*pipelines.TokenClassificationPipeline
	TextClassificationPipelines  map[string]*pipelines.TextClassificationPipeline
	FeatureExtractionPipelines   map[string]*pipelines.FeatureExtractionPipeline
}

func NewSession() *Session {
	Session := &Session{
		TokenClassificationPipelines: map[string]*pipelines.TokenClassificationPipeline{},
		TextClassificationPipelines:  map[string]*pipelines.TextClassificationPipeline{},
		FeatureExtractionPipelines:   map[string]*pipelines.FeatureExtractionPipeline{},
	}
	if ort.IsInitialized() {
		log.Fatal().Msg("Another session is currently active and only one session can be active at one time.")
	} else {
		checks.Check(ort.InitializeEnvironment())
	}
	checks.Check(ort.DisableTelemetry())
	return Session
}

func (s *Session) NewTextClassificationPipeline(modelPath string, name string, opts ...pipelines.TextClassificationOption) *pipelines.TextClassificationPipeline {
	pipeline := pipelines.NewTextClassificationPipeline(modelPath, name, opts...)
	s.TextClassificationPipelines[name] = pipeline
	return pipeline
}

func (s *Session) NewTokenClassificationPipeline(modelPath string, name string, opts ...pipelines.TokenClassificationOption) *pipelines.TokenClassificationPipeline {
	pipeline := pipelines.NewTokenClassificationPipeline(modelPath, name, opts...)
	s.TokenClassificationPipelines[name] = pipeline
	return pipeline
}

func (s *Session) NewFeatureExtractionPipeline(modelPath string, name string) *pipelines.FeatureExtractionPipeline {
	pipeline := pipelines.NewFeatureExtractionPipeline(modelPath, name)
	s.FeatureExtractionPipelines[name] = pipeline
	return pipeline
}

func (s *Session) Destroy() {
	log.Info().Msg("Destroying pipelines")
	for _, p := range s.TextClassificationPipelines {
		p.Destroy()
	}
	for _, p := range s.TokenClassificationPipelines {
		p.Destroy()
	}
	log.Info().Msg("Destroying Onnx Runtime")
	checks.Check(ort.DestroyEnvironment())
}

func (s *Session) GetStats() {
	for name, p := range s.FeatureExtractionPipelines {
		logStats(name, p.TokenizerTimings, p.PipelineTimings)
	}
	for name, p := range s.TokenClassificationPipelines {
		logStats(name, p.TokenizerTimings, p.PipelineTimings)
	}
	for name, p := range s.TextClassificationPipelines {
		logStats(name, p.TokenizerTimings, p.PipelineTimings)
	}
}

func logStats(name string, tokenizerTimings *pipelines.Timings, pipelineTimings *pipelines.Timings) {
	log.Info().Msgf("Statistics for pipeline: %s", name)
	log.Info().Msgf("Tokenizer: Total time=%s, Execution count=%d, Average query time=%s", time.Duration(tokenizerTimings.TotalNS), tokenizerTimings.NumCalls, time.Duration(tokenizerTimings.TotalNS/max(1, tokenizerTimings.NumCalls)))
	log.Info().Msgf("ONNX: Total time=%s, Execution count=%d, Average query time=%s", time.Duration(tokenizerTimings.TotalNS), pipelineTimings.NumCalls, time.Duration(pipelineTimings.TotalNS/max(1, pipelineTimings.NumCalls)))
}
