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

// Session allows for the creation of new pipelines and holds the pipeline already created.
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

// NewSession is the main entrypoint to hugot and is used to create a new hugot session object.
// ortLibraryPath should be the path to onnxruntime.so. If it's the empty string, hugot will try
// to load the library from the default location (/usr/lib/onnxruntime.so).
// A new session must be destroyed when it's not needed anymore to avoid memory leaks. See the Destroy method.
// Note moreover that there can be at most one hugot session active (i.e., the Session object is a singleton),
// otherwise NewSession will return an error.
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
			return nil, fmt.Errorf("cannot find the ort library at: %s", ortLibraryPath)
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

// NewTokenClassificationPipeline creates and returns a new token classification pipeline object.
// modelPath should be the path to a folder with the onnx exported transformer model. Name is an identifier
// for the pipeline (see GetTokenClassificationPipeline).
func (s *Session) NewTokenClassificationPipeline(modelPath string, name string, opts ...pipelines.TokenClassificationOption) (*pipelines.TokenClassificationPipeline, error) {
	pipeline, err := pipelines.NewTokenClassificationPipeline(modelPath, name, opts...)
	if err != nil {
		return nil, err
	}
	s.tokenClassificationPipelines[name] = pipeline
	return pipeline, nil
}

// NewTextClassificationPipeline creates and returns a new text classification pipeline object.
// modelPath should be the path to a folder with the onnx exported transformer model. Name is an identifier
// for the pipeline (see GetTextClassificationPipeline).
func (s *Session) NewTextClassificationPipeline(modelPath string, name string, opts ...pipelines.TextClassificationOption) (*pipelines.TextClassificationPipeline, error) {
	pipeline, err := pipelines.NewTextClassificationPipeline(modelPath, name, opts...)
	if err != nil {
		return nil, err
	}
	s.textClassificationPipelines[name] = pipeline
	return pipeline, nil
}

// NewFeatureExtractionPipeline creates and returns a new feature extraction pipeline object.
// modelPath should be the path to a folder with the onnx exported transformer model. Name is an identifier
// for the pipeline (see GetFeatureExtractionPipeline).
func (s *Session) NewFeatureExtractionPipeline(modelPath string, name string) (*pipelines.FeatureExtractionPipeline, error) {
	pipeline, err := pipelines.NewFeatureExtractionPipeline(modelPath, name)
	if err != nil {
		return nil, err
	}
	s.featureExtractionPipelines[name] = pipeline
	return pipeline, nil
}

// GetFeatureExtractionPipeline returns a feature extraction pipeline by name. If the name does not exist, it will return an error.
func (s *Session) GetFeatureExtractionPipeline(name string) (*pipelines.FeatureExtractionPipeline, error) {
	return s.featureExtractionPipelines.GetPipeline(name)
}

// GetTextClassificationPipeline returns a text classification pipeline by name. If the name does not exist, it will return an error.
func (s *Session) GetTextClassificationPipeline(name string) (*pipelines.TextClassificationPipeline, error) {
	return s.textClassificationPipelines.GetPipeline(name)
}

// GetTokenClassificationPipeline returns a token classification pipeline by name. If the name does not exist, it will return an error.
func (s *Session) GetTokenClassificationPipeline(name string) (*pipelines.TokenClassificationPipeline, error) {
	return s.tokenClassificationPipelines.GetPipeline(name)
}

// Destroy deletes the hugot session and onnxruntime environment and all initialized pipelines, freeing memory.
// A hugot session should be destroyed when not neeeded anymore, preferably with a defer() call.
func (s *Session) Destroy() error {
	return errors.Join(
		s.featureExtractionPipelines.Destroy(),
		s.tokenClassificationPipelines.Destroy(),
		s.textClassificationPipelines.Destroy(),
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
	return slices.Concat(s.tokenClassificationPipelines.GetStats(),
		s.textClassificationPipelines.GetStats(),
		s.featureExtractionPipelines.GetStats(),
	)
}
