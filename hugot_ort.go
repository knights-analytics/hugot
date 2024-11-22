package hugot

import (
	"context"
	"errors"
	"fmt"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/knights-analytics/hugot/pipelines"
	util "github.com/knights-analytics/hugot/utils"
)

type ORTSession struct {
	*Session
	ortOptions *ort.SessionOptions
}

// NewORTSession is the main entrypoint to hugot and is used to create a new hugot session object.
// ortLibraryPath should be the path to onnxruntime.so. If it's the empty string, hugot will try
// to load the library from the default location (/usr/lib/onnxruntime.so).
// A new session must be destroyed when it's not needed any more to avoid memory leaks. See the Destroy method.
// Note moreover that there can be at most one hugot session active (i.e., the Session object is a singleton),
// otherwise NewSession will return an error.
func NewORTSession(options ...WithOption) (*ORTSession, error) {

	if ort.IsInitialized() {
		return nil, errors.New("another session is currently active, and only one session can be active at one time")
	}

	session := &Session{
		featureExtractionPipelines:      map[string]*pipelines.FeatureExtractionPipeline{},
		textClassificationPipelines:     map[string]*pipelines.TextClassificationPipeline{},
		tokenClassificationPipelines:    map[string]*pipelines.TokenClassificationPipeline{},
		zeroShotClassificationPipelines: map[string]*pipelines.ZeroShotClassificationPipeline{},
	}

	ortSession := &ORTSession{Session: session}

	// set session options and initialise
	if initialised, err := ortSession.initialiseORT(options...); err != nil {
		if initialised {
			destroyErr := ortSession.Destroy()
			return nil, errors.Join(err, destroyErr)
		}
		return nil, err
	}

	return ortSession, nil
}

func (s *ORTSession) initialiseORT(options ...WithOption) (bool, error) {

	// Collect options into a struct, so they can be applied in the correct order later
	o := &ortOptions{}
	for _, option := range options {
		option(o)
	}

	// Set pre-initialisation options
	if o.libraryPath != "" {
		ortPathExists, err := util.FileSystem.Exists(context.Background(), o.libraryPath)
		if err != nil {
			return false, err
		}
		if !ortPathExists {
			return false, fmt.Errorf("cannot find the ort library at: %s", o.libraryPath)
		}
		ort.SetSharedLibraryPath(o.libraryPath)
	}

	// Start OnnxRuntime
	if err := ort.InitializeEnvironment(); err != nil {
		return false, err
	}

	if o.telemetry {
		if err := ort.EnableTelemetry(); err != nil {
			return true, err
		}
	} else {
		if err := ort.DisableTelemetry(); err != nil {
			return true, err
		}
	}

	// Create session options for use in all pipelines
	sessionOptions, optionsError := ort.NewSessionOptions()
	if optionsError != nil {
		return true, optionsError
	}
	s.ortOptions = sessionOptions

	if o.intraOpNumThreads != 0 {
		if err := sessionOptions.SetIntraOpNumThreads(o.intraOpNumThreads); err != nil {
			return true, err
		}
	}
	if o.interOpNumThreads != 0 {
		if err := sessionOptions.SetInterOpNumThreads(o.interOpNumThreads); err != nil {
			return true, err
		}
	}
	if o.cpuMemArenaSet {
		if err := sessionOptions.SetCpuMemArena(o.cpuMemArena); err != nil {
			return true, err
		}
	}
	if o.memPatternSet {
		if err := sessionOptions.SetMemPattern(o.memPattern); err != nil {
			return true, err
		}
	}
	if o.cudaOptionsSet {
		cudaOptions, optErr := ort.NewCUDAProviderOptions()
		if optErr != nil {
			return true, optErr
		}
		if len(o.cudaOptions) > 0 {
			optErr = cudaOptions.Update(o.cudaOptions)
			if optErr != nil {
				return true, optErr
			}
		}
		if err := sessionOptions.AppendExecutionProviderCUDA(cudaOptions); err != nil {
			return true, err
		}
	}
	if o.coreMLOptionsSet {
		if err := sessionOptions.AppendExecutionProviderCoreML(o.coreMLOptions); err != nil {
			return true, err
		}
	}
	if o.directMLOptionsSet {
		if err := sessionOptions.AppendExecutionProviderDirectML(o.directMLOptions); err != nil {
			return true, err
		}
	}
	if o.openVINOOptionsSet {
		if err := sessionOptions.AppendExecutionProviderOpenVINO(o.openVINOOptions); err != nil {
			return true, err
		}
	}
	if o.tensorRTOptionsSet {
		tensorRTOptions, optErr := ort.NewTensorRTProviderOptions()
		if optErr != nil {
			return true, optErr
		}
		if len(o.cudaOptions) > 0 {
			optErr = tensorRTOptions.Update(o.tensorRTOptions)
			if optErr != nil {
				return true, optErr
			}
		}
		if err := sessionOptions.AppendExecutionProviderTensorRT(tensorRTOptions); err != nil {
			return true, err
		}
	}

	return true, nil
}

// NewORTPipeline can be used to create a new pipeline of type T. The initialised pipeline will be returned and it
// will also be stored in the session object so that all created pipelines can be destroyed with session.Destroy()
// at once.
func NewORTPipeline[T pipelines.Pipeline](s *ORTSession, pipelineConfig pipelines.PipelineConfig[T]) (T, error) {
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
		pipelineInitialised, err := pipelines.NewTokenClassificationPipelineORT(config, s.ortOptions)
		if err != nil {
			return pipeline, err
		}
		s.tokenClassificationPipelines[config.Name] = pipelineInitialised
		pipeline = any(pipelineInitialised).(T)
	case *pipelines.TextClassificationPipeline:
		config := any(pipelineConfig).(pipelines.PipelineConfig[*pipelines.TextClassificationPipeline])
		pipelineInitialised, err := pipelines.NewTextClassificationPipelineORT(config, s.ortOptions)
		if err != nil {
			return pipeline, err
		}
		s.textClassificationPipelines[config.Name] = pipelineInitialised
		pipeline = any(pipelineInitialised).(T)
	case *pipelines.FeatureExtractionPipeline:
		config := any(pipelineConfig).(pipelines.PipelineConfig[*pipelines.FeatureExtractionPipeline])
		pipelineInitialised, err := pipelines.NewFeatureExtractionPipelineORT(config, s.ortOptions)
		if err != nil {
			return pipeline, err
		}
		s.featureExtractionPipelines[config.Name] = pipelineInitialised
		pipeline = any(pipelineInitialised).(T)
	case *pipelines.ZeroShotClassificationPipeline:
		config := any(pipelineConfig).(pipelines.PipelineConfig[*pipelines.ZeroShotClassificationPipeline])
		pipelineInitialised, err := pipelines.NewZeroShotClassificationPipelineORT(config, s.ortOptions)
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

// Destroy deletes the hugot session and onnxruntime environment and all initialized pipelines, freeing memory.
// A hugot session should be destroyed when not neeeded any more, preferably with a defer() call.
func (s *ORTSession) Destroy() error {
	return errors.Join(
		s.featureExtractionPipelines.Destroy(),
		s.tokenClassificationPipelines.Destroy(),
		s.textClassificationPipelines.Destroy(),
		s.zeroShotClassificationPipelines.Destroy(),
		s.ortOptions.Destroy(),
		ort.DestroyEnvironment(),
	)
}
