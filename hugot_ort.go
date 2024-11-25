package hugot

import (
	"context"
	"errors"
	"fmt"

	ort "github.com/yalue/onnxruntime_go"

	util "github.com/knights-analytics/hugot/utils"
)

// NewSession is the main entrypoint to hugot and is used to create a new hugot session object.
// ortLibraryPath should be the path to onnxruntime.so. If it's the empty string, hugot will try
// to load the library from the default location (/usr/lib/onnxruntime.so).
// A new session must be destroyed when it's not needed any more to avoid memory leaks. See the Destroy method.
// Note moreover that there can be at most one hugot session active (i.e., the Session object is a singleton),
// otherwise NewSession will return an error.
func ortSession(session *Session, options []WithOption) error {

	if ort.IsInitialized() {
		return errors.New("another session is currently active, and only one session can be active at one time")
	}

	// set session options and initialise
	if initialised, err := session.initialiseORT(options...); err != nil {
		if initialised {
			destroyErr := session.Destroy()
			return errors.Join(err, destroyErr)
		}
		return err
	}

	return nil
}

func (s *Session) initialiseORT(options ...WithOption) (bool, error) {

	// Collect options into a struct, so they can be applied in the correct order later
	o := &ortOptions{}
	for _, option := range options {
		option(o)
	}

	// Set pre-initialisation options
	if o.libraryPath != nil {
		ortPathExists, err := util.FileSystem.Exists(context.Background(), *o.libraryPath)
		if err != nil {
			return false, err
		}
		if !ortPathExists {
			return false, fmt.Errorf("cannot find the ort library at: %s", *o.libraryPath)
		}
		ort.SetSharedLibraryPath(*o.libraryPath)
	}

	// Start OnnxRuntime
	if err := ort.InitializeEnvironment(); err != nil {
		return false, err
	}

	if o.telemetry != nil {
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

	if o.intraOpNumThreads != nil {
		if err := sessionOptions.SetIntraOpNumThreads(*o.intraOpNumThreads); err != nil {
			return true, err
		}
	}
	if o.interOpNumThreads != nil {
		if err := sessionOptions.SetInterOpNumThreads(*o.interOpNumThreads); err != nil {
			return true, err
		}
	}
	if o.cpuMemArena != nil {
		if err := sessionOptions.SetCpuMemArena(*o.cpuMemArena); err != nil {
			return true, err
		}
	}
	if o.memPattern != nil {
		if err := sessionOptions.SetMemPattern(*o.memPattern); err != nil {
			return true, err
		}
	}
	if o.cudaOptions != nil {
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
	if o.coreMLOptions != nil {
		if err := sessionOptions.AppendExecutionProviderCoreML(*o.coreMLOptions); err != nil {
			return true, err
		}
	}
	if o.directMLOptions != nil {
		if err := sessionOptions.AppendExecutionProviderDirectML(*o.directMLOptions); err != nil {
			return true, err
		}
	}
	if o.openVINOOptions != nil {
		if err := sessionOptions.AppendExecutionProviderOpenVINO(o.openVINOOptions); err != nil {
			return true, err
		}
	}
	if o.tensorRTOptions != nil {
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
