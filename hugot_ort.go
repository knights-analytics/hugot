//go:build !NOORT || ALL

package hugot

import (
	"context"
	"errors"
	"fmt"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/util"
)

func NewORTSession(opts ...options.WithOption) (*Session, error) {
	session, err := newSession("ORT", ortSession, opts...)
	if err != nil {
		return nil, err
	}
	return session, nil
}

func ortSession(session *Session) (*Session, error) {

	if ort.IsInitialized() {
		return nil, errors.New("another session is currently active, and only one session can be active at one time")
	}

	// set session options and initialise
	if initialised, err := session.initialiseORT(); err != nil {
		if initialised {
			destroyErr := session.Destroy()
			envErr := ort.DestroyEnvironment()
			return nil, errors.Join(err, destroyErr, envErr)
		}
		return nil, err
	}
	session.environmentDestroy = func() error {
		return ort.DestroyEnvironment()
	}

	return session, nil
}

func (s *Session) initialiseORT() (bool, error) {

	o := s.options.ORTOptions
	// Set pre-initialisation options
	if o.LibraryPath != nil {
		ortPathExists, err := util.FileSystem.Exists(context.Background(), *o.LibraryPath)
		if err != nil {
			return false, err
		}
		if !ortPathExists {
			return false, fmt.Errorf("cannot find the ort library at: %s", *o.LibraryPath)
		}
		ort.SetSharedLibraryPath(*o.LibraryPath)
	}

	// Start OnnxRuntime
	if err := ort.InitializeEnvironment(); err != nil {
		return false, err
	}

	if o.Telemetry != nil {
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
	s.options.RuntimeOptions = sessionOptions
	s.options.Destroy = func() error {
		return sessionOptions.Destroy()
	}

	if o.IntraOpNumThreads != nil {
		if err := sessionOptions.SetIntraOpNumThreads(*o.IntraOpNumThreads); err != nil {
			return true, err
		}
	}
	if o.InterOpNumThreads != nil {
		if err := sessionOptions.SetInterOpNumThreads(*o.InterOpNumThreads); err != nil {
			return true, err
		}
	}
	if o.CPUMemArena != nil {
		if err := sessionOptions.SetCpuMemArena(*o.CPUMemArena); err != nil {
			return true, err
		}
	}
	if o.MemPattern != nil {
		if err := sessionOptions.SetMemPattern(*o.MemPattern); err != nil {
			return true, err
		}
	}
	if o.CudaOptions != nil {
		cudaOptions, optErr := ort.NewCUDAProviderOptions()
		if optErr != nil {
			return true, optErr
		}
		if len(o.CudaOptions) > 0 {
			optErr = cudaOptions.Update(o.CudaOptions)
			if optErr != nil {
				return true, optErr
			}
		}
		if err := sessionOptions.AppendExecutionProviderCUDA(cudaOptions); err != nil {
			return true, err
		}
	}
	if o.CoreMLOptions != nil {
		if err := sessionOptions.AppendExecutionProviderCoreML(*o.CoreMLOptions); err != nil {
			return true, err
		}
	}
	if o.DirectMLOptions != nil {
		if err := sessionOptions.AppendExecutionProviderDirectML(*o.DirectMLOptions); err != nil {
			return true, err
		}
	}
	if o.OpenVINOOptions != nil {
		if err := sessionOptions.AppendExecutionProviderOpenVINO(o.OpenVINOOptions); err != nil {
			return true, err
		}
	}
	if o.TensorRTOptions != nil {
		tensorRTOptions, optErr := ort.NewTensorRTProviderOptions()
		if optErr != nil {
			return true, optErr
		}
		if len(o.TensorRTOptions) > 0 {
			optErr = tensorRTOptions.Update(o.TensorRTOptions)
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
