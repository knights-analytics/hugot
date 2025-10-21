package options

import (
	"fmt"
)

type Options struct {
	Backend        string
	ORTOptions     *OrtOptions
	GoMLXOptions   *GoMLXOptions
	Destroy        func() error
	BackendOptions any
}

func Defaults() *Options {
	return &Options{
		ORTOptions:   &OrtOptions{},
		GoMLXOptions: &GoMLXOptions{},
		Destroy: func() error {
			return nil
		},
	}
}

type OrtOptions struct {
	LibraryPath           *string
	Telemetry             *bool
	IntraOpNumThreads     *int
	InterOpNumThreads     *int
	CPUMemArena           *bool
	MemPattern            *bool
	ParallelExecutionMode *bool
	IntraOpSpinning       *bool
	InterOpSpinning       *bool
	CudaOptions           map[string]string
	CoreMLOptions         map[string]string
	DirectMLOptions       *int
	OpenVINOOptions       map[string]string
	TensorRTOptions       map[string]string
}

type GoMLXOptions struct {
	Cuda bool
	XLA  bool
}

// WithOption is the interface for all option functions
type WithOption func(o *Options) error

// WithOnnxLibraryPath (ORT only) Use this function to set the path to the "onnxbackend.so" or "onnxbackend.dll" function.
// By default, it will be set to "onnxbackend.so" on non-Windows systems, and "onnxbackend.dll" on Windows.
func WithOnnxLibraryPath(ortLibraryPath string) WithOption {
	return func(o *Options) error {
		if o.Backend == "ORT" {
			o.ORTOptions.LibraryPath = &ortLibraryPath
			return nil
		}
		return fmt.Errorf("WithOnnxLibraryPath is only supported for ORT backend")
	}
}

// WithTelemetry (ORT only) Enables telemetry events for the onnxbackend environment. Default is off.
func WithTelemetry() WithOption {
	return func(o *Options) error {
		if o.Backend == "ORT" {
			enabled := true
			o.ORTOptions.Telemetry = &enabled
			return nil
		}
		return fmt.Errorf("WithTelemetry is only supported for ORT backend")
	}
}

// WithIntraOpNumThreads (ORT only) Sets the number of threads used to parallelize execution within onnxbackend
// graph nodes. If unspecified, onnxbackend uses the number of physical CPU cores.
func WithIntraOpNumThreads(numThreads int) WithOption {
	return func(o *Options) error {
		if o.Backend == "ORT" {
			o.ORTOptions.IntraOpNumThreads = &numThreads
			return nil
		}
		return fmt.Errorf("WithIntraOpNumThreads is only supported for ORT backend")
	}
}

// WithInterOpNumThreads (ORT only) Sets the number of threads used to parallelize execution across separate
// onnxbackend graph nodes. If unspecified, onnxbackend uses the number of physical CPU cores.
func WithInterOpNumThreads(numThreads int) WithOption {
	return func(o *Options) error {
		if o.Backend == "ORT" {
			o.ORTOptions.InterOpNumThreads = &numThreads
			return nil
		}
		return fmt.Errorf("WithInterOpNumThreads is only supported for ORT backend")
	}
}

// WithCpuMemArena (ORT only) Enable/Disable the usage of the memory arena on CPU.
// Arena may pre-allocate memory for future usage. Default is true.
func WithCpuMemArena(enable bool) WithOption {
	return func(o *Options) error {
		if o.Backend == "ORT" {
			o.ORTOptions.CPUMemArena = &enable
			return nil
		}
		return fmt.Errorf("WithCpuMemArena is only supported for ORT backend")
	}
}

// WithMemPattern (ORT only) Enable/Disable the memory pattern optimization.
// If this is enabled memory is preallocated if all shapes are known. Default is true.
func WithMemPattern(enable bool) WithOption {
	return func(o *Options) error {
		if o.Backend == "ORT" {
			o.ORTOptions.MemPattern = &enable
			return nil
		}
		return fmt.Errorf("WithMemPattern is only supported for ORT backend")
	}
}

// WithExecutionMode sets the parallel execution mode for the ORT backend. Returns an error if the backend is not ORT.
func WithExecutionMode(parallel bool) WithOption {
	return func(o *Options) error {
		if o.Backend == "ORT" {
			o.ORTOptions.ParallelExecutionMode = &parallel
			return nil
		}
		return fmt.Errorf("WithExecutionMode is only supported for ORT backend")
	}
}

// WithIntraOpSpinning configures whether intra-op spinning is enabled for the ORT backend.
// It returns an error if used with a backend other than ORT.
func WithIntraOpSpinning(spinning bool) WithOption {
	return func(o *Options) error {
		if o.Backend == "ORT" {
			o.ORTOptions.IntraOpSpinning = &spinning
			return nil
		}
		return fmt.Errorf("WithIntraOpSpinning is only supported for ORT backend")
	}
}

// WithInterOpSpinning sets the spinning behavior for inter-op threads when the backend is ORT.
// It returns an error if used with a backend other than ORT.
func WithInterOpSpinning(spinning bool) WithOption {
	return func(o *Options) error {
		if o.Backend == "ORT" {
			o.ORTOptions.InterOpSpinning = &spinning
			return nil
		}
		return fmt.Errorf("WithInterOpSpinning is only supported for ORT backend")
	}
}

// WithCuda Use this function to set the options for CUDA provider.
// It takes a map of CUDA parameters as input.
// The options will be applied to the OrtOptions or GoMLXOptions struct, depending on your current backend.
func WithCuda(options map[string]string) WithOption {
	return func(o *Options) error {
		switch o.Backend {
		case "ORT":
			o.ORTOptions.CudaOptions = options
			return nil
		case "XLA":
			o.GoMLXOptions.Cuda = true
			return nil
		default:
			return fmt.Errorf("WithCuda is only supported for ORT or XLA backends")
		}
	}
}

// WithCoreML (ORT only) Use this function to set the CoreML options flags for the ONNX backend configuration.
// The `flags` parameter represents the CoreML options flags.
// The `o.CoreMLOptions` field in `OrtOptions` struct will be set to the provided flags parameter.
func WithCoreML(flags map[string]string) WithOption {
	return func(o *Options) error {
		if o.Backend == "ORT" {
			o.ORTOptions.CoreMLOptions = flags
			return nil
		}
		return fmt.Errorf("WithCoreML is only supported for ORT backend")
	}
}

// WithDirectML (ORT only) Use this function to set the DirectML device ID for the
// onnxbackend. By default, this option is not set.
func WithDirectML(deviceID int) WithOption {
	return func(o *Options) error {
		if o.Backend == "ORT" {
			o.ORTOptions.DirectMLOptions = &deviceID
			return nil
		}
		return fmt.Errorf("WithDirectML is only supported for ORT backend")
	}
}

// WithOpenVINO (ORT only) Use this function to set the OpenVINO options for the OpenVINO execution provider.
// The options parameter should be a map of string keys and string values, representing the configuration options.
// For each key-value pair in the map, the specified option will be set in the OpenVINO execution provider.
// Example usage: WithOpenVINO(map[string]string{"device_type": "CPU", "num_threads": "4"})
// This will configure the OpenVINO execution provider to use CPU as the device type and set the number of threads to 4.
func WithOpenVINO(options map[string]string) WithOption {
	return func(o *Options) error {
		if o.Backend == "ORT" {
			o.ORTOptions.OpenVINOOptions = options
			return nil
		}
		return fmt.Errorf("WithOpenVINO is only supported for ORT backend")
	}
}

// WithTensorRT (ORT only) Use this function to set the options for the TensorRT provider.
// The options parameter should be a pointer to an instance of TensorRTProviderOptions.
// By default, the options will be nil and the TensorRT provider will not be used.
// Example usage:
//
//	options := &onnxbackend_go.TensorRTProviderOptions{
//	    DeviceID: 0,
//	}
//	WithTensorRT(options)
//
// Note: For the TensorRT provider to work, the onnxbackend library must be built with TensorRT support.
func WithTensorRT(options map[string]string) WithOption {
	return func(o *Options) error {
		if o.Backend == "ORT" {
			o.ORTOptions.TensorRTOptions = options
			return nil
		}
		return fmt.Errorf("WithTensorRT is only supported for ORT backend")
	}
}
