package options

import (
	"fmt"
)

type Options struct {
	Runtime        string
	GoOptions      *GoOptions
	ORTOptions     *OrtOptions
	XLAOptions     *XLAOptions
	Destroy        func() error
	RuntimeOptions any
}

func Defaults() *Options {
	return &Options{
		GoOptions:  &GoOptions{},
		ORTOptions: &OrtOptions{},
		XLAOptions: &XLAOptions{},
		Destroy: func() error {
			return nil
		},
	}
}

type GoOptions struct {
}

type OrtOptions struct {
	LibraryPath       *string
	Telemetry         *bool
	IntraOpNumThreads *int
	InterOpNumThreads *int
	CPUMemArena       *bool
	MemPattern        *bool
	CudaOptions       map[string]string
	CoreMLOptions     *uint32
	DirectMLOptions   *int
	OpenVINOOptions   map[string]string
	TensorRTOptions   map[string]string
}

type XLAOptions struct {
	Cuda bool
}

// WithOption is the interface for all option functions
type WithOption func(o *Options) error

// WithOnnxLibraryPath (ORT only) Use this function to set the path to the "onnxruntime.so" or "onnxruntime.dll" function.
// By default, it will be set to "onnxruntime.so" on non-Windows systems, and "onnxruntime.dll" on Windows.
func WithOnnxLibraryPath(ortLibraryPath string) WithOption {
	return func(o *Options) error {
		if o.Runtime == "ORT" {
			o.ORTOptions.LibraryPath = &ortLibraryPath
			return nil
		} else {
			return fmt.Errorf("WithOnnxLibraryPath is only supported for ORT runtime")
		}
	}
}

// WithTelemetry (ORT only) Enables telemetry events for the onnxruntime environment. Default is off.
func WithTelemetry() WithOption {
	return func(o *Options) error {
		if o.Runtime == "ORT" {
			enabled := true
			o.ORTOptions.Telemetry = &enabled
			return nil
		} else {
			return fmt.Errorf("WithTelemetry is only supported for ORT runtime")
		}
	}
}

// WithIntraOpNumThreads (ORT only) Sets the number of threads used to parallelize execution within onnxruntime
// graph nodes. If unspecified, onnxruntime uses the number of physical CPU cores.
func WithIntraOpNumThreads(numThreads int) WithOption {
	return func(o *Options) error {
		if o.Runtime == "ORT" {
			o.ORTOptions.IntraOpNumThreads = &numThreads
			return nil
		} else {
			return fmt.Errorf("WithIntraOpNumThreads is only supported for ORT runtime")
		}
	}
}

// WithInterOpNumThreads (ORT only) Sets the number of threads used to parallelize execution across separate
// onnxruntime graph nodes. If unspecified, onnxruntime uses the number of physical CPU cores.
func WithInterOpNumThreads(numThreads int) WithOption {
	return func(o *Options) error {
		if o.Runtime == "ORT" {
			o.ORTOptions.InterOpNumThreads = &numThreads
			return nil
		} else {
			return fmt.Errorf("WithInterOpNumThreads is only supported for ORT runtime")
		}
	}
}

// WithCpuMemArena (ORT only) Enable/Disable the usage of the memory arena on CPU.
// Arena may pre-allocate memory for future usage. Default is true.
func WithCpuMemArena(enable bool) WithOption {
	return func(o *Options) error {
		if o.Runtime == "ORT" {
			o.ORTOptions.CPUMemArena = &enable
			return nil
		} else {
			return fmt.Errorf("WithCpuMemArena is only supported for ORT runtime")
		}
	}
}

// WithMemPattern (ORT only) Enable/Disable the memory pattern optimization.
// If this is enabled memory is preallocated if all shapes are known. Default is true.
func WithMemPattern(enable bool) WithOption {
	return func(o *Options) error {
		if o.Runtime == "ORT" {
			o.ORTOptions.MemPattern = &enable
			return nil
		} else {
			return fmt.Errorf("WithMemPattern is only supported for ORT runtime")
		}
	}
}

// WithCuda Use this function to set the options for CUDA provider.
// It takes a pointer to an instance of CUDAProviderOptions struct as input.
// The options will be applied to the OrtOptions struct and the cudaOptionsSet flag will be set to true.
func WithCuda(options map[string]string) WithOption {
	return func(o *Options) error {
		switch o.Runtime {
		case "ORT":
			o.ORTOptions.CudaOptions = options
			return nil
		case "XLA":
			o.XLAOptions.Cuda = true
			return nil
		default:
			return fmt.Errorf("WithCuda is only supported for ORT runtime")
		}
	}
}

// WithCoreML (ORT only) Use this function to set the CoreML options flags for the ONNX runtime configuration.
// The `flags` parameter represents the CoreML options flags.
// The `o.CoreMLOptions` field in `OrtOptions` struct will be set to the provided flags parameter.
// The `o.coreMLOptionsSet` field in `OrtOptions` struct will be set to true.
func WithCoreML(flags uint32) WithOption {
	return func(o *Options) error {
		if o.Runtime == "ORT" {
			o.ORTOptions.CoreMLOptions = &flags
			return nil
		} else {
			return fmt.Errorf("WithCoreML is only supported for ORT runtime")
		}
	}
}

// WithDirectML (ORT only) Use this function to set the DirectML device ID for the
// onnxruntime. By default, this option is not set.
func WithDirectML(deviceID int) WithOption {
	return func(o *Options) error {
		if o.Runtime == "ORT" {
			o.ORTOptions.DirectMLOptions = &deviceID
			return nil
		} else {
			return fmt.Errorf("WithDirectML is only supported for ORT runtime")
		}
	}
}

// WithOpenVINO (ORT only) Use this function to set the OpenVINO options for the OpenVINO execution provider.
// The options parameter should be a map of string keys and string values, representing the configuration options.
// For each key-value pair in the map, the specified option will be set in the OpenVINO execution provider.
// Once the options are set, the openVINOOptionsSet flag in the OrtOptions struct will be set to true.
// Example usage: WithOpenVINO(map[string]string{"device_type": "CPU", "num_threads": "4"})
// This will configure the OpenVINO execution provider to use CPU as the device type and set the number of threads to 4.
func WithOpenVINO(options map[string]string) WithOption {
	return func(o *Options) error {
		if o.Runtime == "ORT" {
			o.ORTOptions.OpenVINOOptions = options
			return nil
		} else {
			return fmt.Errorf("WithOpenVINO is only supported for ORT runtime")
		}
	}
}

// WithTensorRT (ORT only) Use this function to set the options for the TensorRT provider.
// The options parameter should be a pointer to an instance of TensorRTProviderOptions.
// By default, the options will be nil and the TensorRT provider will not be used.
// Example usage:
//
//	options := &onnxruntime_go.TensorRTProviderOptions{
//	    DeviceID: 0,
//	}
//	WithTensorRT(options)
//
// Note: For the TensorRT provider to work, the onnxruntime library must be built with TensorRT support.
func WithTensorRT(options map[string]string) WithOption {
	return func(o *Options) error {
		if o.Runtime == "ORT" {
			o.ORTOptions.TensorRTOptions = options
			return nil
		} else {
			return fmt.Errorf("WithTensorRT is only supported for ORT runtime")
		}
	}
}
