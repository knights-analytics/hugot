package hugot

type ortOptions struct {
	libraryPath       *string
	telemetry         *bool
	intraOpNumThreads *int
	interOpNumThreads *int
	cpuMemArena       *bool
	memPattern        *bool
	cudaOptions       map[string]string
	coreMLOptions     *uint32
	directMLOptions   *int
	openVINOOptions   map[string]string
	tensorRTOptions   map[string]string
}

// WithOption is the interface for all option functions
type WithOption func(o *ortOptions)

// WithOnnxLibraryPath Use this function to set the path to the "onnxruntime.so" or "onnxruntime.dll" function.
// By default, it will be set to "onnxruntime.so" on non-Windows systems, and "onnxruntime.dll" on Windows.
func WithOnnxLibraryPath(ortLibraryPath string) WithOption {
	return func(o *ortOptions) {
		o.libraryPath = &ortLibraryPath
	}
}

// WithTelemetry Enables telemetry events for the onnxruntime environment. Default is off.
func WithTelemetry() WithOption {
	return func(o *ortOptions) {
		enabled := true
		o.telemetry = &enabled
	}
}

// WithIntraOpNumThreads Sets the number of threads used to parallelize execution within onnxruntime
// graph nodes. If unspecified, onnxruntime uses the number of physical CPU cores.
func WithIntraOpNumThreads(numThreads int) WithOption {
	return func(o *ortOptions) {
		o.intraOpNumThreads = &numThreads
	}
}

// WithInterOpNumThreads Sets the number of threads used to parallelize execution across separate
// onnxruntime graph nodes. If unspecified, onnxruntime uses the number of physical CPU cores.
func WithInterOpNumThreads(numThreads int) WithOption {
	return func(o *ortOptions) {
		o.interOpNumThreads = &numThreads
	}
}

// WithCpuMemArena Enable/Disable the usage of the memory arena on CPU.
// Arena may pre-allocate memory for future usage. Default is true.
func WithCpuMemArena(enable bool) WithOption {
	return func(o *ortOptions) {
		o.cpuMemArena = &enable
	}
}

// WithMemPattern Enable/Disable the memory pattern optimization.
// If this is enabled memory is preallocated if all shapes are known. Default is true.
func WithMemPattern(enable bool) WithOption {
	return func(o *ortOptions) {
		o.memPattern = &enable
	}
}

// WithCuda Use this function to set the options for CUDA provider.
// It takes a pointer to an instance of CUDAProviderOptions struct as input.
// The options will be applied to the ortOptions struct and the cudaOptionsSet flag will be set to true.
func WithCuda(options map[string]string) WithOption {
	return func(o *ortOptions) {
		o.cudaOptions = options
	}
}

// WithCoreML Use this function to set the CoreML options flags for the ONNX runtime configuration.
// The `flags` parameter represents the CoreML options flags.
// The `o.coreMLOptions` field in `ortOptions` struct will be set to the provided flags parameter.
// The `o.coreMLOptionsSet` field in `ortOptions` struct will be set to true.
func WithCoreML(flags uint32) WithOption {
	return func(o *ortOptions) {
		o.coreMLOptions = &flags
	}
}

// WithDirectML Use this function to set the DirectML device ID for the
// onnxruntime. By default, this option is not set.
func WithDirectML(deviceID int) WithOption {
	return func(o *ortOptions) {
		o.directMLOptions = &deviceID
	}
}

// WithOpenVINO Use this function to set the OpenVINO options for the OpenVINO execution provider.
// The options parameter should be a map of string keys and string values, representing the configuration options.
// For each key-value pair in the map, the specified option will be set in the OpenVINO execution provider.
// Once the options are set, the openVINOOptionsSet flag in the ortOptions struct will be set to true.
// Example usage: WithOpenVINO(map[string]string{"device_type": "CPU", "num_threads": "4"})
// This will configure the OpenVINO execution provider to use CPU as the device type and set the number of threads to 4.
func WithOpenVINO(options map[string]string) WithOption {
	return func(o *ortOptions) {
		o.openVINOOptions = options
	}
}

// WithTensorRT Use this function to set the options for the TensorRT provider.
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
	return func(o *ortOptions) {
		o.tensorRTOptions = options
	}
}
