package hugot

type ortOptions struct {
	libraryPath       string
	telemetry         bool
	intraOpNumThreads int
	interOpNumThreads int
	cpuMemArena       bool
	cpuMemArenaSet    bool
	memPattern        bool
	memPatternSet     bool
}

// WithOption is the interface for all option functions
type WithOption func(o *ortOptions)

// WithOnnxLibraryPath Use this function to set the path to the "onnxruntime.so" or "onnxruntime.dll" function.
// By default, it will be set to "onnxruntime.so" on non-Windows systems, and "onnxruntime.dll" on Windows.
func WithOnnxLibraryPath(ortLibraryPath string) WithOption {
	return func(o *ortOptions) {
		o.libraryPath = ortLibraryPath
	}
}

// WithTelemetry Enables telemetry events for the onnxruntime environment. Default is off.
func WithTelemetry() WithOption {
	return func(o *ortOptions) {
		o.telemetry = true
	}
}

// WithIntraOpNumThreads Sets the number of threads used to parallelize execution within onnxruntime
// graph nodes. If unspecified, onnxruntime uses the number of physical CPU cores.
func WithIntraOpNumThreads(numThreads int) WithOption {
	return func(o *ortOptions) {
		o.intraOpNumThreads = numThreads
	}
}

// WithInterOpNumThreads Sets the number of threads used to parallelize execution across separate
// onnxruntime graph nodes. If unspecified, onnxruntime uses the number of physical CPU cores.
func WithInterOpNumThreads(numThreads int) WithOption {
	return func(o *ortOptions) {
		o.interOpNumThreads = numThreads
	}
}

// WithCpuMemArena Enable/Disable the usage of the memory arena on CPU.
// Arena may pre-allocate memory for future usage. Default is true.
func WithCpuMemArena(enable bool) WithOption {
	return func(o *ortOptions) {
		o.cpuMemArena = enable
		o.cpuMemArenaSet = true
	}
}

// WithMemPattern Enable/Disable the memory pattern optimization.
// If this is enabled memory is preallocated if all shapes are known. Default is true.
func WithMemPattern(enable bool) WithOption {
	return func(o *ortOptions) {
		o.memPattern = enable
		o.memPatternSet = true
	}
}
