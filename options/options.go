package options

import (
	"fmt"
	"runtime"

	"github.com/knights-analytics/hugot/util/fileutil"
)

type Options struct {
	BackendOptions any
	ORTOptions     *OrtOptions
	GoMLXOptions   *GoMLXOptions
	Destroy        func() error
	Backend        string
}

func Defaults() *Options {
	_, libraryDirDefault, libraryPathDefault := getDefaultLibraryPaths()
	return &Options{
		ORTOptions: &OrtOptions{
			LibraryDir:  &libraryDirDefault,
			LibraryPath: &libraryPathDefault,
		},
		GoMLXOptions: &GoMLXOptions{},
		Destroy: func() error {
			return nil
		},
	}
}

func getDefaultLibraryPaths() (string, string, string) {
	switch runtime.GOOS {
	case "windows":
		return `onnxruntime.dll`, `.\`, `.\onnxuntime.dll`
	case "darwin":
		return "libonnxruntime.dylib", "/usr/local/lib", "/usr/local/lib/libonnxruntime.dylib"
	default:
		return "libonnxruntime.so", "/usr/lib", "/usr/lib/libonnxruntime.so"
	}
}

type GraphOptimizationLevel int

const (
	GraphOptimizationLevelDisableAll     GraphOptimizationLevel = 0
	GraphOptimizationLevelEnableBasic    GraphOptimizationLevel = 1
	GraphOptimizationLevelEnableExtended GraphOptimizationLevel = 2
	GraphOptimizationLevelEnableAll      GraphOptimizationLevel = 99
)

type LoggingLevel int

const (
	LoggingLevelVerbose LoggingLevel = 0
	LoggingLevelInfo    LoggingLevel = 1
	LoggingLevelWarning LoggingLevel = 2
	LoggingLevelError   LoggingLevel = 3
	LoggingLevelFatal   LoggingLevel = 4
)

type OrtOptions struct {
	LibraryPath             *string
	LibraryDir              *string
	Telemetry               *bool
	IntraOpNumThreads       *int
	InterOpNumThreads       *int
	CPUMemArena             *bool
	MemPattern              *bool
	ParallelExecutionMode   *bool
	IntraOpSpinning         *bool
	InterOpSpinning         *bool
	LogSeverityLevel        *LoggingLevel
	EnvLoggingLevel         *LoggingLevel
	GraphOptimizationLevel  *GraphOptimizationLevel
	CudaOptions             map[string]string
	CoreMLOptions           map[string]string
	DirectMLOptions         *int
	OpenVINOOptions         map[string]string
	TensorRTOptions         map[string]string
	ExtraExecutionProviders []ExtraExecutionProvider
}

type ExtraExecutionProvider struct {
	Name    string
	Options map[string]string
}
type GoMLXOptions struct {
	// BatchBuckets defines the bucket sizes for batch dimension padding.
	// Coarse bucketing reduces JIT cache pressure by limiting unique shapes.
	// Default: []int{1, 8, 32}
	BatchBuckets []int
	// SequenceBuckets defines the bucket sizes for sequence length padding.
	// Coarse bucketing reduces JIT cache pressure by limiting unique shapes.
	// Default: []int{32, 128, 512}
	SequenceBuckets []int
	Cuda            bool
	XLA             bool
	TPU             bool
}

// WithOption is the interface for all option functions.
type WithOption func(o *Options) error

// WithOnnxLibraryPath (ORT only) Use this function to set the path to the "libonnxuntime.so", "libonnxuntime.dylib" or "onnxruntime.dll" files.
func WithOnnxLibraryPath(ortLibraryPath string) WithOption {
	return func(o *Options) error {
		if o.Backend == "ORT" {
			object, err := fileutil.FileStats(ortLibraryPath)
			if err != nil {
				return fmt.Errorf("failed to access ONNX Runtime library path %q: %w", ortLibraryPath, err)
			}
			if !object.IsDir() {
				return fmt.Errorf("%s is not a directory", ortLibraryPath)
			}

			libraryName, _, _ := getDefaultLibraryPaths()
			ortLibraryFullPath := fileutil.PathJoinSafe(ortLibraryPath, libraryName)
			exists, err := fileutil.FileExists(ortLibraryPath)
			if err != nil {
				return fmt.Errorf("error checking for existence of ONNX Runtime library file: %w", err)
			}
			if !exists {
				return fmt.Errorf("ONNX Runtime library %s does not exist at %q", libraryName, ortLibraryPath)
			}
			o.ORTOptions.LibraryPath = &ortLibraryFullPath
			o.ORTOptions.LibraryDir = &ortLibraryPath
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

// WithCPUMemArena (ORT only) Enable/Disable the usage of the memory arena on CPU.
// Arena may pre-allocate memory for future usage. Default is true.
func WithCPUMemArena(enable bool) WithOption {
	return func(o *Options) error {
		if o.Backend == "ORT" {
			o.ORTOptions.CPUMemArena = &enable
			return nil
		}
		return fmt.Errorf("WithCPUMemArena is only supported for ORT backend")
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

// WithTPU (XLA only) Use this function to enable TPU acceleration for the XLA backend.
// Requires libtpu.so to be available (pre-installed on GKE TPU nodes).
// Set PJRT_PLUGIN_LIBRARY_PATH to the directory containing pjrt_plugin_tpu.so or libtpu.so.
func WithTPU() WithOption {
	return func(o *Options) error {
		if o.Backend == "XLA" {
			o.GoMLXOptions.TPU = true
			return nil
		}
		return fmt.Errorf("WithTPU is only supported for XLA backend")
	}
}

// WithGoMLXBatchBuckets (XLA and GO only) sets the bucket sizes for batch dimension padding.
// Inputs are padded to the smallest bucket >= their batch size.
// Fewer/coarser buckets reduce JIT cache pressure but increase padding overhead.
// Default is []int{1, 8, 32}.
// IMPORTANT: Ensure MaxCache >= len(BatchBuckets) * len(SequenceBuckets).
func WithGoMLXBatchBuckets(buckets []int) WithOption {
	return func(o *Options) error {
		if o.Backend == "XLA" || o.Backend == "GO" {
			o.GoMLXOptions.BatchBuckets = buckets
			return nil
		}
		return fmt.Errorf("WithGoMLXBatchBuckets is only supported for XLA and Go backends")
	}
}

// WithGoMLXSequenceBuckets (XLA and GO only) sets the bucket sizes for sequence length padding.
// Inputs are padded to the smallest bucket >= their sequence length.
// Fewer/coarser buckets reduce JIT cache pressure but increase padding overhead.
// Default is []int{32, 128, 512}.
// IMPORTANT: Ensure MaxCache >= len(BatchBuckets) * len(SequenceBuckets).
func WithGoMLXSequenceBuckets(buckets []int) WithOption {
	return func(o *Options) error {
		if o.Backend == "XLA" || o.Backend == "GO" {
			o.GoMLXOptions.SequenceBuckets = buckets
			return nil
		}
		return fmt.Errorf("WithGoMLXSequenceBuckets is only supported for XLA and Go backends")
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

// WithLogSeverityLevel (ORT only) Sets the log severity level for the session.
func WithLogSeverityLevel(level LoggingLevel) WithOption {
	return func(o *Options) error {
		if o.Backend == "ORT" {
			o.ORTOptions.LogSeverityLevel = &level
			return nil
		}
		return fmt.Errorf("WithLogSeverityLevel is only supported for ORT backend")
	}
}

// WithEnvLoggingLevel (ORT only) Sets the log severity level for the environment.
func WithEnvLoggingLevel(level LoggingLevel) WithOption {
	return func(o *Options) error {
		if o.Backend == "ORT" {
			o.ORTOptions.EnvLoggingLevel = &level
			return nil
		}
		return fmt.Errorf("WithEnvLoggingLevel is only supported for ORT backend")
	}
}

// WithGraphOptimizationLevel (ORT only) Sets the graph optimization level for the session.
func WithGraphOptimizationLevel(level GraphOptimizationLevel) WithOption {
	return func(o *Options) error {
		if o.Backend == "ORT" {
			o.ORTOptions.GraphOptimizationLevel = &level
			return nil
		}
		return fmt.Errorf("WithGraphOptimizationLevel is only supported for ORT backend")
	}
}

// WithExtraExecutionProvider (ORT only) Adds an extra execution provider to the session.
func WithExtraExecutionProvider(name string, options map[string]string) WithOption {
	return func(o *Options) error {
		if o.Backend == "ORT" {
			o.ORTOptions.ExtraExecutionProviders = append(o.ORTOptions.ExtraExecutionProviders, ExtraExecutionProvider{
				Name:    name,
				Options: options,
			})
			return nil
		}
		return fmt.Errorf("WithExtraExecutionProvider is only supported for ORT backend")
	}
}
