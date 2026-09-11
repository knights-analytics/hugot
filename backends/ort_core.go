//go:build cgo && (ORT || ALL)

package backends

import (
	"context"
	"errors"
	"fmt"
	"os"

	ort "github.com/microsoft/onnxruntime/go/onnxruntime"

	"github.com/knights-analytics/hugot/options"
)

type coreORTOptions struct {
	*ort.SessionOptions
}

func (o *coreORTOptions) Destroy() error {
	if o == nil {
		return nil
	}
	return o.Close()
}

type coreORTSession struct {
	session     *ort.Session
	inputNames  []string
	outputNames []string
}

type coreORTTensor struct {
	tensor *ort.Tensor
}

func newCoreInt64Tensor(shape []int64, data []int64) (*coreORTTensor, error) {
	tensor, err := ort.CreateTensor(shape, data)
	if err != nil {
		return nil, err
	}
	return &coreORTTensor{tensor: tensor}, nil
}

func newCoreFloat32Tensor(shape []int64, data []float32) (*coreORTTensor, error) {
	tensor, err := ort.CreateTensor(shape, data)
	if err != nil {
		return nil, err
	}
	return &coreORTTensor{tensor: tensor}, nil
}

func (t *coreORTTensor) Close() error {
	if t == nil || t.tensor == nil {
		return nil
	}
	return t.tensor.Close()
}

func (t *coreORTTensor) Float32Data() ([]float32, error) {
	if t == nil || t.tensor == nil {
		return nil, errors.New("ORT tensor is not initialized")
	}
	return ort.TensorData[float32](t.tensor)
}

func (t *coreORTTensor) Int64Data() ([]int64, error) {
	if t == nil || t.tensor == nil {
		return nil, errors.New("ORT tensor is not initialized")
	}
	return ort.TensorData[int64](t.tensor)
}

func (s *coreORTSession) Close() error {
	if s == nil || s.session == nil {
		return nil
	}
	return s.session.Close()
}

func (s *coreORTSession) Run(ctx context.Context, inputs []*coreORTTensor) ([]*coreORTTensor, error) {
	if s == nil || s.session == nil {
		return nil, errors.New("ORT session is not initialized")
	}
	if len(inputs) != len(s.inputNames) {
		return nil, fmt.Errorf("ORT session requires %d inputs, got %d", len(s.inputNames), len(inputs))
	}
	inputMap := make(map[string]*ort.Tensor, len(inputs))
	for i, input := range inputs {
		if input == nil || input.tensor == nil {
			return nil, fmt.Errorf("ORT session input %q is not initialized", s.inputNames[i])
		}
		inputMap[s.inputNames[i]] = input.tensor
	}
	outputMap, err := s.session.Run(ctx, inputMap, s.outputNames)
	if err != nil {
		return nil, err
	}
	outputs := make([]*coreORTTensor, len(s.outputNames))
	for i, name := range s.outputNames {
		outputs[i] = &coreORTTensor{tensor: outputMap[name]}
	}
	return outputs, nil
}

func initializeCoreORT(opts *options.Options) error {
	if opts == nil || opts.ORTOptions == nil {
		return errors.New("invalid ORT options")
	}
	o := opts.ORTOptions
	if o.LibraryPath != nil {
		if _, err := os.Stat(*o.LibraryPath); err != nil {
			return fmt.Errorf("cannot find the ORT library at %s: %w", *o.LibraryPath, err)
		}
		ort.SetSharedLibraryPath(*o.LibraryPath)
	}
	if err := ort.Init(); err != nil {
		return err
	}
	if o.Telemetry != nil && *o.Telemetry {
		if err := ort.EnableTelemetry(); err != nil {
			return err
		}
	} else if err := ort.DisableTelemetry(); err != nil {
		return err
	}

	sessionOptions, err := ort.NewSessionOptions()
	if err != nil {
		return err
	}
	coreOptions := &coreORTOptions{SessionOptions: sessionOptions}
	if err := configureCoreORTOptions(coreOptions, o); err != nil {
		return errors.Join(err, coreOptions.Destroy())
	}
	opts.BackendOptions = coreOptions
	return nil
}

func configureCoreORTOptions(sessionOptions *coreORTOptions, o *options.OrtOptions) error {
	if o.IntraOpNumThreads != nil {
		if err := sessionOptions.SetIntraOpNumThreads(*o.IntraOpNumThreads); err != nil {
			return err
		}
	}
	if o.InterOpNumThreads != nil {
		if err := sessionOptions.SetInterOpNumThreads(*o.InterOpNumThreads); err != nil {
			return err
		}
	}
	if o.GraphOptimizationLevel != nil {
		if err := sessionOptions.SetGraphOptimizationLevel(ort.GraphOptimizationLevel(*o.GraphOptimizationLevel)); err != nil {
			return err
		}
	}
	if o.CPUMemArena != nil {
		if *o.CPUMemArena {
			if err := sessionOptions.EnableCpuMemArena(); err != nil {
				return err
			}
		} else if err := sessionOptions.DisableCpuMemArena(); err != nil {
			return err
		}
	}
	if o.MemPattern != nil {
		if *o.MemPattern {
			if err := sessionOptions.EnableMemPattern(); err != nil {
				return err
			}
		} else if err := sessionOptions.DisableMemPattern(); err != nil {
			return err
		}
	}
	if o.ParallelExecutionMode != nil {
		mode := ort.ExecutionModeSequential
		if *o.ParallelExecutionMode {
			mode = ort.ExecutionModeParallel
		}
		if err := sessionOptions.SetExecutionMode(mode); err != nil {
			return err
		}
	}
	if o.IntraOpSpinning != nil {
		value := "0"
		if *o.IntraOpSpinning {
			value = "1"
		}
		if err := sessionOptions.AddConfigEntry("session.intra_op.allow_spinning", value); err != nil {
			return err
		}
	}
	if o.InterOpSpinning != nil {
		value := "0"
		if *o.InterOpSpinning {
			value = "1"
		}
		if err := sessionOptions.AddConfigEntry("session.inter_op.allow_spinning", value); err != nil {
			return err
		}
	}
	providers := []struct {
		name   string
		values map[string]string
	}{
		{"CUDA", o.CudaOptions}, {"CoreML", o.CoreMLOptions}, {"OpenVINO", o.OpenVINOOptions}, {"TensorRT", o.TensorRTOptions}, {"NvTensorRTRTX", o.NvTensorRTRTXOptions},
	}
	if o.DirectMLOptions != nil {
		providers = append(providers, struct {
			name   string
			values map[string]string
		}{"DML", map[string]string{"device_id": fmt.Sprint(*o.DirectMLOptions)}})
	}
	for _, provider := range providers {
		if provider.values != nil {
			if err := sessionOptions.AppendExecutionProvider(provider.name, provider.values); err != nil {
				return err
			}
		}
	}
	for _, provider := range o.ExtraExecutionProviders {
		if err := sessionOptions.AppendExecutionProvider(provider.Name, provider.Options); err != nil {
			return err
		}
	}
	if o.OptimizedModelFilePath != nil {
		if err := sessionOptions.SetOptimizedModelFilePath(*o.OptimizedModelFilePath); err != nil {
			return err
		}
	}
	if o.ProfilingEnabled != nil {
		if *o.ProfilingEnabled {
			prefix := ""
			if o.ProfilingFilePrefix != nil {
				prefix = *o.ProfilingFilePrefix
			}
			if err := sessionOptions.EnableProfiling(prefix); err != nil {
				return err
			}
		} else if err := sessionOptions.DisableProfiling(); err != nil {
			return err
		}
	}
	return nil
}

// InitializeORT initializes the official core ORT binding and stores its
// backend-owned session options in opts.
func InitializeORT(opts *options.Options) error { return initializeCoreORT(opts) }

func IsORTInitialized() bool { return ort.IsInitialized() }

func ShutdownORT() error { return ort.Shutdown() }

func newCoreORTSession(modelPath string, modelData []byte, sessionOptions *coreORTOptions) (*coreORTSession, []InputOutputInfo, []InputOutputInfo, error) {
	if sessionOptions == nil {
		return nil, nil, nil, errors.New("invalid ORT session options")
	}
	var session *ort.Session
	var err error
	if len(modelData) > 0 {
		session, err = ort.NewSessionFromBytes(modelData, sessionOptions.SessionOptions)
	} else {
		session, err = ort.NewSession(modelPath, sessionOptions.SessionOptions)
	}
	if err != nil {
		return nil, nil, nil, err
	}
	inputs, outputs := convertCoreORTIO(session.Inputs()), convertCoreORTIO(session.Outputs())
	inputNames, outputNames := make([]string, len(inputs)), make([]string, len(outputs))
	for i := range inputs {
		inputNames[i] = inputs[i].Name
	}
	for i := range outputs {
		outputNames[i] = outputs[i].Name
	}
	return &coreORTSession{session: session, inputNames: inputNames, outputNames: outputNames}, inputs, outputs, nil
}

func convertCoreORTIO(values []ort.IOInfo) []InputOutputInfo {
	converted := make([]InputOutputInfo, len(values))
	for i, value := range values {
		converted[i] = InputOutputInfo{Name: value.Name, Dimensions: value.Shape}
	}
	return converted
}
