package backends

import (
	"context"
	"errors"
	"fmt"
	"math"
	"path/filepath"
	"strings"

	"github.com/knights-analytics/hugot/util/fileutil"
)

// Tensor holds a runtime shape and flat []float32, []int64, or []bool data.
// An empty shape denotes a scalar with one element.
type Tensor struct {
	Shape []int64
	Data  any
}

// RunTensors executes named inputs without pipeline padding or reshaping.
// Returned shapes and data are copies owned by the caller. Inputs must not be
// modified until the call returns. Close waits for active calls to finish.
func (model *Model) RunTensors(ctx context.Context, inputs map[string]Tensor) (map[string]Tensor, error) {
	if ctx == nil {
		return nil, errors.New("nil inference context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if model == nil {
		return nil, errors.New("model is nil")
	}
	model.closeMu.Lock()
	defer model.closeMu.Unlock()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if model.closed {
		return nil, errors.New("model is closed")
	}
	if model.IsGenerative {
		return nil, errors.New("named tensor inference requires a non-generative graph")
	}
	if err := validateNamedTensors(model.InputsMeta, inputs); err != nil {
		return nil, err
	}
	if model.ORTModel != nil {
		return runNamedTensorsORT(ctx, model, inputs)
	}
	if model.GoMLXModel != nil {
		return runNamedTensorsGoMLX(ctx, model, inputs)
	}
	return nil, errors.New("model runtime is not initialized")
}

func validateNamedTensors(metadata []InputOutputInfo, inputs map[string]Tensor) error {
	names := make(map[string]bool, len(metadata))
	for _, meta := range metadata {
		names[meta.Name] = true
		input, ok := inputs[meta.Name]
		if !ok {
			return fmt.Errorf("missing input tensor %q", meta.Name)
		}
		if len(input.Shape) != len(meta.Dimensions) {
			return fmt.Errorf("input %q has rank %d, expected %d", meta.Name, len(input.Shape), len(meta.Dimensions))
		}
		count := int64(1)
		for axis, dim := range input.Shape {
			if dim < 0 || dim > int64(math.MaxInt) {
				return fmt.Errorf("input %q has invalid dimension %d at axis %d", meta.Name, dim, axis)
			}
			if expected := meta.Dimensions[axis]; expected >= 0 && dim != expected {
				return fmt.Errorf("input %q dimension %d is %d, expected %d", meta.Name, axis, dim, expected)
			}
			if dim != 0 && count > int64(math.MaxInt)/dim {
				return fmt.Errorf("input %q shape element count overflows", meta.Name)
			}
			count *= dim
		}
		var length int
		switch data := input.Data.(type) {
		case []float32:
			length = len(data)
		case []int64:
			length = len(data)
		case []bool:
			length = len(data)
		default:
			return fmt.Errorf("input %q has unsupported data type %T; expected []float32, []int64, or []bool", meta.Name, input.Data)
		}
		if int64(length) != count {
			return fmt.Errorf("input %q has %d elements, expected %d for shape %v", meta.Name, length, count, input.Shape)
		}
	}
	for name := range inputs {
		if !names[name] {
			return fmt.Errorf("unknown input tensor %q", name)
		}
	}
	return nil
}

// LoadGraph loads and caches an additional ONNX graph relative to the model
// directory, using the parent's backend settings and no tokenizer. The parent
// owns the returned model and closes it when the parent is closed.
func (model *Model) LoadGraph(ctx context.Context, filename string) (*Model, error) {
	if ctx == nil {
		return nil, errors.New("nil graph loading context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if model == nil {
		return nil, errors.New("model is nil")
	}
	filename = filepath.Clean(strings.ReplaceAll(filename, "\\", "/"))
	if filename == "." || filepath.IsAbs(filename) || filepath.VolumeName(filename) != "" || filename == ".." || strings.HasPrefix(filename, ".."+string(filepath.Separator)) || !strings.HasSuffix(filename, ".onnx") {
		return nil, fmt.Errorf("invalid graph filename %q: expected a relative .onnx path", filename)
	}
	model.closeMu.Lock()
	defer model.closeMu.Unlock()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if model.closed {
		return nil, errors.New("model is closed")
	}
	if graph := model.graphs[filename]; graph != nil {
		graph.closeMu.Lock()
		closed := graph.closed
		graph.closeMu.Unlock()
		if !closed {
			return graph, nil
		}
	}
	if model.loadOptions == nil {
		return nil, errors.New("model loading options are not configured")
	}
	graph := &Model{
		ID:           model.Path + ":" + filename,
		Path:         model.Path,
		OnnxFilename: filename,
		Pipelines:    map[string]Pipeline{},
		loadOptions:  model.loadOptions,
	}
	ctx = fileutil.WithFileSystem(ctx, model.loadOptions.FileSystem)
	if err := CreateModelBackend(ctx, graph, model.loadOptions); err != nil {
		return nil, errors.Join(err, graph.Close())
	}
	if err := ctx.Err(); err != nil {
		return nil, errors.Join(err, graph.Close())
	}
	if model.graphs == nil {
		model.graphs = make(map[string]*Model)
	}
	model.graphs[filename] = graph
	return graph, nil
}
