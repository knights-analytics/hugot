package pipelines

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"sync/atomic"
	"time"

	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/util/fileutil"
	"github.com/knights-analytics/hugot/util/safeconv"
)

func readNativeConfig(ctx context.Context, path, name string, value any) error {
	data, err := fileutil.ReadFileBytes(ctx, fileutil.PathJoinSafe(path, name))
	if err != nil {
		return err
	}
	return json.Unmarshal(data, value)
}

func validatePipelineContext(ctx context.Context, p *backends.BasePipeline) error {
	if ctx == nil {
		return errors.New("nil pipeline context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if p.SessionContext != nil {
		return p.SessionContext.Err()
	}
	return nil
}

func lastTokenArgmax(tensor backends.Tensor) (int64, error) {
	data, ok := tensor.Data.([]float32)
	if !ok || len(tensor.Shape) != 3 || tensor.Shape[0] != 1 || tensor.Shape[1] <= 0 || tensor.Shape[2] <= 0 || int64(len(data)) != tensor.Shape[1]*tensor.Shape[2] {
		return 0, errors.New("decoder logits must have shape [1,sequence,vocabulary]")
	}
	vocab := int(tensor.Shape[2])
	row := data[len(data)-vocab:]
	best := 0
	for i, value := range row {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return 0, errors.New("decoder logits are nonfinite")
		}
		if value > row[best] {
			best = i
		}
	}
	return int64(best), nil
}

func requireTensorNames(model *backends.Model, inputs, outputs []string) error {
	if model == nil || model.IsGenerative {
		return errors.New("task requires a non-generative ONNX model")
	}
	for _, group := range []struct {
		names []string
		meta  []backends.InputOutputInfo
	}{{inputs, model.InputsMeta}, {outputs, model.OutputsMeta}} {
		for _, name := range group.names {
			if !tensorPresent(group.meta, name) {
				return fmt.Errorf("task model is missing tensor %q", name)
			}
		}
	}
	return nil
}

func nativeMetadata(model *backends.Model) backends.PipelineMetadata {
	if model == nil {
		return backends.PipelineMetadata{}
	}
	out := make([]backends.OutputInfo, len(model.OutputsMeta))
	for i, info := range model.OutputsMeta {
		out[i] = backends.OutputInfo{Name: info.Name, Dimensions: info.Dimensions}
	}
	return backends.PipelineMetadata{OutputsInfo: out}
}

func nativeStatistics(p *backends.BasePipeline) backends.PipelineStatistics {
	out := backends.PipelineStatistics{}
	out.ComputeOnnxStatistics(p.ONNXTimings)
	out.ComputeTokenizerStatistics(p.TokenizerTimings)
	return out
}

func runNativeTensors(ctx context.Context, p *backends.BasePipeline, model *backends.Model, inputs map[string]backends.Tensor) (map[string]backends.Tensor, error) {
	if ctx == nil {
		return nil, errors.New("nil inference context")
	}
	if p.SessionContext != nil {
		if err := p.SessionContext.Err(); err != nil {
			return nil, err
		}
		merged, cancel := context.WithCancel(ctx)
		stop := context.AfterFunc(p.SessionContext, cancel)
		defer func() { stop(); cancel() }()
		ctx = merged
	}
	start := time.Now()
	out, err := model.RunTensors(ctx, inputs)
	atomic.AddUint64(&p.ONNXTimings.NumCalls, 1)
	atomic.AddUint64(&p.ONNXTimings.TotalNS, safeconv.DurationToU64(time.Since(start)))
	return out, err
}

func tensorPresent(meta []backends.InputOutputInfo, name string) bool {
	for _, info := range meta {
		if info.Name == name {
			return true
		}
	}
	return false
}
