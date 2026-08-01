package pipelines

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync/atomic"
	"time"

	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/util/safeconv"
)

// NativeInt8FeatureExtractionPipeline returns already-pooled native signed INT8 embeddings
// from exactly one construction-time selected ORT output.
//
// Unlike FeatureExtractionPipeline, this pipeline never mean-pools, normalizes, widens, or
// otherwise transforms coordinates. It requires OnnxOutputNames to select exactly one output
// before ORT session creation and expects that output to arrive as Go-owned [][]int8 with
// shape [batch, dimension].
type NativeInt8FeatureExtractionPipeline struct {
	*backends.BasePipeline
	Output backends.InputOutputInfo
}

// Int8FeatureExtractionOutput holds one native signed INT8 embedding vector per input text.
type Int8FeatureExtractionOutput struct {
	Embeddings [][]int8
}

// GetOutput exposes each INT8 embedding through the PipelineBatchOutput interface without
// converting or widening the coordinates.
func (t *Int8FeatureExtractionOutput) GetOutput() []any {
	out := make([]any, len(t.Embeddings))
	for i, embedding := range t.Embeddings {
		out[i] = any(embedding)
	}
	return out
}

// NewNativeInt8FeatureExtractionPipeline constructs an ORT-only native INT8 feature-extraction
// pipeline that consumes exactly one selected rank-two INT8 ONNX output.
//
// Construction fails when the backend is not ORT, when OnnxOutputNames does not contain
// exactly one name, or when the loaded model does not expose exactly one selected output.
// Inference copies signed coordinates into Go-owned storage and never pools or normalizes.
//
// sessionContext: session lifetime context used for ORT runs.
// config: pipeline configuration; OnnxOutputNames must contain exactly one output name.
// s: Hugot options whose Backend must be ORT.
// model: model loaded with the matching single-output ORT session contract.
//
// Returns the validated pipeline or an actionable construction error.
func NewNativeInt8FeatureExtractionPipeline(sessionContext context.Context, config backends.PipelineConfig[*NativeInt8FeatureExtractionPipeline], s *options.Options, model *backends.Model) (*NativeInt8FeatureExtractionPipeline, error) {
	if s == nil || s.Backend != "ORT" {
		return nil, fmt.Errorf("NativeInt8FeatureExtractionPipeline requires the ORT backend")
	}
	if len(config.OnnxOutputNames) != 1 {
		return nil, fmt.Errorf(
			"NativeInt8FeatureExtractionPipeline requires exactly one OnnxOutputNames entry; got %d (%s)",
			len(config.OnnxOutputNames),
			strings.Join(config.OnnxOutputNames, ", "),
		)
	}
	if len(model.OnnxOutputNames) != 1 || len(model.OutputsMeta) != 1 {
		return nil, fmt.Errorf(
			"NativeInt8FeatureExtractionPipeline requires a model session with exactly one selected output; model has OnnxOutputNames=%s OutputsMeta=%s",
			strings.Join(model.OnnxOutputNames, ", "),
			strings.Join(backends.GetNames(model.OutputsMeta), ", "),
		)
	}
	if model.OnnxOutputNames[0] != config.OnnxOutputNames[0] || model.OutputsMeta[0].Name != config.OnnxOutputNames[0] {
		return nil, fmt.Errorf(
			"selected output mismatch: config=%q model.OnnxOutputNames=%q model.OutputsMeta=%q",
			config.OnnxOutputNames[0],
			model.OnnxOutputNames[0],
			model.OutputsMeta[0].Name,
		)
	}

	defaultPipeline, err := backends.NewBasePipeline(sessionContext, config, s, model)
	if err != nil {
		return nil, err
	}
	pipeline := &NativeInt8FeatureExtractionPipeline{
		BasePipeline: defaultPipeline,
		Output:       model.OutputsMeta[0],
	}
	for _, o := range config.Options {
		err = o(pipeline)
		if err != nil {
			return nil, err
		}
	}
	if err = pipeline.Validate(); err != nil {
		return nil, err
	}
	return pipeline, nil
}

// IsGenerative reports that native INT8 feature extraction is non-generative.
func (p *NativeInt8FeatureExtractionPipeline) IsGenerative() bool {
	return false
}

// GetModel returns the underlying Hugot model.
func (p *NativeInt8FeatureExtractionPipeline) GetModel() *backends.Model {
	return p.Model
}

// GetMetadata returns the single selected output name and static ONNX dimensions.
func (p *NativeInt8FeatureExtractionPipeline) GetMetadata() backends.PipelineMetadata {
	return backends.PipelineMetadata{
		OutputsInfo: []backends.OutputInfo{
			{
				Name:       p.Output.Name,
				Dimensions: p.Output.Dimensions,
			},
		},
	}
}

// GetStatistics returns tokenizer and ORT runtime statistics for the pipeline.
func (p *NativeInt8FeatureExtractionPipeline) GetStatistics() backends.PipelineStatistics {
	statistics := backends.PipelineStatistics{}
	if p.TokenizerTimings != nil {
		statistics.ComputeTokenizerStatistics(p.TokenizerTimings)
	}
	statistics.ComputeOnnxStatistics(p.ONNXTimings)
	return statistics
}

// Validate checks that the pipeline can tokenize text and that input ranks are supported.
func (p *NativeInt8FeatureExtractionPipeline) Validate() error {
	var validationErrors []error
	if p.Model.Tokenizer == nil {
		validationErrors = append(validationErrors, fmt.Errorf("native INT8 feature extraction pipeline requires a tokenizer"))
	}
	if p.Runtime != "ORT" {
		validationErrors = append(validationErrors, fmt.Errorf("native INT8 feature extraction pipeline requires the ORT backend"))
	}
	for _, input := range p.Model.InputsMeta {
		dims := []int64(input.Dimensions)
		if len(dims) > 4 {
			validationErrors = append(validationErrors, fmt.Errorf("inputs currently can have at most 4 dimensions"))
		}
	}
	return errors.Join(validationErrors...)
}

func (p *NativeInt8FeatureExtractionPipeline) preprocess(batch *backends.PipelineBatch, inputs []string) error {
	start := time.Now()
	backends.TokenizeInputs(batch, p.Model.Tokenizer, inputs)
	atomic.AddUint64(&p.TokenizerTimings.NumCalls, 1)
	atomic.AddUint64(&p.TokenizerTimings.TotalNS, safeconv.DurationToU64(time.Since(start)))
	return backends.CreateInputTensors(batch, p.Model, p.Runtime)
}

func (p *NativeInt8FeatureExtractionPipeline) forward(ctx context.Context, batch *backends.PipelineBatch) error {
	start := time.Now()
	err := backends.RunSessionOnBatch(ctx, batch, p.BasePipeline)
	if err != nil {
		return err
	}
	atomic.AddUint64(&p.ONNXTimings.NumCalls, 1)
	atomic.AddUint64(&p.ONNXTimings.TotalNS, safeconv.DurationToU64(time.Since(start)))
	return nil
}

func (p *NativeInt8FeatureExtractionPipeline) postprocess(batch *backends.PipelineBatch) (*Int8FeatureExtractionOutput, error) {
	if len(batch.OutputValues) != 1 {
		return nil, fmt.Errorf(
			"native INT8 feature extraction expected exactly one ORT output for %q; got %d",
			p.Output.Name,
			len(batch.OutputValues),
		)
	}
	output := batch.OutputValues[0]
	embeddings, ok := output.([][]int8)
	if !ok {
		return nil, fmt.Errorf(
			"ORT output %q has type %T; expected Go-owned [][]int8 from a rank-two INT8 tensor",
			p.Output.Name,
			output,
		)
	}
	if len(embeddings) != batch.Size {
		return nil, fmt.Errorf(
			"ORT output %q returned %d vectors; expected batch size %d",
			p.Output.Name,
			len(embeddings),
			batch.Size,
		)
	}
	return &Int8FeatureExtractionOutput{Embeddings: embeddings}, nil
}

// Run runs native INT8 feature extraction on a batch of strings.
func (p *NativeInt8FeatureExtractionPipeline) Run(ctx context.Context, inputs []string) (backends.PipelineBatchOutput, error) {
	return p.RunPipeline(ctx, inputs)
}

// RunPipeline runs native INT8 feature extraction and returns the concrete INT8 result type.
//
// Each input produces one independently owned []int8 vector in input order. Coordinates are
// returned exactly as produced by the selected ORT INT8 output with no pooling, normalization,
// widening, clamping, or other transformation.
func (p *NativeInt8FeatureExtractionPipeline) RunPipeline(ctx context.Context, inputs []string) (*Int8FeatureExtractionOutput, error) {
	var runErrors []error
	batch := backends.NewBatch(len(inputs))
	defer func(*backends.PipelineBatch) {
		runErrors = append(runErrors, batch.Destroy())
	}(batch)
	runErrors = append(runErrors, p.preprocess(batch, inputs))
	if e := errors.Join(runErrors...); e != nil {
		return nil, e
	}
	runErrors = append(runErrors, p.forward(ctx, batch))
	if e := errors.Join(runErrors...); e != nil {
		return nil, e
	}
	result, postErr := p.postprocess(batch)
	runErrors = append(runErrors, postErr)
	return result, errors.Join(runErrors...)
}
