package pipelines

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sort"
	"sync/atomic"
	"time"

	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/util/safeconv"
)

// FillMaskPipeline predicts replacements for a single mask token in each input.
type FillMaskPipeline struct {
	*backends.BasePipeline
	TopK      int
	MaskToken string
}

type FillMaskResult struct {
	Token    string
	TokenID  int
	Sequence string
	Score    float32
}

type FillMaskOutput struct {
	Predictions [][]FillMaskResult
}

func (o *FillMaskOutput) GetOutput() []any {
	out := make([]any, len(o.Predictions))
	for i, predictions := range o.Predictions {
		out[i] = predictions
	}
	return out
}

func WithFillMaskTopK(topK int) backends.PipelineOption[*FillMaskPipeline] {
	return func(pipeline *FillMaskPipeline) error {
		pipeline.TopK = topK
		return nil
	}
}

func WithMaskToken(token string) backends.PipelineOption[*FillMaskPipeline] {
	return func(pipeline *FillMaskPipeline) error {
		pipeline.MaskToken = token
		return nil
	}
}

func NewFillMaskPipeline(ctx context.Context, config backends.PipelineConfig[*FillMaskPipeline], model *backends.Model) (*FillMaskPipeline, error) {
	pipeline := &FillMaskPipeline{
		BasePipeline: backends.NewBasePipeline(ctx, config, model),
		TopK:         5,
		MaskToken:    "<mask>",
	}
	for _, option := range config.Options {
		if err := option(pipeline); err != nil {
			return nil, err
		}
	}
	if err := pipeline.Validate(); err != nil {
		return nil, err
	}
	return pipeline, nil
}

func (p *FillMaskPipeline) IsGenerative() bool { return false }

func (p *FillMaskPipeline) GetModel() *backends.Model { return p.Model }

func (p *FillMaskPipeline) GetMetadata() backends.PipelineMetadata {
	return backends.PipelineMetadata{OutputsInfo: []backends.OutputInfo{{
		Name: p.Model.OutputsMeta[0].Name, Dimensions: p.Model.OutputsMeta[0].Dimensions,
	}}}
}

func (p *FillMaskPipeline) GetStatistics() backends.PipelineStatistics {
	statistics := backends.PipelineStatistics{}
	statistics.ComputeTokenizerStatistics(p.TokenizerTimings)
	statistics.ComputeOnnxStatistics(p.ONNXTimings)
	return statistics
}

func (p *FillMaskPipeline) Validate() error {
	var validationErrors []error
	if p.Model.Tokenizer == nil {
		validationErrors = append(validationErrors, errors.New("fill-mask pipeline requires a tokenizer"))
	}
	if p.TopK <= 0 {
		validationErrors = append(validationErrors, fmt.Errorf("top-k must be positive, got %d", p.TopK))
	}
	if len(p.Model.OutputsMeta) == 0 {
		validationErrors = append(validationErrors, errors.New("fill-mask pipeline requires model outputs"))
	}
	return errors.Join(validationErrors...)
}

func (p *FillMaskPipeline) preprocess(batch *backends.PipelineBatch, inputs []string) error {
	start := time.Now()
	backends.TokenizeInputs(batch, p.Model.Tokenizer, inputs)
	atomic.AddUint64(&p.TokenizerTimings.NumCalls, 1)
	atomic.AddUint64(&p.TokenizerTimings.TotalNS, safeconv.DurationToU64(time.Since(start)))
	for i, input := range batch.Input {
		count := 0
		for _, token := range input.Tokens {
			if token == p.MaskToken {
				count++
			}
		}
		if count != 1 {
			return fmt.Errorf("input %d must contain exactly one mask token %q, found %d", i, p.MaskToken, count)
		}
	}
	return backends.CreateInputTensors(batch, p.Model)
}

func (p *FillMaskPipeline) forward(ctx context.Context, batch *backends.PipelineBatch) error {
	start := time.Now()
	if err := backends.RunSessionOnBatch(ctx, batch, p.BasePipeline); err != nil {
		return err
	}
	atomic.AddUint64(&p.ONNXTimings.NumCalls, 1)
	atomic.AddUint64(&p.ONNXTimings.TotalNS, safeconv.DurationToU64(time.Since(start)))
	return nil
}

func (p *FillMaskPipeline) postprocess(batch *backends.PipelineBatch) (*FillMaskOutput, error) {
	if len(batch.OutputValues) == 0 {
		return nil, errors.New("fill-mask model returned no outputs")
	}
	logits, ok := batch.OutputValues[0].([][][]float32)
	if !ok {
		return nil, fmt.Errorf("fill-mask output type %T is not supported; expected 3D logits", batch.OutputValues[0])
	}
	if len(logits) != len(batch.Input) {
		return nil, fmt.Errorf("fill-mask output batch size %d does not match input batch size %d", len(logits), len(batch.Input))
	}
	predictions := make([][]FillMaskResult, len(logits))
	for batchIndex, inputLogits := range logits {
		maskIndex := -1
		for index, token := range batch.Input[batchIndex].Tokens {
			if token == p.MaskToken {
				maskIndex = index
				break
			}
		}
		if maskIndex < 0 || maskIndex >= len(inputLogits) {
			return nil, fmt.Errorf("mask position for input %d is not present in model output", batchIndex)
		}
		row := inputLogits[maskIndex]
		indices := make([]int, len(row))
		for i := range row {
			indices[i] = i
		}
		sort.Slice(indices, func(i, j int) bool { return row[indices[i]] > row[indices[j]] })
		if p.TopK < len(indices) {
			indices = indices[:p.TopK]
		}
		maxLogit := float32(-math.MaxFloat32)
		for _, value := range row {
			if value > maxLogit {
				maxLogit = value
			}
		}
		var denominator float32
		for _, value := range row {
			denominator += float32(math.Exp(float64(value - maxLogit)))
		}
		for _, tokenID := range indices {
			if tokenID < 0 || uint64(tokenID) > uint64(^uint32(0)) {
				return nil, fmt.Errorf("fill-mask token ID %d cannot be represented as uint32", tokenID)
			}
			ids := append([]uint32(nil), batch.Input[batchIndex].TokenIDs...)
			ids[maskIndex] = uint32(tokenID)
			sequence, err := backends.Decode(ids, p.Model.Tokenizer)
			if err != nil {
				return nil, err
			}
			predictions[batchIndex] = append(predictions[batchIndex], FillMaskResult{
				Token: batch.Input[batchIndex].Tokens[maskIndex], TokenID: tokenID,
				Sequence: sequence, Score: float32(math.Exp(float64(row[tokenID]-maxLogit))) / denominator,
			})
		}
	}
	return &FillMaskOutput{Predictions: predictions}, nil
}

func (p *FillMaskPipeline) Run(ctx context.Context, inputs []string) (backends.PipelineBatchOutput, error) {
	return p.RunPipeline(ctx, inputs)
}

func (p *FillMaskPipeline) RunPipeline(ctx context.Context, inputs []string) (*FillMaskOutput, error) {
	return backends.RunPipeline(ctx, len(inputs), func(batch *backends.PipelineBatch) error {
		return p.preprocess(batch, inputs)
	}, p.forward, p.postprocess)
}
