package pipelines

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"sync/atomic"
	"time"

	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/util/safeconv"
	"github.com/knights-analytics/hugot/util/vectorutil"
)

type CrossEncoderPipeline struct {
	*backends.BasePipeline
	scoreThreshold *float32
	statistics     CrossEncoderStatistics
	batchSize      int
	sortResults    bool
}
type CrossEncoderStatistics struct {
	TotalQueries     uint64
	TotalDocuments   uint64
	AverageLatency   time.Duration
	AverageBatchSize float64
	FilteredResults  uint64
}
type CrossEncoderResult struct {
	Document string
	Score    float32
	Index    int
}
type CrossEncoderOutput struct {
	Results []CrossEncoderResult
}

func WithBatchSize(size int) backends.PipelineOption[*CrossEncoderPipeline] {
	return func(p *CrossEncoderPipeline) error {
		p.batchSize = size
		return nil
	}
}

func WithSortResults() backends.PipelineOption[*CrossEncoderPipeline] {
	return func(p *CrossEncoderPipeline) error {
		p.sortResults = true
		return nil
	}
}

func WithScoreThreshold(threshold float32) backends.PipelineOption[*CrossEncoderPipeline] {
	return func(p *CrossEncoderPipeline) error {
		p.scoreThreshold = &threshold
		return nil
	}
}

func (t *CrossEncoderOutput) GetOutput() []any {
	out := make([]any, len(t.Results))
	for i, result := range t.Results {
		out[i] = any(result)
	}
	return out
}

func NewCrossEncoderPipeline(sessionContext context.Context, config backends.PipelineConfig[*CrossEncoderPipeline], s *options.Options, model *backends.Model) (*CrossEncoderPipeline, error) {
	defaultPipeline, err := backends.NewBasePipeline(sessionContext, config, s, model)
	if err != nil {
		return nil, err
	}
	pipeline := &CrossEncoderPipeline{
		BasePipeline: defaultPipeline,
		batchSize:    1,
		sortResults:  true,
	}
	for _, o := range config.Options {
		err = o(pipeline)
		if err != nil {
			return nil, err
		}
	}
	err = pipeline.Validate()
	if err != nil {
		return nil, err
	}
	return pipeline, nil
}

// INTERFACE IMPLEMENTATIONS.
func (p *CrossEncoderPipeline) IsGenerative() bool {
	return false
}

func (p *CrossEncoderPipeline) GetModel() *backends.Model {
	return p.Model
}

func (p *CrossEncoderPipeline) GetMetadata() backends.PipelineMetadata {
	return backends.PipelineMetadata{
		OutputsInfo: []backends.OutputInfo{
			{
				Name:       p.Model.OutputsMeta[0].Name,
				Dimensions: p.Model.OutputsMeta[0].Dimensions,
			},
		},
	}
}

func (p *CrossEncoderPipeline) GetStatistics() backends.PipelineStatistics {
	avgLatency := p.statistics.AverageLatency
	if p.statistics.TotalQueries > 0 {
		avgLatency = time.Duration(float64(p.statistics.AverageLatency) / float64(p.statistics.TotalQueries))
	}
	statistics := backends.PipelineStatistics{}
	statistics.ComputeTokenizerStatistics(p.Model.Tokenizer.TokenizerTimings)
	statistics.ComputeOnnxStatistics(p.PipelineTimings)
	statistics.TotalQueries = p.statistics.TotalQueries
	statistics.TotalDocuments = p.statistics.TotalDocuments
	statistics.AverageLatency = avgLatency
	statistics.AverageBatchSize = p.statistics.AverageBatchSize
	statistics.FilteredResults = p.statistics.FilteredResults
	return statistics
}

func (p *CrossEncoderPipeline) Validate() error {
	var validationErrors []error
	if p.Model.Tokenizer == nil {
		validationErrors = append(validationErrors, fmt.Errorf("cross encoder pipeline requires a tokenizer"))
	}
	if p.Model.SeparatorToken == "" {
		validationErrors = append(validationErrors, fmt.Errorf("cross encoder pipeline requires a separator token to be set in the model"))
	}
	if p.Model.SeparatorToken != "[SEP]" && p.Model.SeparatorToken != "</s>" {
		validationErrors = append(validationErrors, fmt.Errorf("cross encoder pipeline only supports [SEP] (BERT) and </s> (Roberta) as separator tokens, got %s", p.Model.SeparatorToken))
	}
	outDims := p.Model.OutputsMeta[0].Dimensions
	if len(outDims) != 2 {
		validationErrors = append(validationErrors, fmt.Errorf("pipeline configuration invalid: cross encoder must have 2 dimensional output"))
	} else if outDims[1] != 1 {
		validationErrors = append(validationErrors, fmt.Errorf("pipeline configuration invalid: cross encoder output's second dimension must be 1, but got %d", outDims[1]))
	}
	dynamicBatch := false
	for _, d := range outDims {
		if d == -1 {
			if dynamicBatch {
				validationErrors = append(validationErrors, fmt.Errorf("pipeline configuration invalid: cross encoder must have max one dynamic dimensions (input)"))
				break
			}
			dynamicBatch = true
		}
	}
	if p.batchSize <= 0 {
		validationErrors = append(validationErrors, fmt.Errorf("batch size must be positive, got %d", p.batchSize))
	}
	return errors.Join(validationErrors...)
}

func (p *CrossEncoderPipeline) preprocess(batch *backends.PipelineBatch, inputs []string) error {
	start := time.Now()
	backends.TokenizeInputs(batch, p.Model.Tokenizer, inputs)
	atomic.AddUint64(&p.Model.Tokenizer.TokenizerTimings.NumCalls, 1)
	atomic.AddUint64(&p.Model.Tokenizer.TokenizerTimings.TotalNS, safeconv.DurationToU64(time.Since(start)))
	err := backends.CreateInputTensors(batch, p.Model, p.Runtime)
	return err
}

func (p *CrossEncoderPipeline) preprocessPairs(batch *backends.PipelineBatch, inputs [][2]string) error {
	start := time.Now()
	backends.TokenizeInputPairs(batch, p.Model.Tokenizer, inputs, p.Model.SeparatorToken)
	atomic.AddUint64(&p.Model.Tokenizer.TokenizerTimings.NumCalls, 1)
	atomic.AddUint64(&p.Model.Tokenizer.TokenizerTimings.TotalNS, safeconv.DurationToU64(time.Since(start)))
	err := backends.CreateInputTensors(batch, p.Model, p.Runtime)
	return err
}

func (p *CrossEncoderPipeline) forward(ctx context.Context, batch *backends.PipelineBatch) error {
	start := time.Now()
	if p.Model.Tokenizer.MaxAllowedTokens >= p.Model.MaxPositionEmbeddings {
		p.Model.Tokenizer.MaxAllowedTokens = p.Model.MaxPositionEmbeddings - 1
	}
	err := backends.RunSessionOnBatch(ctx, batch, p.BasePipeline)
	if err != nil {
		return err
	}
	atomic.AddUint64(&p.PipelineTimings.NumCalls, 1)
	atomic.AddUint64(&p.PipelineTimings.TotalNS, safeconv.DurationToU64(time.Since(start)))
	return nil
}

func (p *CrossEncoderPipeline) postprocess(batch *backends.PipelineBatch, documents []string) (*CrossEncoderOutput, error) {
	var outputCast [][]float32
	output := batch.OutputValues[0]
	switch v := output.(type) {
	case [][]float32:
		outputCast = make([][]float32, len(v))
		for i, logits := range v {
			scores := vectorutil.Sigmoid(logits)
			outputCast[i] = scores
		}
	default:
		return nil, fmt.Errorf("output is not 2D, expected batch size x logits, got %T", output)
	}
	results := make([]CrossEncoderResult, 0, len(documents))
	for i := range documents {
		score := outputCast[i][0]
		if p.scoreThreshold != nil && score < *p.scoreThreshold {
			atomic.AddUint64(&p.statistics.FilteredResults, 1)
			continue
		}
		result := CrossEncoderResult{
			Document: documents[i],
			Score:    score,
			Index:    i,
		}
		results = append(results, result)
	}
	return &CrossEncoderOutput{Results: results}, nil
}

func (p *CrossEncoderPipeline) Run(ctx context.Context, inputs []string) (backends.PipelineBatchOutput, error) {
	if len(inputs) < 2 {
		return nil, errors.New("cross encoder pipeline requires at least two inputs: a query and one or more documents")
	}
	query := inputs[0]
	documents := inputs[1:]
	return p.RunPipeline(ctx, query, documents)
}

func (p *CrossEncoderPipeline) RunPipeline(ctx context.Context, query string, documents []string) (*CrossEncoderOutput, error) {
	start := time.Now()
	defer func() {
		duration := time.Since(start)
		p.statistics.TotalQueries++
		p.statistics.TotalDocuments += uint64(len(documents))
		p.statistics.AverageLatency += duration
	}()
	var runErrors []error
	allResults := make([]CrossEncoderResult, 0, len(documents))
	for i := 0; i < len(documents); i += p.batchSize {
		end := min(i+p.batchSize, len(documents))
		out, err := p.runBatch(ctx, query, documents[i:end], i)
		if err != nil {
			runErrors = append(runErrors, err)
			continue
		}
		allResults = append(allResults, out.Results...)
	}
	if p.sortResults {
		sort.SliceStable(allResults, func(i, j int) bool {
			if allResults[i].Score == allResults[j].Score {
				return allResults[i].Index < allResults[j].Index
			}
			return allResults[i].Score > allResults[j].Score
		})
	}
	output := &CrossEncoderOutput{
		Results: allResults,
	}
	return output, errors.Join(runErrors...)
}

func (p *CrossEncoderPipeline) runBatch(ctx context.Context, query string, documents []string, startIndex int) (*CrossEncoderOutput, error) {
	var runErrors []error
	inputs := make([][2]string, len(documents))
	for i, doc := range documents {
		inputs[i] = [2]string{query, doc}
	}
	batch := backends.NewBatch(len(inputs))
	defer func(*backends.PipelineBatch) {
		runErrors = append(runErrors, batch.Destroy())
	}(batch)
	runErrors = append(runErrors, p.preprocessPairs(batch, inputs))
	if e := errors.Join(runErrors...); e != nil {
		return nil, e
	}
	runErrors = append(runErrors, p.forward(ctx, batch))
	if e := errors.Join(runErrors...); e != nil {
		return nil, e
	}
	result, postErr := p.postprocess(batch, documents)
	runErrors = append(runErrors, postErr)
	if result != nil {
		for i := range result.Results {
			result.Results[i].Index += startIndex
		}
	}
	return result, errors.Join(runErrors...)
}
