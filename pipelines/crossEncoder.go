package pipelines

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"sync/atomic"
	"time"

	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/pipelineBackends"
	"github.com/knights-analytics/hugot/util"
)

var _ pipelineBackends.Pipeline = (*CrossEncoderPipeline)(nil)

type CrossEncoderPipeline struct {
	*pipelineBackends.BasePipeline
	batchSize         int
	maxConcurrentReqs int
	sortResults       bool
	scoreThreshold    *float32

	// Performance monitoring
	batchProcessor *BatchProcessor
	stats          CrossEncoderStats
}

type CrossEncoderStats struct {
	TotalQueries       uint64
	TotalDocuments     uint64
	AverageLatency     time.Duration
	AverageBatchSize   float64
	TruncatedSequences uint64
	FilteredResults    uint64
}

type CrossEncoderResult struct {
	Document string
	Score    float32
	Index    int // Original position for stable sorting
}

type CrossEncoderOutput struct {
	Results   []CrossEncoderResult
	QueryTime time.Duration
	Stats     CrossEncoderStats
}

type BatchProcessor struct {
	maxBatchSize int
	timeout      time.Duration
	requests     chan *BatchRequest
}

type BatchRequest struct {
	query     string
	documents []string
	response  chan *BatchResponse
}

type BatchResponse struct {
	output *CrossEncoderOutput
	err    error
}

func WithBatchSize(size int) pipelineBackends.PipelineOption[*CrossEncoderPipeline] {
	return func(p *CrossEncoderPipeline) {
		p.batchSize = size
	}
}

func WithSortResults(sort bool) pipelineBackends.PipelineOption[*CrossEncoderPipeline] {
	return func(p *CrossEncoderPipeline) {
		p.sortResults = sort
	}
}

func WithScoreThreshold(threshold float32) pipelineBackends.PipelineOption[*CrossEncoderPipeline] {
	return func(p *CrossEncoderPipeline) {
		p.scoreThreshold = &threshold
	}
}

func (t *CrossEncoderOutput) GetOutput() []any {
	out := make([]any, len(t.Results))
	for i, result := range t.Results {
		out[i] = any(result)
	}
	return out
}

func NewCrossEncoderPipeline(config pipelineBackends.PipelineConfig[*CrossEncoderPipeline], s *options.Options, model *pipelineBackends.Model) (*CrossEncoderPipeline, error) {
	defaultPipeline, err := pipelineBackends.NewBasePipeline(config, s, model)
	if err != nil {
		return nil, err
	}

	pipeline := &CrossEncoderPipeline{
		BasePipeline: defaultPipeline,
		batchSize:    1, // Not sure why but I always get the best performance with 1
		sortResults:  true,
	}

	for _, o := range config.Options {
		o(pipeline)
	}

	err = pipeline.Validate()
	if err != nil {
		return nil, err
	}
	return pipeline, nil
}

func (p *CrossEncoderPipeline) GetModel() *pipelineBackends.Model {
	return p.Model
}

func (p *CrossEncoderPipeline) GetMetadata() pipelineBackends.PipelineMetadata {
	return pipelineBackends.PipelineMetadata{
		OutputsInfo: []pipelineBackends.OutputInfo{
			{
				Name:       p.Model.OutputsMeta[0].Name,
				Dimensions: p.Model.OutputsMeta[0].Dimensions,
			},
		},
	}
}

func (p *CrossEncoderPipeline) GetStats() []string {
	avgLatency := p.stats.AverageLatency
	if p.stats.TotalQueries > 0 {
		avgLatency = time.Duration(float64(p.stats.AverageLatency) / float64(p.stats.TotalQueries))
	}

	return []string{
		fmt.Sprintf("Statistics for pipeline: %s", p.PipelineName),
		fmt.Sprintf("Total queries processed: %d", p.stats.TotalQueries),
		fmt.Sprintf("Total documents scored: %d", p.stats.TotalDocuments),
		fmt.Sprintf("Average latency per query: %s", avgLatency),
		fmt.Sprintf("Average batch size: %.2f", p.stats.AverageBatchSize),
		fmt.Sprintf("Truncated sequences: %d", p.stats.TruncatedSequences),
		fmt.Sprintf("Filtered results: %d", p.stats.FilteredResults),
		fmt.Sprintf("Tokenizer: Total time=%s, Execution count=%d, Average query time=%s",
			time.Duration(p.Model.Tokenizer.TokenizerTimings.TotalNS),
			p.Model.Tokenizer.TokenizerTimings.NumCalls,
			time.Duration(float64(p.Model.Tokenizer.TokenizerTimings.TotalNS)/math.Max(1, float64(p.Model.Tokenizer.TokenizerTimings.NumCalls)))),
		fmt.Sprintf("ONNX: Total time=%s, Execution count=%d, Average query time=%s",
			time.Duration(p.PipelineTimings.TotalNS),
			p.PipelineTimings.NumCalls,
			time.Duration(float64(p.PipelineTimings.TotalNS)/math.Max(1, float64(p.PipelineTimings.NumCalls)))),
	}
}

func (p *CrossEncoderPipeline) Validate() error {
	var validationErrors []error

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

func (p *CrossEncoderPipeline) Preprocess(batch *pipelineBackends.PipelineBatch, inputs []string) error {
	start := time.Now()

	pipelineBackends.TokenizeInputs(batch, p.Model.Tokenizer, inputs)

	// Track truncated sequences (tokenizer already handles truncation)
	for _, tokenizedInput := range batch.Input {
		if len(tokenizedInput.TokenIDs) >= p.Model.Tokenizer.MaxAllowedTokens {
			atomic.AddUint64(&p.stats.TruncatedSequences, 1)
		}
	}

	atomic.AddUint64(&p.Model.Tokenizer.TokenizerTimings.NumCalls, 1)
	atomic.AddUint64(&p.Model.Tokenizer.TokenizerTimings.TotalNS, uint64(time.Since(start)))
	err := pipelineBackends.CreateInputTensors(batch, p.Model.InputsMeta, p.Runtime)
	return err
}

func (p *CrossEncoderPipeline) Forward(batch *pipelineBackends.PipelineBatch) error {
	start := time.Now()

	if p.Model.Tokenizer.MaxAllowedTokens >= p.Model.MaxPositionEmbeddings {
		p.Model.Tokenizer.MaxAllowedTokens = p.Model.MaxPositionEmbeddings - 1
	}

	err := pipelineBackends.RunSessionOnBatch(batch, p.BasePipeline)
	if err != nil {
		return err
	}
	atomic.AddUint64(&p.PipelineTimings.NumCalls, 1)
	atomic.AddUint64(&p.PipelineTimings.TotalNS, uint64(time.Since(start)))
	return nil
}

func (p *CrossEncoderPipeline) Postprocess(batch *pipelineBackends.PipelineBatch, documents []string) (*CrossEncoderOutput, error) {
	var outputCast [][]float32

	output := batch.OutputValues[0]

	switch v := output.(type) {
	case [][]float32:
		outputCast = make([][]float32, len(v))
		for i, logits := range v {
			scores := util.Sigmoid(logits)
			outputCast[i] = scores
		}
	default:
		return nil, fmt.Errorf("output is not 2D, expected batch size x logits, got %T", output)
	}

	results := make([]CrossEncoderResult, 0, len(documents))
	for i := range documents {
		score := outputCast[i][0]

		if p.scoreThreshold != nil && score < *p.scoreThreshold {
			atomic.AddUint64(&p.stats.FilteredResults, 1)
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

func (p *CrossEncoderPipeline) Run(inputs []string) (pipelineBackends.PipelineBatchOutput, error) {
	if len(inputs) < 2 {
		return nil, errors.New("cross encoder pipeline requires at least two inputs: a query and one or more documents")
	}
	query := inputs[0]
	documents := inputs[1:]
	return p.RunPipeline(query, documents)
}

func (p *CrossEncoderPipeline) RunPipeline(query string, documents []string) (*CrossEncoderOutput, error) {
	start := time.Now()
	defer func() {
		duration := time.Since(start)
		p.stats.TotalQueries++
		p.stats.TotalDocuments += uint64(len(documents))
		p.stats.AverageLatency += duration
	}()

	return p.runSequential(query, documents)
}

func (p *CrossEncoderPipeline) runSequential(query string, documents []string) (*CrossEncoderOutput, error) {
	var runErrors []error
	allResults := make([]CrossEncoderResult, 0, len(documents))

	for i := 0; i < len(documents); i += p.batchSize {
		end := min(i+p.batchSize, len(documents))
		out, err := p.runBatch(query, documents[i:end], i)
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
		Results:   allResults,
		QueryTime: time.Since(time.Now()),
	}

	return output, errors.Join(runErrors...)
}

func (p *CrossEncoderPipeline) runBatch(query string, documents []string, startIndex int) (*CrossEncoderOutput, error) {
	var runErrors []error
	batch := pipelineBackends.NewBatch()

	defer func(*pipelineBackends.PipelineBatch) {
		runErrors = append(runErrors, batch.Destroy())
	}(batch)

	inputs := make([]string, len(documents))
	for i, doc := range documents {
		inputs[i] = fmt.Sprintf("[CLS] %s [SEP] %s [SEP]", query, doc)
	}

	runErrors = append(runErrors, p.Preprocess(batch, inputs))
	if e := errors.Join(runErrors...); e != nil {
		return nil, e
	}

	runErrors = append(runErrors, p.Forward(batch))
	if e := errors.Join(runErrors...); e != nil {
		return nil, e
	}

	result, postErr := p.Postprocess(batch, documents)
	runErrors = append(runErrors, postErr)

	if result != nil {
		for i := range result.Results {
			result.Results[i].Index += startIndex
		}
	}

	return result, errors.Join(runErrors...)
}
