package pipelines

import (
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

func WithSortResults(sort bool) backends.PipelineOption[*CrossEncoderPipeline] {
	return func(p *CrossEncoderPipeline) error {
		p.sortResults = sort
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

func NewCrossEncoderPipeline(config backends.PipelineConfig[*CrossEncoderPipeline], s *options.Options, model *backends.Model) (*CrossEncoderPipeline, error) {
	defaultPipeline, err := backends.NewBasePipeline(config, s, model)
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

func patchBertSequenceTokenTypeIDs(batch *backends.PipelineBatch, sepToken string) {
	// Fix token_type_ids for BERT-style models when we manually concatenated the pair as a single sequence.
	// Pattern expected: [CLS] query [SEP] doc [SEP]
	// HF sets token_type_ids=0 up to and including first [SEP], then 1 for remainder (including final [SEP]).
	for index := range batch.Input {
		input := &batch.Input[index]
		// Only adjust if type ids exist and are all zero
		allZero := true
		for _, t := range input.TypeIDs {
			if t != 0 {
				allZero = false
				break
			}
		}
		if !allZero || len(input.TypeIDs) == 0 {
			continue
		}
		// Find first [SEP] token index (skip position 0 which should be [CLS])
		firstSep := -1
		for iTok := 1; iTok < len(input.Tokens); iTok++ {
			if input.Tokens[iTok] == sepToken {
				firstSep = iTok
				break
			}
		}
		if firstSep == -1 || firstSep == len(input.Tokens)-1 { // nothing to split
			continue
		}
		for iTok := firstSep + 1; iTok < len(input.TypeIDs); iTok++ {
			input.TypeIDs[iTok] = 1
		}
	}
}

func (p *CrossEncoderPipeline) Preprocess(batch *backends.PipelineBatch, inputs []string) error {
	start := time.Now()
	backends.TokenizeInputs(batch, p.Model.Tokenizer, inputs)
	if p.Model != nil && p.Model.Tokenizer != nil && p.Model.SeparatorToken == "[SEP]" {
		patchBertSequenceTokenTypeIDs(batch, p.Model.SeparatorToken)
	}
	atomic.AddUint64(&p.Model.Tokenizer.TokenizerTimings.NumCalls, 1)
	atomic.AddUint64(&p.Model.Tokenizer.TokenizerTimings.TotalNS, safeconv.DurationToU64(time.Since(start)))
	err := backends.CreateInputTensors(batch, p.Model, p.Runtime)
	return err
}

func (p *CrossEncoderPipeline) Forward(batch *backends.PipelineBatch) error {
	start := time.Now()
	if p.Model.Tokenizer.MaxAllowedTokens >= p.Model.MaxPositionEmbeddings {
		p.Model.Tokenizer.MaxAllowedTokens = p.Model.MaxPositionEmbeddings - 1
	}
	err := backends.RunSessionOnBatch(batch, p.BasePipeline)
	if err != nil {
		return err
	}
	atomic.AddUint64(&p.PipelineTimings.NumCalls, 1)
	atomic.AddUint64(&p.PipelineTimings.TotalNS, safeconv.DurationToU64(time.Since(start)))
	return nil
}

func (p *CrossEncoderPipeline) Postprocess(batch *backends.PipelineBatch, documents []string) (*CrossEncoderOutput, error) {
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

func (p *CrossEncoderPipeline) Run(inputs []string) (backends.PipelineBatchOutput, error) {
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
		p.statistics.TotalQueries++
		p.statistics.TotalDocuments += uint64(len(documents))
		p.statistics.AverageLatency += duration
	}()
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
		Results: allResults,
	}
	return output, errors.Join(runErrors...)
}

func (p *CrossEncoderPipeline) runBatch(query string, documents []string, startIndex int) (*CrossEncoderOutput, error) {
	var runErrors []error
	inputs := make([]string, len(documents))
	sep := p.Model.SeparatorToken
	for i, doc := range documents {
		if sep == "</s>" {
			// RoBERTa style: query </s> </s> document
			inputs[i] = fmt.Sprintf("%s%s%s%s", query, sep, sep, doc)
		} else {
			// BERT style: query [SEP] document [SEP]
			inputs[i] = fmt.Sprintf("%s%s%s", query, sep, doc)
		}
	}
	batch := backends.NewBatch(len(inputs))
	defer func(*backends.PipelineBatch) {
		runErrors = append(runErrors, batch.Destroy())
	}(batch)
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
