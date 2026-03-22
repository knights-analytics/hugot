package pipelines

// QuestionAnsweringPipeline is a go implementation of Hugging Face's question answering pipeline.
// https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/question_answering.py
//
// It supports extractive QA: given a question and a context passage the model predicts start and
// end logits over the context tokens and the top-k highest-scoring spans are returned as answers.
// By default (TopK=1) only the single best span is returned per input.
//
// The underlying ONNX model is expected to expose two outputs in order:
//
//	0 – start_logits  [batch, sequence_length]
//	1 – end_logits    [batch, sequence_length]
//
// Typical models: distilbert-base-uncased-distilled-squad, bert-large-uncased-whole-word-masking-finetuned-squad.

import (
	"errors"
	"fmt"
	"slices"
	"sync/atomic"
	"time"

	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/util/safeconv"
	"github.com/knights-analytics/hugot/util/vectorutil"
)

// QuestionAnsweringInput holds a single question/context pair.
type QuestionAnsweringInput struct {
	Question string
	Context  string
}

// QuestionAnsweringOutput holds the result for a single question/context pair.
type QuestionAnsweringOutput struct {
	// Answer is the extracted answer string from Context.
	Answer string
	// Score is start_prob[best_start] * end_prob[best_end].
	Score float32
	// Start is the byte offset of the answer start inside Context.
	Start uint
	// End is the byte offset (exclusive) of the answer end inside Context.
	End uint
}

// QuestionAnsweringBatchOutput holds results for a whole batch.
// Each element of Outputs corresponds to one input and contains answers ranked by score (best first).
// With the default TopK=1 each inner slice has exactly one element.
type QuestionAnsweringBatchOutput struct {
	Outputs [][]QuestionAnsweringOutput
}

func (o *QuestionAnsweringBatchOutput) GetOutput() []any {
	out := make([]any, len(o.Outputs))
	for i, result := range o.Outputs {
		out[i] = any(result)
	}
	return out
}

// QuestionAnsweringPipeline holds the pipeline configuration.
type QuestionAnsweringPipeline struct {
	*backends.BasePipeline
	// MaxAnswerLength is the maximum number of tokens allowed in the answer span (default 15).
	MaxAnswerLength int
	// TopK is the number of ranked answer spans to return per input (default 1).
	TopK int
}

// PIPELINE OPTIONS

// WithMaxAnswerLength sets the maximum number of tokens that the answer span may cover.
func WithMaxAnswerLength(n int) backends.PipelineOption[*QuestionAnsweringPipeline] {
	return func(p *QuestionAnsweringPipeline) error {
		p.MaxAnswerLength = n
		return nil
	}
}

// WithTopKAnswers sets the number of ranked answer spans to return per input.
// When k > 1 each element of QuestionAnsweringBatchOutput.Outputs is a slice of up to k answers
// sorted by score descending.
func WithTopKAnswers(k int) backends.PipelineOption[*QuestionAnsweringPipeline] {
	return func(p *QuestionAnsweringPipeline) error {
		p.TopK = k
		return nil
	}
}

// NewQuestionAnsweringPipeline initialises a question answering pipeline.
func NewQuestionAnsweringPipeline(config backends.PipelineConfig[*QuestionAnsweringPipeline], s *options.Options, model *backends.Model) (*QuestionAnsweringPipeline, error) {
	defaultPipeline, err := backends.NewBasePipeline(config, s, model)
	if err != nil {
		return nil, err
	}
	pipeline := &QuestionAnsweringPipeline{
		BasePipeline:    defaultPipeline,
		MaxAnswerLength: 15,
		TopK:            1,
	}
	for _, o := range config.Options {
		if err = o(pipeline); err != nil {
			return nil, err
		}
	}
	if err = pipeline.Validate(); err != nil {
		return nil, err
	}
	err = backends.AllInputTokens(pipeline.BasePipeline)
	if err != nil {
		return nil, err
	}
	return pipeline, nil
}

// INTERFACE IMPLEMENTATIONS

func (p *QuestionAnsweringPipeline) IsGenerative() bool {
	return false
}

func (p *QuestionAnsweringPipeline) GetModel() *backends.Model {
	return p.Model
}

// GetMetadata returns metadata for both output tensors (start_logits, end_logits).
func (p *QuestionAnsweringPipeline) GetMetadata() backends.PipelineMetadata {
	infos := make([]backends.OutputInfo, 0, len(p.Model.OutputsMeta))
	for _, o := range p.Model.OutputsMeta {
		infos = append(infos, backends.OutputInfo{
			Name:       o.Name,
			Dimensions: o.Dimensions,
		})
	}
	return backends.PipelineMetadata{OutputsInfo: infos}
}

// GetStatistics returns runtime statistics for the pipeline.
func (p *QuestionAnsweringPipeline) GetStatistics() backends.PipelineStatistics {
	stats := backends.PipelineStatistics{}
	if p.Model.Tokenizer != nil && p.Model.Tokenizer.TokenizerTimings != nil {
		stats.ComputeTokenizerStatistics(p.Model.Tokenizer.TokenizerTimings)
	}
	stats.ComputeOnnxStatistics(p.PipelineTimings)
	return stats
}

// Validate checks that the pipeline configuration is valid.
func (p *QuestionAnsweringPipeline) Validate() error {
	var errs []error
	if p.Model.Tokenizer == nil {
		errs = append(errs, fmt.Errorf("question answering pipeline requires a tokenizer"))
	}
	if p.Model.SeparatorToken == "" {
		errs = append(errs, fmt.Errorf("question answering pipeline requires a separator token (e.g. [SEP] or </s>)"))
	}
	if len(p.Model.OutputsMeta) < 2 {
		errs = append(errs, fmt.Errorf("question answering model must expose at least 2 outputs (start_logits, end_logits); got %d", len(p.Model.OutputsMeta)))
	} else {
		for idx, out := range p.Model.OutputsMeta[:2] {
			if len(out.Dimensions) != 2 {
				errs = append(errs, fmt.Errorf("output %d (%s) must be 2-dimensional [batch, sequence]; got %d dims", idx, out.Name, len(out.Dimensions)))
			}
		}
	}
	if p.MaxAnswerLength <= 0 {
		errs = append(errs, fmt.Errorf("MaxAnswerLength must be positive, got %d", p.MaxAnswerLength))
	}
	if p.TopK <= 0 {
		errs = append(errs, fmt.Errorf("TopK must be positive, got %d", p.TopK))
	}
	return errors.Join(errs...)
}

// Preprocess tokenises each question/context pair into a single combined sequence.
func (p *QuestionAnsweringPipeline) Preprocess(batch *backends.PipelineBatch, inputs []QuestionAnsweringInput) error {
	start := time.Now()
	inputPairs := make([][2]string, 0, len(inputs))
	for _, in := range inputs {
		inputPairs = append(inputPairs, [2]string{in.Question, in.Context})
	}
	backends.TokenizeInputPairs(batch, p.Model.Tokenizer, inputPairs, p.Model.SeparatorToken)
	atomic.AddUint64(&p.Model.Tokenizer.TokenizerTimings.NumCalls, 1)
	atomic.AddUint64(&p.Model.Tokenizer.TokenizerTimings.TotalNS, safeconv.DurationToU64(time.Since(start)))
	err := backends.CreateInputTensors(batch, p.Model, p.Runtime)
	return err
}

// Forward runs the ONNX session on the batch.
func (p *QuestionAnsweringPipeline) Forward(batch *backends.PipelineBatch) error {
	start := time.Now()
	if err := backends.RunSessionOnBatch(batch, p.BasePipeline); err != nil {
		return err
	}
	atomic.AddUint64(&p.PipelineTimings.NumCalls, 1)
	atomic.AddUint64(&p.PipelineTimings.TotalNS, safeconv.DurationToU64(time.Since(start)))
	return nil
}

// Postprocess extracts the top-k answer spans for each item in the batch, ranked by score descending.
func (p *QuestionAnsweringPipeline) Postprocess(batch *backends.PipelineBatch, inputs []QuestionAnsweringInput) (*QuestionAnsweringBatchOutput, error) {
	if len(batch.OutputValues) < 2 {
		return nil, fmt.Errorf("expected at least 2 output tensors (start_logits, end_logits), got %d", len(batch.OutputValues))
	}
	startLogits, ok1 := batch.OutputValues[0].([][]float32)
	endLogits, ok2 := batch.OutputValues[1].([][]float32)
	if !ok1 || !ok2 {
		return nil, fmt.Errorf("start_logits and end_logits must be [][]float32; got %T and %T", batch.OutputValues[0], batch.OutputValues[1])
	}

	results := make([][]QuestionAnsweringOutput, batch.Size)
	for i := range batch.Size {
		input := batch.Input[i]
		seqLen := len(input.Tokens)

		// Convert logits to probabilities over the sequence dimension.
		startProbs := vectorutil.SoftMax(startLogits[i][:seqLen])
		endProbs := vectorutil.SoftMax(endLogits[i][:seqLen])

		// Determine the first context token (type_id == 1 or after the separator).
		ctxStart := contextStartIndex(input, p.Model.SeparatorToken)

		// Collect the top-k scoring spans and map them to character offsets.
		candidates := rankedSpans(startProbs, endProbs, ctxStart, seqLen, p.MaxAnswerLength, p.TopK)
		ranked := make([]QuestionAnsweringOutput, len(candidates))
		for j, c := range candidates {
			answerStart, answerEnd := spanToContextChars(input, c.start, c.end, inputs[i].Context)
			ranked[j] = QuestionAnsweringOutput{
				Answer: inputs[i].Context[answerStart:answerEnd],
				Score:  c.score,
				Start:  answerStart,
				End:    answerEnd,
			}
		}
		results[i] = ranked
	}
	return &QuestionAnsweringBatchOutput{Outputs: results}, nil
}

// Run implements the Pipeline interface. Inputs are interleaved [question0, context0, question1, context1, …].
func (p *QuestionAnsweringPipeline) Run(inputs []string) (backends.PipelineBatchOutput, error) {
	if len(inputs) == 0 || len(inputs)%2 != 0 {
		return nil, fmt.Errorf("question answering pipeline requires an even, non-zero number of strings: [question0, context0, question1, context1, ...]")
	}
	qaInputs := make([]QuestionAnsweringInput, len(inputs)/2)
	for i := 0; i < len(inputs); i += 2 {
		qaInputs[i/2] = QuestionAnsweringInput{Question: inputs[i], Context: inputs[i+1]}
	}
	return p.RunPipeline(qaInputs)
}

// RunPipeline is the typed entry point for the question answering pipeline.
func (p *QuestionAnsweringPipeline) RunPipeline(inputs []QuestionAnsweringInput) (*QuestionAnsweringBatchOutput, error) {
	var runErrors []error
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

	result, postErr := p.Postprocess(batch, inputs)
	runErrors = append(runErrors, postErr)
	return result, errors.Join(runErrors...)
}

// HELPERS

// spanCandidate holds a scored token span.
type spanCandidate struct {
	start, end int
	score      float32
}

// rankedSpans returns up to k highest-scoring valid (start, end) token spans in descending score order.
// Spans are non-overlapping: once a span is selected, any candidate that shares at least one token
// with it is discarded before the next pick.
func rankedSpans(startProbs, endProbs []float32, ctxStart, seqLen, maxAnswerLen, k int) []spanCandidate {
	candidates := make([]spanCandidate, 0, seqLen*maxAnswerLen)
	for s := ctxStart; s < seqLen; s++ {
		for e := s; e < min(s+maxAnswerLen, seqLen); e++ {
			candidates = append(candidates, spanCandidate{s, e, startProbs[s] * endProbs[e]})
		}
	}
	slices.SortFunc(candidates, func(a, b spanCandidate) int {
		if a.score > b.score {
			return -1
		}
		if a.score < b.score {
			return 1
		}
		return 0
	})

	selected := make([]spanCandidate, 0, k)
	for _, c := range candidates {
		if len(selected) == k {
			break
		}
		overlaps := false
		for _, s := range selected {
			if c.start <= s.end && s.start <= c.end {
				overlaps = true
				break
			}
		}
		if !overlaps {
			selected = append(selected, c)
		}
	}
	return selected
}

// contextStartIndex returns the index of the first token that belongs to the context
// (i.e. the answer candidate region). It prefers type_id == 1 when available; otherwise
// it falls back to finding the separator token and returning the index after it.
func contextStartIndex(input backends.TokenizedInput, sepToken string) int {
	for i, tid := range input.TypeIDs {
		if tid == 1 {
			return i
		}
	}
	// Fallback: skip past the first occurrence of the separator token.
	for i := 1; i < len(input.Tokens); i++ {
		if input.Tokens[i] == sepToken {
			if i+1 < len(input.Tokens) {
				return i + 1
			}
		}
	}
	return 0
}

// spanToContextChars converts a [startTok, endTok] token span into byte offsets that are
// relative to the original context string (not the combined question+sep+context string).
//
// The token offsets stored in TokenizedInput are relative to the full combined string;
// we derive the context character offset from the first token whose type_id == 1 (or the
// first token after the separator, as a fallback).
func spanToContextChars(input backends.TokenizedInput, startTok, endTok int, context string) (uint, uint) {
	if len(input.Offsets) == 0 {
		return 0, 0
	}

	// Find the character offset where the context begins in the combined string.
	ctxCharOffset := uint(0)
	found := false
	for i, tid := range input.TypeIDs {
		if tid == 1 && i < len(input.Offsets) {
			ctxCharOffset = input.Offsets[i][0]
			found = true
			break
		}
	}
	// Fallback: when TypeIDs are not populated (e.g. DistilBERT which has no token_type_ids input),
	// derive the context's start offset from the raw combined string.
	// input.Raw = question + separator + context, so the context begins at len(input.Raw) - len(context).
	if !found && len(input.Raw) >= len(context) {
		diff := len(input.Raw) - len(context) // non-negative by the guard above
		ctxCharOffset = uint(diff)            //nolint:gosec // diff is guaranteed non-negative
	}

	if startTok >= len(input.Offsets) || endTok >= len(input.Offsets) {
		return 0, 0
	}

	rawStart := input.Offsets[startTok][0]
	rawEnd := input.Offsets[endTok][1]

	if rawStart < ctxCharOffset {
		rawStart = ctxCharOffset
	}
	if rawEnd < rawStart {
		rawEnd = rawStart
	}

	answerStart := rawStart - ctxCharOffset
	answerEnd := rawEnd - ctxCharOffset

	// Clamp to context length.
	contextLen := uint(len(context))
	if answerStart > contextLen {
		answerStart = contextLen
	}
	if answerEnd > contextLen {
		answerEnd = contextLen
	}
	return answerStart, answerEnd
}
