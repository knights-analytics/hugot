package pipelines

import (
	"errors"
	"fmt"
	"sync/atomic"
	"time"

	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/util/safeconv"
)

// Seq2SeqPipeline enables encoder-decoder inference for T5, BART, and similar models.
// This is useful for doc2query, translation, summarization, and other seq2seq tasks.
//
// The pipeline expects models exported in the fastT5 format with three ONNX files:
//   - encoder.onnx (or {model_name}-encoder-quantized.onnx)
//   - decoder-init.onnx (first decoding step, no past_key_values)
//   - decoder.onnx (subsequent steps, with past_key_values)
//
// Decoding methods:
//   - Greedy search (default)
//   - Top-p (nucleus) sampling with temperature
//
// Example usage:
//
//	session, err := hugot.NewORTSession()
//	check(err)
//	defer session.Destroy()
//
//	config := backends.PipelineConfig[*Seq2SeqPipeline]{
//		ModelPath: "./models/doc2query-t5-small",
//		Name:      "doc2query",
//		Options: []backends.PipelineOption[*Seq2SeqPipeline]{
//			WithSeq2SeqMaxTokens(64),
//			WithNumReturnSequences(5),
//		},
//	}
//
//	pipeline, err := NewSeq2SeqPipeline(config, session.Options, nil)
//	check(err)
//
//	output, err := pipeline.Run([]string{"Python is an interpreted programming language."})
//	// output.GeneratedTexts[0] = ["what is python", "python programming", ...]
type Seq2SeqPipeline struct {
	// Models - seq2seq requires separate encoder and decoder models
	EncoderModel     *backends.Model
	DecoderInitModel *backends.Model
	DecoderModel     *backends.Model

	// Shared tokenizer (encoder and decoder typically share)
	Tokenizer *backends.Tokenizer

	// Pipeline configuration
	PipelineName    string
	PipelineTimings *seq2seqTimings
	Runtime         string

	// Generation parameters
	MaxNewTokens      int
	NumReturnSeqs     int
	DoSample          bool
	TopP              float32
	Temperature       float32
	RepetitionPenalty float32

	// Special token IDs
	DecoderStartTokenID int64
	EosTokenIDs         map[int64]bool
	PadTokenID          int64

	// Model dimensions (from config.json)
	NumDecoderLayers int
	NumHeads         int
	HeadDim          int
	DModel           int // Hidden size
	VocabSize        int
}

// seq2seqTimings tracks timing statistics for seq2seq pipeline components.
type seq2seqTimings struct {
	TokenizerNumCalls uint64
	TokenizerTotalNS  uint64
	EncoderNumCalls   uint64
	EncoderTotalNS    uint64
	DecoderNumCalls   uint64
	DecoderTotalNS    uint64
}

// Seq2SeqOutput contains the generated sequences from the pipeline.
type Seq2SeqOutput struct {
	// GeneratedTexts[i][j] is the j-th generated sequence for the i-th input
	GeneratedTexts [][]string
	// GeneratedTokens[i][j] is the token IDs for GeneratedTexts[i][j]
	GeneratedTokens [][][]uint32
}

// GetOutput implements backends.PipelineBatchOutput interface.
func (o *Seq2SeqOutput) GetOutput() []any {
	out := make([]any, len(o.GeneratedTexts))
	for i, texts := range o.GeneratedTexts {
		out[i] = texts
	}
	return out
}

// Seq2SeqBatch holds the intermediate state during seq2seq generation.
type Seq2SeqBatch struct {
	// Input
	Inputs            []string
	InputTokenIDs     [][]int64
	InputAttentionMask [][]int64
	Size              int
	MaxInputLength    int

	// Encoder output (cached for decoder)
	EncoderHiddenStates any // Backend-specific tensor
	EncoderAttentionMask any

	// Decoder state
	DecoderInputIDs [][]int64
	PastKeyValues   []any // List of KV cache tensors
	Logits          any

	// Generation tracking
	GeneratedTokens [][]int64
	Finished        []bool
	FinishedCount   int

	// Cleanup functions
	DestroyEncoder func() error
	DestroyDecoder func() error
}

// NewSeq2SeqBatch creates a new batch for seq2seq inference.
func NewSeq2SeqBatch(size int) *Seq2SeqBatch {
	return &Seq2SeqBatch{
		Size:            size,
		GeneratedTokens: make([][]int64, size),
		Finished:        make([]bool, size),
		DestroyEncoder:  func() error { return nil },
		DestroyDecoder:  func() error { return nil },
	}
}

// Destroy cleans up batch resources.
func (b *Seq2SeqBatch) Destroy() error {
	return errors.Join(b.DestroyEncoder(), b.DestroyDecoder())
}

// Pipeline options

// WithSeq2SeqMaxTokens sets the maximum number of tokens to generate.
func WithSeq2SeqMaxTokens(maxTokens int) backends.PipelineOption[*Seq2SeqPipeline] {
	return func(p *Seq2SeqPipeline) error {
		if maxTokens <= 0 {
			return errors.New("maxTokens must be positive")
		}
		p.MaxNewTokens = maxTokens
		return nil
	}
}

// WithNumReturnSequences sets how many sequences to generate per input.
// Use with DoSample=true for diverse outputs.
func WithNumReturnSequences(n int) backends.PipelineOption[*Seq2SeqPipeline] {
	return func(p *Seq2SeqPipeline) error {
		if n <= 0 {
			return errors.New("numReturnSequences must be positive")
		}
		p.NumReturnSeqs = n
		return nil
	}
}

// WithSampling enables top-p (nucleus) sampling with the given temperature.
// topP controls diversity (0.9 = consider tokens comprising top 90% probability mass)
// temperature controls randomness (higher = more random, lower = more deterministic)
func WithSampling(topP, temperature float32) backends.PipelineOption[*Seq2SeqPipeline] {
	return func(p *Seq2SeqPipeline) error {
		if topP <= 0 || topP > 1 {
			return errors.New("topP must be in (0, 1]")
		}
		if temperature <= 0 {
			return errors.New("temperature must be positive")
		}
		p.DoSample = true
		p.TopP = topP
		p.Temperature = temperature
		return nil
	}
}

// WithRepetitionPenalty sets the repetition penalty for generation.
// Values > 1.0 penalize repetition, 1.0 = no penalty.
func WithRepetitionPenalty(penalty float32) backends.PipelineOption[*Seq2SeqPipeline] {
	return func(p *Seq2SeqPipeline) error {
		if penalty <= 0 {
			return errors.New("repetitionPenalty must be positive")
		}
		p.RepetitionPenalty = penalty
		return nil
	}
}

// NewSeq2SeqPipeline creates a new seq2seq pipeline from the given model path.
// The model path should contain encoder.onnx, decoder-init.onnx, decoder.onnx,
// tokenizer.json, and config.json files.
func NewSeq2SeqPipeline(
	config backends.PipelineConfig[*Seq2SeqPipeline],
	opts *options.Options,
) (*Seq2SeqPipeline, error) {
	pipeline := &Seq2SeqPipeline{
		PipelineName:    config.Name,
		PipelineTimings: &seq2seqTimings{},
		Runtime:         opts.Backend,

		// Defaults
		MaxNewTokens:      64,
		NumReturnSeqs:     1,
		DoSample:          false,
		TopP:              0.9,
		Temperature:       1.0,
		RepetitionPenalty: 1.0,
	}

	// Apply user options
	for _, opt := range config.Options {
		if err := opt(pipeline); err != nil {
			return nil, fmt.Errorf("applying option: %w", err)
		}
	}

	// Load models
	if err := pipeline.loadModels(config.ModelPath, opts); err != nil {
		return nil, fmt.Errorf("loading models: %w", err)
	}

	// Validate pipeline
	if err := pipeline.Validate(); err != nil {
		return nil, fmt.Errorf("validation: %w", err)
	}

	return pipeline, nil
}

// loadModels loads the encoder, decoder-init, and decoder ONNX models.
func (p *Seq2SeqPipeline) loadModels(modelPath string, opts *options.Options) error {
	var err error

	// Load encoder
	p.EncoderModel, err = backends.LoadSeq2SeqEncoder(modelPath, opts)
	if err != nil {
		return fmt.Errorf("loading encoder: %w", err)
	}

	// Load decoder-init (first step, no past_key_values)
	p.DecoderInitModel, err = backends.LoadSeq2SeqDecoderInit(modelPath, opts)
	if err != nil {
		return fmt.Errorf("loading decoder-init: %w", err)
	}

	// Load decoder (with past_key_values)
	p.DecoderModel, err = backends.LoadSeq2SeqDecoder(modelPath, opts)
	if err != nil {
		return fmt.Errorf("loading decoder: %w", err)
	}

	// Load shared tokenizer
	p.Tokenizer, err = backends.LoadSeq2SeqTokenizer(modelPath, opts)
	if err != nil {
		return fmt.Errorf("loading tokenizer: %w", err)
	}

	// Load model config for special tokens and dimensions
	if err := p.loadConfig(modelPath); err != nil {
		return fmt.Errorf("loading config: %w", err)
	}

	return nil
}

// loadConfig loads model configuration from config.json.
func (p *Seq2SeqPipeline) loadConfig(modelPath string) error {
	config, err := backends.LoadSeq2SeqConfig(modelPath)
	if err != nil {
		return err
	}

	p.DecoderStartTokenID = config.DecoderStartTokenID
	p.EosTokenIDs = config.EosTokenIDs
	p.PadTokenID = config.PadTokenID
	p.NumDecoderLayers = config.NumDecoderLayers
	p.NumHeads = config.NumHeads
	p.HeadDim = config.HeadDim
	p.DModel = config.DModel
	p.VocabSize = config.VocabSize

	return nil
}

// Validate checks that the pipeline is correctly configured.
func (p *Seq2SeqPipeline) Validate() error {
	var errs []error

	if p.EncoderModel == nil {
		errs = append(errs, errors.New("encoder model not loaded"))
	}
	if p.DecoderInitModel == nil {
		errs = append(errs, errors.New("decoder-init model not loaded"))
	}
	if p.DecoderModel == nil {
		errs = append(errs, errors.New("decoder model not loaded"))
	}
	if p.Tokenizer == nil {
		errs = append(errs, errors.New("tokenizer not loaded"))
	}
	if len(p.EosTokenIDs) == 0 {
		errs = append(errs, errors.New("no EOS token IDs configured"))
	}
	if p.VocabSize == 0 {
		errs = append(errs, errors.New("vocab size is 0"))
	}
	if p.MaxNewTokens <= 0 {
		errs = append(errs, errors.New("maxNewTokens must be positive"))
	}

	return errors.Join(errs...)
}

// GetModel returns the encoder model (primary model for the pipeline).
func (p *Seq2SeqPipeline) GetModel() *backends.Model {
	return p.EncoderModel
}

// GetMetadata returns pipeline metadata.
func (p *Seq2SeqPipeline) GetMetadata() backends.PipelineMetadata {
	return backends.PipelineMetadata{}
}

// GetStatistics returns runtime statistics for the pipeline.
func (p *Seq2SeqPipeline) GetStatistics() backends.PipelineStatistics {
	stats := backends.PipelineStatistics{}

	// Tokenizer stats
	stats.TokenizerTotalTime = safeconv.U64ToDuration(p.PipelineTimings.TokenizerTotalNS)
	stats.TokenizerExecutionCount = p.PipelineTimings.TokenizerNumCalls

	// Combined ONNX stats (encoder + decoder)
	totalOnnxNS := p.PipelineTimings.EncoderTotalNS + p.PipelineTimings.DecoderTotalNS
	totalOnnxCalls := p.PipelineTimings.EncoderNumCalls + p.PipelineTimings.DecoderNumCalls
	stats.OnnxTotalTime = safeconv.U64ToDuration(totalOnnxNS)
	stats.OnnxExecutionCount = totalOnnxCalls

	return stats
}

// IsGenerative returns true as Seq2Seq is a generative model.
func (p *Seq2SeqPipeline) IsGenerative() bool {
	return true
}

// Run generates sequences for the given input texts.
func (p *Seq2SeqPipeline) Run(inputs []string) (backends.PipelineBatchOutput, error) {
	return p.RunPipeline(inputs)
}

// RunPipeline is the main entry point for seq2seq generation.
func (p *Seq2SeqPipeline) RunPipeline(inputs []string) (*Seq2SeqOutput, error) {
	batch := NewSeq2SeqBatch(len(inputs))
	batch.Inputs = inputs
	defer batch.Destroy()

	// 1. Tokenize inputs
	if err := p.Preprocess(batch); err != nil {
		return nil, fmt.Errorf("preprocess: %w", err)
	}

	// 2. Run encoder
	if err := p.Encode(batch); err != nil {
		return nil, fmt.Errorf("encode: %w", err)
	}

	// 3. Generate sequences
	if err := p.Generate(batch); err != nil {
		return nil, fmt.Errorf("generate: %w", err)
	}

	// 4. Decode tokens to text
	output, err := p.Postprocess(batch)
	if err != nil {
		return nil, fmt.Errorf("postprocess: %w", err)
	}

	return output, nil
}

// Preprocess tokenizes the input texts.
func (p *Seq2SeqPipeline) Preprocess(batch *Seq2SeqBatch) error {
	start := time.Now()

	tokenized, err := backends.TokenizeSeq2SeqInputs(batch.Inputs, p.Tokenizer, p.PadTokenID)
	if err != nil {
		return err
	}

	batch.InputTokenIDs = tokenized.TokenIDs
	batch.InputAttentionMask = tokenized.AttentionMask
	batch.MaxInputLength = tokenized.MaxLength

	atomic.AddUint64(&p.PipelineTimings.TokenizerNumCalls, 1)
	atomic.AddUint64(&p.PipelineTimings.TokenizerTotalNS, safeconv.DurationToU64(time.Since(start)))

	return nil
}

// Encode runs the encoder model on the input tokens.
func (p *Seq2SeqPipeline) Encode(batch *Seq2SeqBatch) error {
	start := time.Now()

	err := backends.RunSeq2SeqEncoder(batch, p.EncoderModel, p.Runtime)
	if err != nil {
		return err
	}

	atomic.AddUint64(&p.PipelineTimings.EncoderNumCalls, 1)
	atomic.AddUint64(&p.PipelineTimings.EncoderTotalNS, safeconv.DurationToU64(time.Since(start)))

	return nil
}

// Generate performs autoregressive decoding.
func (p *Seq2SeqPipeline) Generate(batch *Seq2SeqBatch) error {
	start := time.Now()

	var err error
	if p.DoSample {
		err = backends.RunSeq2SeqGenerationSampling(batch, p)
	} else {
		err = backends.RunSeq2SeqGenerationGreedy(batch, p)
	}

	atomic.AddUint64(&p.PipelineTimings.DecoderNumCalls, uint64(p.MaxNewTokens))
	atomic.AddUint64(&p.PipelineTimings.DecoderTotalNS, safeconv.DurationToU64(time.Since(start)))

	return err
}

// Postprocess decodes the generated token IDs back to text.
func (p *Seq2SeqPipeline) Postprocess(batch *Seq2SeqBatch) (*Seq2SeqOutput, error) {
	output := &Seq2SeqOutput{
		GeneratedTexts:  make([][]string, batch.Size),
		GeneratedTokens: make([][][]uint32, batch.Size),
	}

	for i := 0; i < batch.Size; i++ {
		// Currently returns a single sequence per input. Multiple return sequences
		// (e.g., beam search or multiple sampling runs) would require changes to
		// the generation loop to maintain multiple candidate sequences per input.
		tokens := batch.GeneratedTokens[i]
		convertedTokens := make([]uint32, len(tokens))
		for j, tok := range tokens {
			convertedTokens[j] = safeconv.Int64ToUint32(tok)
		}

		text, err := backends.Decode(convertedTokens, p.Tokenizer, true)
		if err != nil {
			return nil, fmt.Errorf("decoding output %d: %w", i, err)
		}

		output.GeneratedTexts[i] = []string{text}
		output.GeneratedTokens[i] = [][]uint32{convertedTokens}
	}

	return output, nil
}

// Destroy cleans up pipeline resources.
func (p *Seq2SeqPipeline) Destroy() error {
	var errs []error

	if p.EncoderModel != nil && p.EncoderModel.Destroy != nil {
		errs = append(errs, p.EncoderModel.Destroy())
	}
	if p.DecoderInitModel != nil && p.DecoderInitModel.Destroy != nil {
		errs = append(errs, p.DecoderInitModel.Destroy())
	}
	if p.DecoderModel != nil && p.DecoderModel.Destroy != nil {
		errs = append(errs, p.DecoderModel.Destroy())
	}
	if p.Tokenizer != nil {
		errs = append(errs, p.Tokenizer.Destroy())
	}

	return errors.Join(errs...)
}

// GetGenerationConfig returns the current generation configuration.
// Useful for debugging or logging.
func (p *Seq2SeqPipeline) GetGenerationConfig() map[string]any {
	return map[string]any{
		"max_new_tokens":      p.MaxNewTokens,
		"num_return_seqs":     p.NumReturnSeqs,
		"do_sample":           p.DoSample,
		"top_p":               p.TopP,
		"temperature":         p.Temperature,
		"repetition_penalty":  p.RepetitionPenalty,
		"decoder_start_token": p.DecoderStartTokenID,
		"eos_token_ids":       p.EosTokenIDs,
		"pad_token_id":        p.PadTokenID,
	}
}

// Interface implementations for backends.Seq2SeqPipelineInterface

func (p *Seq2SeqPipeline) GetEncoderModel() *backends.Model     { return p.EncoderModel }
func (p *Seq2SeqPipeline) GetDecoderInitModel() *backends.Model { return p.DecoderInitModel }
func (p *Seq2SeqPipeline) GetDecoderModel() *backends.Model     { return p.DecoderModel }
func (p *Seq2SeqPipeline) GetTokenizer() *backends.Tokenizer    { return p.Tokenizer }
func (p *Seq2SeqPipeline) GetRuntime() string                   { return p.Runtime }
func (p *Seq2SeqPipeline) GetMaxNewTokens() int                 { return p.MaxNewTokens }
func (p *Seq2SeqPipeline) GetNumReturnSeqs() int                { return p.NumReturnSeqs }
func (p *Seq2SeqPipeline) GetDoSample() bool                    { return p.DoSample }
func (p *Seq2SeqPipeline) GetTopP() float32                     { return p.TopP }
func (p *Seq2SeqPipeline) GetTemperature() float32              { return p.Temperature }
func (p *Seq2SeqPipeline) GetRepetitionPenalty() float32        { return p.RepetitionPenalty }
func (p *Seq2SeqPipeline) GetDecoderStartTokenID() int64        { return p.DecoderStartTokenID }
func (p *Seq2SeqPipeline) GetEosTokenIDs() map[int64]bool       { return p.EosTokenIDs }
func (p *Seq2SeqPipeline) GetPadTokenID() int64                 { return p.PadTokenID }
func (p *Seq2SeqPipeline) GetNumDecoderLayers() int             { return p.NumDecoderLayers }
func (p *Seq2SeqPipeline) GetVocabSize() int                    { return p.VocabSize }

// Interface implementations for backends.Seq2SeqBatchInterface

func (b *Seq2SeqBatch) GetSize() int                       { return b.Size }
func (b *Seq2SeqBatch) GetInputTokenIDs() [][]int64        { return b.InputTokenIDs }
func (b *Seq2SeqBatch) GetInputAttentionMask() [][]int64   { return b.InputAttentionMask }
func (b *Seq2SeqBatch) GetMaxInputLength() int             { return b.MaxInputLength }
func (b *Seq2SeqBatch) SetEncoderHiddenStates(states any)  { b.EncoderHiddenStates = states }
func (b *Seq2SeqBatch) GetEncoderHiddenStates() any        { return b.EncoderHiddenStates }
func (b *Seq2SeqBatch) SetEncoderAttentionMask(mask any)   { b.EncoderAttentionMask = mask }
func (b *Seq2SeqBatch) GetEncoderAttentionMask() any       { return b.EncoderAttentionMask }
func (b *Seq2SeqBatch) SetPastKeyValues(pkv []any)         { b.PastKeyValues = pkv }
func (b *Seq2SeqBatch) GetPastKeyValues() []any            { return b.PastKeyValues }
func (b *Seq2SeqBatch) SetLogits(logits any)               { b.Logits = logits }
func (b *Seq2SeqBatch) GetLogits() any                     { return b.Logits }
func (b *Seq2SeqBatch) GetGeneratedTokens() [][]int64      { return b.GeneratedTokens }
func (b *Seq2SeqBatch) SetGeneratedTokens(tokens [][]int64) { b.GeneratedTokens = tokens }
func (b *Seq2SeqBatch) GetFinished() []bool                { return b.Finished }
func (b *Seq2SeqBatch) SetFinished(finished []bool)        { b.Finished = finished }
func (b *Seq2SeqBatch) GetFinishedCount() int              { return b.FinishedCount }
func (b *Seq2SeqBatch) SetFinishedCount(count int)         { b.FinishedCount = count }
func (b *Seq2SeqBatch) SetDestroyEncoder(fn func() error)  { b.DestroyEncoder = fn }
func (b *Seq2SeqBatch) SetDestroyDecoder(fn func() error)  { b.DestroyDecoder = fn }

// Helper functions for different seq2seq model input formats

// FormatLMQGInput formats input for LMQG question generation models.
// These models expect input in the format: "generate question: <hl> {answer} <hl> {context}"
// where the answer is highlighted within the context.
//
// Example:
//
//	input := FormatLMQGInput("Beyonce", "Beyonce starred as Etta James in Cadillac Records.")
//	// Returns: "generate question: <hl> Beyonce <hl> Beyonce starred as Etta James in Cadillac Records."
func FormatLMQGInput(answer, context string) string {
	return "generate question: <hl> " + answer + " <hl> " + context
}

// FormatLMQGInputBatch formats multiple answer-context pairs for LMQG models.
func FormatLMQGInputBatch(pairs []AnswerContextPair) []string {
	inputs := make([]string, len(pairs))
	for i, pair := range pairs {
		inputs[i] = FormatLMQGInput(pair.Answer, pair.Context)
	}
	return inputs
}

// AnswerContextPair holds an answer and its context for question generation.
type AnswerContextPair struct {
	Answer  string
	Context string
}

// RunQuestionGeneration is a convenience method for LMQG-style question generation.
// It formats the inputs and runs the pipeline.
func (p *Seq2SeqPipeline) RunQuestionGeneration(pairs []AnswerContextPair) (*Seq2SeqOutput, error) {
	inputs := FormatLMQGInputBatch(pairs)
	return p.RunPipeline(inputs)
}
