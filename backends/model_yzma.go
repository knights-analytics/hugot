//go:build YZMA || ALL

package backends

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/hybridgroup/yzma/pkg/llama"

	"github.com/knights-analytics/hugot/options"
)

// yzmaLibOnce ensures llama.Load is called exactly once per process.
var (
	yzmaLibOnce sync.Once
	errYzmaLib  error
)

// YZMAModel holds all llama.cpp resources for a YZMA generative session.
type YZMAModel struct {
	mu           sync.Mutex
	lmodel       llama.Model
	lctx         llama.Context
	vocab        llama.Vocab
	chatTemplate string

	// cumulative statistics
	statsMu                        sync.Mutex
	cumulativeTokens               int
	cumulativeTokenDurationSeconds float64
	cumulativePrefillSum           float64
	cumulativePrefillCount         int
}

// createYZMAGenerativeSession loads a GGUF model from model.Path and initialises a llama.cpp
// context, storing the result in model.YZMAModel.
func createYZMAGenerativeSession(_ context.Context, model *Model, opts *options.Options) error {
	yopts := opts.YZMAOptions
	if yopts == nil {
		yopts = &options.YZMAOptions{}
	}

	// Load the shared library once per process.
	libPath := ""
	if yopts.LibPath != nil {
		libPath = *yopts.LibPath
	}
	yzmaLibOnce.Do(func() {
		errYzmaLib = llama.Load(libPath)
	})
	if errYzmaLib != nil {
		return fmt.Errorf("yzma: failed to load llama.cpp library: %w", errYzmaLib)
	}

	llama.Init()

	mParams := llama.ModelDefaultParams()
	mParams.NGpuLayers = yopts.NGpuLayers

	lmodel, err := llama.ModelLoadFromFile(model.Path, mParams)
	if err != nil {
		return fmt.Errorf("yzma: failed to load model from %q: %w", model.Path, err)
	}

	ctxParams := llama.ContextDefaultParams()
	if yopts.NCtx > 0 {
		ctxParams.NCtx = yopts.NCtx
	}
	if yopts.NThreads > 0 {
		ctxParams.NThreads = yopts.NThreads
		ctxParams.NThreadsBatch = yopts.NThreads
	}

	lctx, err := llama.InitFromModel(lmodel, ctxParams)
	if err != nil {
		_ = llama.ModelFree(lmodel)
		return fmt.Errorf("yzma: failed to initialise context: %w", err)
	}

	vocab := llama.ModelGetVocab(lmodel)

	// Detect the model's preferred chat template; fall back to chatml.
	chatTemplate := llama.ModelChatTemplate(lmodel, "")
	if chatTemplate == "" {
		chatTemplate = "chatml"
	}

	model.YZMAModel = &YZMAModel{
		lmodel:       lmodel,
		lctx:         lctx,
		vocab:        vocab,
		chatTemplate: chatTemplate,
	}
	return nil
}

// Generate implements the GenerativeModel interface.
// Each conversation in inputs is processed sequentially; tokens are streamed via the returned channels.
func (y *YZMAModel) Generate(ctx context.Context, inputs [][]Message, _ []string, genOpts *GenerativeOptions) (chan SequenceDelta, chan error, error) {
	if genOpts == nil {
		genOpts = &GenerativeOptions{}
	}

	tokenStream := make(chan SequenceDelta, 64)
	errorStream := make(chan error, 1)

	go func() {
		defer close(tokenStream)

		for seqIdx, conversation := range inputs {
			select {
			case <-ctx.Done():
				errorStream <- ctx.Err()
				return
			default:
			}

			if err := y.generateOne(ctx, seqIdx, conversation, genOpts, tokenStream); err != nil {
				errorStream <- err
				return
			}
		}
	}()

	return tokenStream, errorStream, nil
}

// generateOne runs the decode loop for a single conversation and sends tokens to tokenStream.
func (y *YZMAModel) generateOne(ctx context.Context, seqIdx int, conversation []Message, opts *GenerativeOptions, tokenStream chan<- SequenceDelta) error {
	y.mu.Lock()
	defer y.mu.Unlock()

	// Build the chat template input.
	chatMsgs := make([]llama.ChatMessage, len(conversation))
	for i, m := range conversation {
		chatMsgs[i] = llama.NewChatMessage(m.Role, m.Content)
	}

	// Apply the chat template to produce a formatted prompt string.
	buf := make([]byte, 8192)
	n := llama.ChatApplyTemplate(y.chatTemplate, chatMsgs, true, buf)
	if n < 0 {
		// Retry with a larger buffer.
		buf = make([]byte, -n+1)
		n = llama.ChatApplyTemplate(y.chatTemplate, chatMsgs, true, buf)
		if n < 0 {
			return fmt.Errorf("yzma: chat template application failed for sequence %d", seqIdx)
		}
	}
	prompt := string(buf[:n])

	// Tokenize the prompt.
	tokens := llama.Tokenize(y.vocab, prompt, true, false)
	if len(tokens) == 0 {
		return nil
	}

	// Build the sampler chain.
	sampler := y.buildSampler(opts)
	defer llama.SamplerFree(sampler)

	// Clear the KV cache for a fresh context.
	mem, memErr := llama.GetMemory(y.lctx)
	if memErr == nil {
		_ = llama.MemoryClear(mem, false)
	}

	// Prefill: process the prompt tokens in one batch.
	prefillStart := time.Now()
	batch := llama.BatchGetOne(tokens)
	if _, decErr := llama.Decode(y.lctx, batch); decErr != nil {
		return fmt.Errorf("yzma: prefill decode error for sequence %d: %w", seqIdx, decErr)
	}
	prefillDuration := time.Since(prefillStart).Seconds()

	maxNewTokens := opts.MaxLength
	if maxNewTokens <= 0 {
		maxNewTokens = 1028
	}

	// Generation loop.
	genStart := time.Now()
	var tokenCount int
	var outputBuilder strings.Builder

	for range maxNewTokens {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		token := llama.SamplerSample(sampler, y.lctx, -1)
		llama.SamplerAccept(sampler, token)

		if llama.VocabIsEOG(y.vocab, token) {
			break
		}

		pieceBuf := make([]byte, 256)
		pieceLen := llama.TokenToPiece(y.vocab, token, pieceBuf, 0, true)
		if pieceLen <= 0 {
			// Advance to next token without emitting.
			nextBatch := llama.BatchGetOne([]llama.Token{token})
			if _, decErr := llama.Decode(y.lctx, nextBatch); decErr != nil {
				return fmt.Errorf("yzma: generation decode error at token %d: %w", tokenCount, decErr)
			}
			continue
		}
		piece := string(pieceBuf[:pieceLen])
		outputBuilder.WriteString(piece)
		tokenCount++

		tokenStream <- SequenceDelta{Token: piece, Sequence: seqIdx}

		nextBatch := llama.BatchGetOne([]llama.Token{token})
		if _, decErr := llama.Decode(y.lctx, nextBatch); decErr != nil {
			return fmt.Errorf("yzma: generation decode error at token %d: %w", tokenCount, decErr)
		}
	}

	genDuration := time.Since(genStart).Seconds()

	// Update statistics.
	y.statsMu.Lock()
	y.cumulativeTokens += tokenCount
	y.cumulativeTokenDurationSeconds += genDuration
	y.cumulativePrefillSum += prefillDuration
	y.cumulativePrefillCount++
	y.statsMu.Unlock()

	return nil
}

// buildSampler constructs a llama sampler chain from GenerativeOptions.
func (y *YZMAModel) buildSampler(opts *GenerativeOptions) llama.Sampler {
	chain := llama.SamplerChainInit(llama.SamplerChainDefaultParams())

	// Grammar-based guidance (LarkGrammar only; JSON Schema / Regex deferred).
	if opts.Guidance != nil && opts.Guidance.Type == GuidanceTypeLarkGrammar && opts.Guidance.Data != "" {
		grammarSampler := llama.SamplerInitGrammar(y.vocab, opts.Guidance.Data, "root")
		llama.SamplerChainAdd(chain, grammarSampler)
	}

	// Temperature / TopP.
	var temperature float32 = 0.8
	if opts.Temperature != nil {
		temperature = float32(*opts.Temperature)
	}
	var topP float32 = 0.95
	if opts.TopP != nil {
		topP = float32(*opts.TopP)
	}

	llama.SamplerChainAdd(chain, llama.SamplerInitTopP(topP, 1))
	llama.SamplerChainAdd(chain, llama.SamplerInitTempExt(temperature, 0, 1))

	// Seed.
	var seed uint32 = llama.DefaultSeed
	if opts.Seed != nil {
		seed = uint32(*opts.Seed) //nolint:gosec // user-supplied seed; overflow is intentional (wraps mod 2^32)
	}
	llama.SamplerChainAdd(chain, llama.SamplerInitDist(seed))

	return chain
}

// GetStatistics implements GenerativeModel.
func (y *YZMAModel) GetStatistics() PipelineStatistics {
	y.statsMu.Lock()
	defer y.statsMu.Unlock()

	var tps float64
	if y.cumulativeTokenDurationSeconds > 0 {
		tps = float64(y.cumulativeTokens) / y.cumulativeTokenDurationSeconds
	}
	var avgPrefill float64
	if y.cumulativePrefillCount > 0 {
		avgPrefill = y.cumulativePrefillSum / float64(y.cumulativePrefillCount)
	}

	return PipelineStatistics{
		AvgPrefillSeconds:              avgPrefill,
		TokensPerSecond:                tps,
		CumulativePrefillSum:           y.cumulativePrefillSum,
		CumulativePrefillCount:         y.cumulativePrefillCount,
		CumulativeTokens:               y.cumulativeTokens,
		CumulativeTokenDurationSeconds: y.cumulativeTokenDurationSeconds,
	}
}

// Destroy implements GenerativeModel; frees all llama.cpp resources.
func (y *YZMAModel) Destroy() error {
	y.mu.Lock()
	defer y.mu.Unlock()

	return errors.Join(
		llama.Free(y.lctx),
		llama.ModelFree(y.lmodel),
	)
}

// CreateMessagesYZMA stores the conversation messages (and optional system prompt) into the batch.
// Unlike ORT, YZMA handles chat-template formatting at generation time, so we just store the
// raw [][]Message for later use.
func CreateMessagesYZMA(batch *PipelineBatch, inputs any, systemPrompt string) error {
	switch inputCast := inputs.(type) {
	case []string:
		msgs := make([][]Message, len(inputCast))
		for i, s := range inputCast {
			if systemPrompt != "" {
				msgs[i] = []Message{
					{Role: "system", Content: systemPrompt},
					{Role: "user", Content: s},
				}
			} else {
				msgs[i] = []Message{{Role: "user", Content: s}}
			}
		}
		batch.InputValues = msgs
	case [][]Message:
		if systemPrompt == "" {
			batch.InputValues = inputCast
			return nil
		}
		msgs := make([][]Message, len(inputCast))
		sysmsg := Message{Role: "system", Content: systemPrompt}
		for i, conv := range inputCast {
			msgs[i] = append([]Message{sysmsg}, conv...)
		}
		batch.InputValues = msgs
	default:
		return fmt.Errorf("yzma: unsupported input type %T", inputs)
	}
	return nil
}

// runGenerativeYZMAOnBatch is the pipeline dispatch entry point for the YZMA backend.
func runGenerativeYZMAOnBatch(ctx context.Context, batch *PipelineBatch, p *BasePipeline, maxLength int, _ []string, temperature *float64, topP *float64, seed *int, _ []string, guidance *Guidance) (chan SequenceDelta, chan error, error) {
	if p.SessionContext == nil {
		return nil, nil, errors.New("yzma: no session context")
	}
	select {
	case <-ctx.Done():
		return nil, nil, ctx.Err()
	case <-p.SessionContext.Done():
		return nil, nil, p.SessionContext.Err()
	default:
	}

	yzmaModel := p.Model.YZMAModel
	if yzmaModel == nil {
		return nil, nil, errors.New("yzma: model is not initialized")
	}

	inputs, ok := batch.InputValues.([][]Message)
	if !ok {
		return nil, nil, fmt.Errorf("yzma: invalid input type %T, expected [][]Message", batch.InputValues)
	}

	genOpts := &GenerativeOptions{
		MaxLength:   maxLength,
		Temperature: temperature,
		TopP:        topP,
		Seed:        seed,
		Guidance:    guidance,
	}

	tokenStream, errorStream, err := yzmaModel.Generate(ctx, inputs, nil, genOpts)
	if err != nil {
		return nil, nil, err
	}

	// Wrap streams so that batch.Destroy() is called when generation completes.
	wrappedTokens := make(chan SequenceDelta, 64)
	wrappedErrors := make(chan error, 1)

	go func() {
		defer close(wrappedTokens)
		defer func() {
			if destroyErr := batch.Destroy(); destroyErr != nil {
				wrappedErrors <- destroyErr
			}
		}()
		for delta := range tokenStream {
			wrappedTokens <- delta
		}
		for err := range errorStream {
			wrappedErrors <- err
		}
	}()

	return wrappedTokens, wrappedErrors, nil
}
