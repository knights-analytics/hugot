//go:build YZMA || ALL

package backends

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"
	"unsafe"

	"github.com/hybridgroup/yzma/pkg/llama"
	"github.com/hybridgroup/yzma/pkg/mtmd"

	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/util/fileutil"
)

// yzmaLibOnce ensures llama.Load is called exactly once per process.
var (
	yzmaLibOnce sync.Once
	yzmaLibPath string
	errYzmaLib  error
	mtmdLibOnce sync.Once
	errMTMDLib  error
)

// YZMAModel holds all llama.cpp resources for a YZMA generative session.
type YZMAModel struct {
	mu            sync.Mutex
	lmodel        llama.Model
	lctx          llama.Context
	vocab         llama.Vocab
	chatTemplate  string
	projector     mtmd.Context
	projectorPath string

	// cumulative statistics
	statsMu                        sync.Mutex
	cumulativeTokens               int
	cumulativeTokenDurationSeconds float64
	cumulativePrefillSum           float64
	cumulativePrefillCount         int
}

// createYZMAGenerativeSession loads a GGUF model from model.Path and initialises a llama.cpp
// context, storing the result in model.YZMAModel.
func createYZMAGenerativeSession(ctx context.Context, model *Model, opts *options.Options) error {
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
		yzmaLibPath = libPath
		errYzmaLib = llama.Load(libPath)
	})
	if errYzmaLib != nil {
		return fmt.Errorf("yzma: failed to load llama.cpp library: %w", errYzmaLib)
	}
	if libPath != yzmaLibPath {
		return fmt.Errorf("yzma: llama.cpp library was already initialized from %q; cannot switch to %q", yzmaLibPath, libPath)
	}

	llama.Init()

	mParams := llama.ModelDefaultParams()
	mParams.NGpuLayers = yopts.NGpuLayers

	modelFile, projectorFile, err := findYZMAModelFiles(ctx, model.Path, yopts.ProjectorPath)
	if err != nil {
		return err
	}
	lmodel, err := llama.ModelLoadFromFile(modelFile, mParams)
	if err != nil {
		return fmt.Errorf("yzma: failed to load model from %q: %w", modelFile, err)
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
	var projector mtmd.Context
	if projectorFile != "" {
		mtmdLibOnce.Do(func() { errMTMDLib = mtmd.Load(libPath) })
		if errMTMDLib != nil {
			_ = llama.ModelFree(lmodel)
			return fmt.Errorf("yzma: failed to load llama.cpp mtmd library: %w", errMTMDLib)
		}
		projectorParams := mtmd.ContextParamsDefault()
		projectorParams.UseGPU = yopts.NGpuLayers > 0
		if yopts.NThreads > 0 {
			projectorParams.Threads = yopts.NThreads
		}
		projector, err = mtmd.InitFromFile(projectorFile, lmodel, projectorParams)
		if err != nil {
			_ = llama.ModelFree(lmodel)
			return fmt.Errorf("yzma: failed to initialize vision projector %q: %w", projectorFile, err)
		}
		if !mtmd.SupportVision(projector) {
			_ = mtmd.Free(projector)
			_ = llama.ModelFree(lmodel)
			return fmt.Errorf("yzma: projector %q does not support vision", projectorFile)
		}
	}

	model.YZMAModel = &YZMAModel{
		lmodel:        lmodel,
		lctx:          lctx,
		vocab:         vocab,
		chatTemplate:  chatTemplate,
		projector:     projector,
		projectorPath: projectorFile,
	}
	return nil
}

func findYZMAModelFiles(ctx context.Context, modelPath string, projectorOverride *string) (string, string, error) {
	searchRoot := modelPath
	var selectedModel string
	if strings.EqualFold(filepath.Ext(modelPath), ".gguf") {
		selectedModel = modelPath
		searchRoot = filepath.Dir(modelPath)
	}
	var models, projectors []string
	err := fileutil.WalkDir(ctx, searchRoot, func(_ context.Context, _ string, parent string, info os.FileInfo, _ io.Reader) (bool, error) {
		if !strings.EqualFold(filepath.Ext(info.Name()), ".gguf") {
			return true, nil
		}
		candidate := filepath.Join(parent, info.Name())
		if !filepath.IsAbs(candidate) {
			candidate = filepath.Join(searchRoot, candidate)
		}
		if strings.Contains(strings.ToLower(info.Name()), "mmproj") {
			projectors = append(projectors, candidate)
		} else if selectedModel == "" {
			models = append(models, candidate)
		}
		return true, nil
	})
	if err != nil {
		return "", "", err
	}
	if selectedModel == "" {
		switch len(models) {
		case 0:
			return "", "", fmt.Errorf("yzma: no GGUF model found at %q", modelPath)
		case 1:
			selectedModel = models[0]
		default:
			return "", "", fmt.Errorf("yzma: multiple model GGUF files found at %q; set ModelPath to the selected file", modelPath)
		}
	}
	if projectorOverride != nil {
		return selectedModel, *projectorOverride, nil
	}
	switch len(projectors) {
	case 0:
		return selectedModel, "", nil
	case 1:
		return selectedModel, projectors[0], nil
	default:
		return "", "", fmt.Errorf("yzma: multiple mmproj GGUF files found at %q; select one with WithYZMAProjectorPath", searchRoot)
	}
}

func findGGUFModel(ctx context.Context, modelPath string) (string, error) {
	model, _, err := findYZMAModelFiles(ctx, modelPath, nil)
	return model, err
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
		defer close(errorStream)

		for seqIdx, conversation := range inputs {
			select {
			case <-ctx.Done():
				select {
				case errorStream <- ctx.Err():
				case <-ctx.Done():
				}
				return
			default:
			}

			if err := y.generateOne(ctx, seqIdx, conversation, genOpts, tokenStream); err != nil {
				select {
				case errorStream <- err:
				case <-ctx.Done():
				}
				return
			}
		}
	}()

	return tokenStream, errorStream, nil
}

// generateOne runs the decode loop for a single conversation and sends tokens to tokenStream.
func (y *YZMAModel) generateOne(ctx context.Context, seqIdx int, conversation []Message, opts *GenerativeOptions, tokenStream chan<- SequenceDelta) error {
	if conversationHasMediaOrBlocks(conversation) {
		return y.generateVisionOne(ctx, seqIdx, conversation, opts, tokenStream)
	}
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
		tokenCount++

		select {
		case tokenStream <- SequenceDelta{Token: piece, Sequence: seqIdx}:
		case <-ctx.Done():
			return ctx.Err()
		}

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

func conversationHasMediaOrBlocks(conversation []Message) bool {
	for _, message := range conversation {
		if len(message.ImageURLs) != 0 || len(message.AudioURLs) != 0 || len(message.ContentBlocks) != 0 {
			return true
		}
	}
	return false
}

func buildYZMAVisionPrompt(conversation []Message) ([]llama.ChatMessage, []string, error) {
	chatMessages := make([]llama.ChatMessage, len(conversation))
	var imagePaths []string
	marker := mtmd.DefaultMarker()
	for i, message := range conversation {
		if len(message.AudioURLs) != 0 {
			return nil, nil, errors.New("yzma: audio input is not supported yet")
		}
		content := message.Content
		if len(message.ContentBlocks) != 0 {
			if message.Content != "" || len(message.ImageURLs) != 0 || len(message.AudioURLs) != 0 {
				return nil, nil, errors.New("yzma: content blocks cannot be combined with text or media attachments")
			}
			content = ""
			for _, block := range message.ContentBlocks {
				switch block.Type {
				case "text":
					if block.URL != "" {
						return nil, nil, errors.New("yzma: text content blocks cannot contain a URL")
					}
					content += block.Text
				case "image":
					if block.URL == "" || block.Text != "" {
						return nil, nil, errors.New("yzma: image content blocks require a URL and no text")
					}
					content += marker
					imagePaths = append(imagePaths, block.URL)
				case "audio":
					return nil, nil, errors.New("yzma: audio input is not supported yet")
				default:
					return nil, nil, fmt.Errorf("yzma: unsupported content block type %q", block.Type)
				}
			}
		} else {
			for _, path := range message.ImageURLs {
				if strings.TrimSpace(path) == "" {
					return nil, nil, errors.New("yzma: image path is empty")
				}
				content += "\n" + marker
				imagePaths = append(imagePaths, path)
			}
		}
		role := message.Role
		if role == "" {
			role = "user"
		}
		chatMessages[i] = llama.NewChatMessage(role, content)
	}
	return chatMessages, imagePaths, nil
}

func (y *YZMAModel) generateVisionOne(ctx context.Context, seqIdx int, conversation []Message, opts *GenerativeOptions, tokenStream chan<- SequenceDelta) error {
	if y.projector == 0 {
		return errors.New("yzma: image input requires an mmproj GGUF; provide one beside the model or use WithYZMAProjectorPath")
	}
	chatMessages, imagePaths, err := buildYZMAVisionPrompt(conversation)
	if err != nil {
		return err
	}
	if len(imagePaths) == 0 {
		return errors.New("yzma: vision prompt contains no images")
	}

	bitmaps := make([]mtmd.Bitmap, 0, len(imagePaths))
	bitmapWrappers := make([]mtmd.BitmapWrapper, 0, len(imagePaths))
	defer func() {
		for _, wrapper := range bitmapWrappers {
			mtmd.BitmapFree(wrapper.Bitmap)
			mtmd.VideoFree(wrapper.VideoContext)
		}
	}()
	for _, imagePath := range imagePaths {
		var data []byte
		if strings.HasPrefix(imagePath, "data:") {
			comma := strings.IndexByte(imagePath, ',')
			if comma < 0 || !strings.Contains(imagePath[:comma], ";base64") {
				return fmt.Errorf("yzma: image data URI must use base64 encoding")
			}
			data, err = base64.StdEncoding.DecodeString(imagePath[comma+1:])
		} else {
			data, err = fileutil.ReadFileBytes(ctx, imagePath)
		}
		if err != nil {
			return fmt.Errorf("yzma: reading image %q: %w", imagePath, err)
		}
		if len(data) == 0 {
			return fmt.Errorf("yzma: image %q is empty", imagePath)
		}
		buffer := unsafe.SliceData(data)
		wrapper := mtmd.BitmapInitFromBuf(y.projector, buffer, uint64(len(data)), false, mtmd.InitOptDefault())
		runtime.KeepAlive(data)
		if wrapper.Bitmap == 0 {
			return fmt.Errorf("yzma: decoding image %q failed", imagePath)
		}
		bitmapWrappers = append(bitmapWrappers, wrapper)
		bitmaps = append(bitmaps, wrapper.Bitmap)
	}
	y.mu.Lock()
	defer y.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return err
	}
	promptBuffer := make([]byte, 16384)
	promptLength := llama.ChatApplyTemplate(y.chatTemplate, chatMessages, true, promptBuffer)
	if promptLength < 0 {
		promptBuffer = make([]byte, -promptLength+1)
		promptLength = llama.ChatApplyTemplate(y.chatTemplate, chatMessages, true, promptBuffer)
	}
	if promptLength < 0 || int(promptLength) > len(promptBuffer) {
		return fmt.Errorf("yzma: applying vision chat template failed for sequence %d", seqIdx)
	}
	input := mtmd.NewInputText(string(promptBuffer[:promptLength]), true, true)
	chunks := mtmd.InputChunksInit()
	defer mtmd.InputChunksFree(chunks)
	if result := mtmd.Tokenize(y.projector, chunks, input, bitmaps); result != 0 {
		return fmt.Errorf("yzma: multimodal tokenization failed for sequence %d (code %d)", seqIdx, result)
	}
	if memory, memoryErr := llama.GetMemory(y.lctx); memoryErr == nil {
		_ = llama.MemoryClear(memory, false)
	}
	var nPast llama.Pos
	prefillStart := time.Now()
	if result := mtmd.HelperEvalChunks(y.projector, y.lctx, chunks, 0, 0, int32(llama.NBatch(y.lctx)), true, &nPast); result != 0 {
		return fmt.Errorf("yzma: multimodal prefill failed for sequence %d (code %d)", seqIdx, result)
	}
	prefillDuration := time.Since(prefillStart).Seconds()
	sampler := y.buildSampler(opts)
	defer llama.SamplerFree(sampler)
	maxNewTokens := opts.MaxLength
	if maxNewTokens <= 0 {
		maxNewTokens = 1028
	}
	genStart := time.Now()
	var tokenCount int
	pieceBuffer := make([]byte, 256)
	for range maxNewTokens {
		if err := ctx.Err(); err != nil {
			return err
		}
		token := llama.SamplerSample(sampler, y.lctx, -1)
		llama.SamplerAccept(sampler, token)
		if llama.VocabIsEOG(y.vocab, token) {
			break
		}
		pieceLength := llama.TokenToPiece(y.vocab, token, pieceBuffer, 0, true)
		if pieceLength < 0 {
			pieceBuffer = make([]byte, -pieceLength)
			pieceLength = llama.TokenToPiece(y.vocab, token, pieceBuffer, 0, true)
		}
		if pieceLength > 0 && int(pieceLength) <= len(pieceBuffer) {
			select {
			case tokenStream <- SequenceDelta{Token: string(pieceBuffer[:pieceLength]), Sequence: seqIdx}:
			case <-ctx.Done():
				return ctx.Err()
			}
			tokenCount++
		}
		batch := llama.BatchGetOne([]llama.Token{token})
		batch.Pos = &nPast
		if _, decodeErr := llama.Decode(y.lctx, batch); decodeErr != nil {
			return fmt.Errorf("yzma: multimodal decode error at token %d: %w", tokenCount, decodeErr)
		}
		nPast++
	}
	y.statsMu.Lock()
	y.cumulativeTokens += tokenCount
	y.cumulativeTokenDurationSeconds += time.Since(genStart).Seconds()
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

	var projectorErr error
	if y.projector != 0 {
		projectorErr = mtmd.Free(y.projector)
		y.projector = 0
	}
	return errors.Join(
		projectorErr,
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
func runGenerativeYZMAOnBatch(ctx context.Context, batch *PipelineBatch, p *BasePipeline, maxLength int, stopSequences []string, temperature *float64, topP *float64, seed *int, tools []string, guidance *Guidance) (chan SequenceDelta, chan error, error) {
	if len(stopSequences) != 0 {
		return nil, nil, errors.New("yzma: stop sequences are not supported yet")
	}
	if len(tools) != 0 {
		return nil, nil, errors.New("yzma: tool calling is not supported yet")
	}
	if guidance != nil && guidance.Type != GuidanceTypeLarkGrammar {
		return nil, nil, fmt.Errorf("yzma: guidance type %q is not supported", guidance.Type)
	}
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
	wrappedErrors := make(chan error, 2)

	go func() {
		defer close(wrappedTokens)
		defer func() {
			if destroyErr := batch.Destroy(); destroyErr != nil {
				wrappedErrors <- destroyErr
			}
			close(wrappedErrors)
		}()
		for delta := range tokenStream {
			select {
			case wrappedTokens <- delta:
			case <-ctx.Done():
				return
			}
		}
		for err := range errorStream {
			wrappedErrors <- err
		}
	}()

	return wrappedTokens, wrappedErrors, nil
}
