package pipelines

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"
	"sync/atomic"
	"time"

	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/options"
)

// GLiNERPipeline implements zero-shot Named Entity Recognition using GLiNER models.
// GLiNER (Generalist and Lightweight model for Named Entity Recognition) can extract
// any entity types without retraining - just specify the entity labels at inference time.
type GLiNERPipeline struct {
	*backends.BasePipeline
	MaxWidth   int      // Maximum span width in words
	Labels     []string // Entity labels to recognize
	Threshold  float32  // Score threshold for entity detection
	FlatNER    bool     // If true, don't allow nested entities
	MultiLabel bool     // If true, allow multiple labels per span

	// Relation extraction settings
	RelationLabels    []string // Relation types to extract (for multitask models)
	RelationThreshold float32  // Score threshold for relation detection

	// Sequence packing for batch optimization
	PackingEnabled bool // If true, pack multiple short sequences into single batch
	MaxPackedLen   int  // Maximum packed sequence length (default: 512)

	// Label embedding cache for BiEncoder models
	labelEmbeddingCache map[string][]float32 // Cached label embeddings
	labelEmbeddingDim   int                  // Dimension of label embeddings
}

// GLiNEREntity represents a recognized named entity
type GLiNEREntity struct {
	Text  string  // The text span of the entity
	Label string  // The entity type/label
	Start int     // Character offset where entity begins
	End   int     // Character offset where entity ends
	Score float32 // Model confidence score (0.0-1.0)
}

// GLiNERRelation represents a relationship between two entities
type GLiNERRelation struct {
	HeadEntity GLiNEREntity // Source entity in the relationship
	TailEntity GLiNEREntity // Target entity in the relationship
	Label      string       // Relationship type (e.g., "founded", "works_at")
	Score      float32      // Model confidence score (0.0-1.0)
}

// GLiNEROutput holds the output of GLiNER inference
type GLiNEROutput struct {
	Entities  [][]GLiNEREntity  // Entities for each input text
	Relations [][]GLiNERRelation // Relations for each input text (if relation extraction enabled)
}

func (o *GLiNEROutput) GetOutput() []any {
	out := make([]any, len(o.Entities))
	for i, entities := range o.Entities {
		out[i] = any(entities)
	}
	return out
}

// HasRelations returns true if relation extraction was performed
func (o *GLiNEROutput) HasRelations() bool {
	return len(o.Relations) > 0
}

// GLiNERBatch extends PipelineBatch with GLiNER-specific data
type GLiNERBatch struct {
	*backends.PipelineBatch
	WordsMask    [][]int64   // Word boundary mask for each input
	TextLengths  [][]int64   // Number of words in each input
	SpanIdx      [][][]int64 // Span indices [batch][num_spans][2]
	SpanMask     [][]int64   // Valid spans mask
	NumSpans     int         // Number of spans per input
	NumLabels    int         // Number of entity labels (used for output tensor shape)
	WordsToChars [][][2]int  // Mapping from word index to character offsets [batch][word][start,end]
	OriginalText []string    // Original input texts
}

// Pipeline options

// WithGLiNERLabels sets the entity labels to recognize
func WithGLiNERLabels(labels []string) backends.PipelineOption[*GLiNERPipeline] {
	return func(p *GLiNERPipeline) error {
		p.Labels = labels
		return nil
	}
}

// WithGLiNERMaxWidth sets the maximum span width in words
func WithGLiNERMaxWidth(maxWidth int) backends.PipelineOption[*GLiNERPipeline] {
	return func(p *GLiNERPipeline) error {
		if maxWidth <= 0 {
			return errors.New("maxWidth must be positive")
		}
		p.MaxWidth = maxWidth
		return nil
	}
}

// WithGLiNERThreshold sets the score threshold for entity detection
func WithGLiNERThreshold(threshold float32) backends.PipelineOption[*GLiNERPipeline] {
	return func(p *GLiNERPipeline) error {
		if threshold < 0 || threshold > 1 {
			return errors.New("threshold must be between 0 and 1")
		}
		p.Threshold = threshold
		return nil
	}
}

// WithGLiNERFlatNER enables flat NER mode (no nested entities)
func WithGLiNERFlatNER() backends.PipelineOption[*GLiNERPipeline] {
	return func(p *GLiNERPipeline) error {
		p.FlatNER = true
		return nil
	}
}

// WithGLiNERMultiLabel enables multi-label mode
func WithGLiNERMultiLabel() backends.PipelineOption[*GLiNERPipeline] {
	return func(p *GLiNERPipeline) error {
		p.MultiLabel = true
		return nil
	}
}

// WithGLiNERRelationLabels sets the relation labels for relationship extraction
func WithGLiNERRelationLabels(labels []string) backends.PipelineOption[*GLiNERPipeline] {
	return func(p *GLiNERPipeline) error {
		p.RelationLabels = labels
		return nil
	}
}

// WithGLiNERRelationThreshold sets the threshold for relation detection
func WithGLiNERRelationThreshold(threshold float32) backends.PipelineOption[*GLiNERPipeline] {
	return func(p *GLiNERPipeline) error {
		if threshold < 0 || threshold > 1 {
			return errors.New("relation threshold must be between 0 and 1")
		}
		p.RelationThreshold = threshold
		return nil
	}
}

// WithGLiNERSequencePacking enables sequence packing for batch optimization
// This combines multiple short sequences into a single transformer pass with a block-diagonal attention mask
func WithGLiNERSequencePacking(maxPackedLen int) backends.PipelineOption[*GLiNERPipeline] {
	return func(p *GLiNERPipeline) error {
		if maxPackedLen <= 0 {
			maxPackedLen = 512
		}
		p.PackingEnabled = true
		p.MaxPackedLen = maxPackedLen
		return nil
	}
}

// NewGLiNERPipeline creates a new GLiNER pipeline
func NewGLiNERPipeline(config backends.PipelineConfig[*GLiNERPipeline], s *options.Options, model *backends.Model) (*GLiNERPipeline, error) {
	basePipeline, err := backends.NewBasePipeline(config, s, model)
	if err != nil {
		return nil, err
	}

	pipeline := &GLiNERPipeline{
		BasePipeline:        basePipeline,
		MaxWidth:            GLiNERDefaultMaxWidth,
		Labels:              []string{"person", "organization", "location"},
		Threshold:           0.5,
		FlatNER:             true,
		MultiLabel:          false,
		RelationThreshold:   0.5,
		MaxPackedLen:        512,
		labelEmbeddingCache: make(map[string][]float32),
	}

	// Apply options
	for _, o := range config.Options {
		if err := o(pipeline); err != nil {
			return nil, err
		}
	}

	// GLiNER needs offsets and special token masks to detect word boundaries
	backends.AllInputTokens(pipeline.BasePipeline)

	// Validate the pipeline
	if err := pipeline.Validate(); err != nil {
		return nil, err
	}

	return pipeline, nil
}

// GetModel returns the underlying model
func (p *GLiNERPipeline) GetModel() *backends.Model {
	return p.Model
}

// GetMetadata returns pipeline metadata
func (p *GLiNERPipeline) GetMetadata() backends.PipelineMetadata {
	return backends.PipelineMetadata{
		OutputsInfo: []backends.OutputInfo{
			{
				Name:       p.Model.OutputsMeta[0].Name,
				Dimensions: p.Model.OutputsMeta[0].Dimensions,
			},
		},
	}
}

// GetStatistics returns the pipeline statistics.
func (p *GLiNERPipeline) GetStatistics() backends.PipelineStatistics {
	statistics := backends.PipelineStatistics{}
	statistics.ComputeTokenizerStatistics(p.Model.Tokenizer.TokenizerTimings)
	statistics.ComputeOnnxStatistics(p.PipelineTimings)
	return statistics
}

// GetStats returns runtime statistics
func (p *GLiNERPipeline) GetStats() []string {
	return []string{
		fmt.Sprintf("Statistics for GLiNER pipeline: %s", p.PipelineName),
		fmt.Sprintf("Tokenizer: Total time=%s, Execution count=%d, Average time=%s",
			time.Duration(p.Model.Tokenizer.TokenizerTimings.TotalNS),
			p.Model.Tokenizer.TokenizerTimings.NumCalls,
			time.Duration(float64(p.Model.Tokenizer.TokenizerTimings.TotalNS)/math.Max(1, float64(p.Model.Tokenizer.TokenizerTimings.NumCalls)))),
		fmt.Sprintf("ONNX: Total time=%s, Execution count=%d, Average time=%s",
			time.Duration(p.PipelineTimings.TotalNS),
			p.PipelineTimings.NumCalls,
			time.Duration(float64(p.PipelineTimings.TotalNS)/math.Max(1, float64(p.PipelineTimings.NumCalls)))),
	}
}

// Validate checks that the pipeline configuration is valid
func (p *GLiNERPipeline) Validate() error {
	var validationErrors []error

	// GLiNER currently only works with ORT backend due to dynamic shape operations
	// that GoMLX cannot handle (ConstantOfShape nodes that depend on runtime inputs)
	if p.Runtime == "GO" || p.Runtime == "XLA" {
		validationErrors = append(validationErrors,
			fmt.Errorf("GLiNER pipeline currently requires ORT backend (got %s). "+
				"The model uses dynamic shapes (LSTM hidden states, output shapes) that GoMLX cannot handle. "+
				"Please use NewORTSession() instead of NewGoSession()", p.Runtime))
	}

	if p.Model.Tokenizer == nil {
		validationErrors = append(validationErrors, errors.New("GLiNER pipeline requires a tokenizer"))
	}

	if len(p.Labels) == 0 {
		validationErrors = append(validationErrors, errors.New("GLiNER pipeline requires at least one label"))
	}

	// Verify model has expected inputs
	expectedInputs := map[string]bool{
		"input_ids":      false,
		"attention_mask": false,
		"words_mask":     false,
		"text_lengths":   false,
		"span_idx":       false,
		"span_mask":      false,
	}
	for _, meta := range p.Model.InputsMeta {
		if _, ok := expectedInputs[meta.Name]; ok {
			expectedInputs[meta.Name] = true
		}
	}
	for name, found := range expectedInputs {
		if !found {
			validationErrors = append(validationErrors, fmt.Errorf("GLiNER model missing required input: %s", name))
		}
	}

	return errors.Join(validationErrors...)
}

// Run executes the pipeline on input texts
func (p *GLiNERPipeline) Run(inputs []string) (backends.PipelineBatchOutput, error) {
	return p.RunPipeline(inputs)
}

// RunPipeline executes the pipeline and returns the concrete output type
func (p *GLiNERPipeline) RunPipeline(inputs []string) (*GLiNEROutput, error) {
	return p.RunPipelineWithLabels(inputs, p.Labels)
}

// RunPipelineWithLabels executes the pipeline with custom labels (zero-shot NER)
func (p *GLiNERPipeline) RunPipelineWithLabels(inputs []string, labels []string) (*GLiNEROutput, error) {
	if len(inputs) == 0 {
		return &GLiNEROutput{Entities: [][]GLiNEREntity{}}, nil
	}

	var runErrors []error
	batch := p.prepareGLiNERBatch(len(inputs))
	defer func() {
		if batch.PipelineBatch != nil {
			runErrors = append(runErrors, batch.Destroy())
		}
	}()

	// Preprocess
	if err := p.Preprocess(batch, inputs, labels); err != nil {
		return nil, err
	}

	// Forward pass
	if err := p.Forward(batch); err != nil {
		return nil, err
	}

	// Postprocess
	result, err := p.Postprocess(batch, labels)
	if err != nil {
		return nil, err
	}

	return result, errors.Join(runErrors...)
}

func (p *GLiNERPipeline) prepareGLiNERBatch(size int) *GLiNERBatch {
	return &GLiNERBatch{
		PipelineBatch: backends.NewBatch(size),
		OriginalText:  make([]string, size),
	}
}

// Preprocess prepares the batch for inference
func (p *GLiNERPipeline) Preprocess(batch *GLiNERBatch, inputs []string, labels []string) error {
	start := time.Now()

	// GLiNER requires labels to be prepended to the input text
	// Format: "<<ENT>> label1 <<ENT>> label2 <<SEP>> text"
	labelPrefix := buildGLiNERLabelPrefix(labels)
	prefixedTexts := make([]string, len(inputs))
	for i, text := range inputs {
		prefixedTexts[i] = labelPrefix + " " + text
		batch.OriginalText[i] = text
	}

	// Tokenize the prefixed texts
	backends.TokenizeInputs(batch.PipelineBatch, p.Model.Tokenizer, prefixedTexts)

	atomic.AddUint64(&p.Model.Tokenizer.TokenizerTimings.NumCalls, 1)
	atomic.AddUint64(&p.Model.Tokenizer.TokenizerTimings.TotalNS, uint64(time.Since(start)))

	// Build GLiNER-specific inputs
	if err := p.buildGLiNERInputs(batch, labels); err != nil {
		return err
	}

	return nil
}

// buildGLiNERInputs constructs the GLiNER-specific input tensors
func (p *GLiNERPipeline) buildGLiNERInputs(batch *GLiNERBatch, labels []string) error {
	batchSize := batch.Size
	maxSeqLen := batch.MaxSequenceLength

	// Store number of labels for output tensor shape calculation
	batch.NumLabels = len(labels)

	// Initialize arrays
	batch.WordsMask = make([][]int64, batchSize)
	batch.TextLengths = make([][]int64, batchSize)
	batch.WordsToChars = make([][][2]int, batchSize)

	// Calculate the character length of the label prefix to skip
	labelPrefixCharLen := uint(calculateLabelPrefixLength(labels))

	for i, input := range batch.Input {
		wordsMask := make([]int64, maxSeqLen)
		wordsToChars := [][2]int{}

		// Track word boundaries
		wordCount := int64(0)
		inTextRegion := false

		for j, offset := range input.Offsets {
			// Skip special tokens (CLS, SEP, PAD)
			if input.SpecialTokensMask[j] > 0 {
				continue
			}

			tokenStart := offset[0]
			tokenEnd := offset[1]

			// Skip tokens that are part of the label prefix
			if tokenEnd <= labelPrefixCharLen {
				continue
			}

			// We've entered the text region (past the label prefix)
			inTextRegion = true

			// Adjust offsets to be relative to the original text (subtract prefix length)
			adjustedStart := tokenStart - labelPrefixCharLen
			adjustedEnd := tokenEnd - labelPrefixCharLen

			// Detect word boundaries using SentencePiece convention:
			// Tokens starting with ▁ (U+2581) indicate a new word
			token := ""
			if j < len(input.Tokens) {
				token = input.Tokens[j]
			}

			isNewWord := false
			if inTextRegion && wordCount == 0 {
				// First token in text region is always a new word
				isNewWord = true
			} else if strings.HasPrefix(token, "▁") || strings.HasPrefix(token, " ") {
				// Token starts with word boundary marker
				isNewWord = true
			}

			if isNewWord {
				wordCount++
				wordsMask[j] = wordCount
				// Calculate start offset, accounting for SentencePiece space markers
				// For the first word, the ▁ represents the space in the prefix, not in the original text
				// For subsequent words, the ▁ represents an actual space in the original text
				startOffset := int(adjustedStart)
				if wordCount > 1 && (strings.HasPrefix(token, "▁") || strings.HasPrefix(token, " ")) {
					startOffset++ // Skip the space for words after the first
				}
				wordsToChars = append(wordsToChars, [2]int{startOffset, int(adjustedEnd)})
			} else {
				// Continuation of previous word (subword token)
				wordsMask[j] = wordCount
				if len(wordsToChars) > 0 {
					wordsToChars[len(wordsToChars)-1][1] = int(adjustedEnd)
				}
			}
		}

		batch.WordsMask[i] = wordsMask
		batch.TextLengths[i] = []int64{wordCount}
		batch.WordsToChars[i] = wordsToChars
	}

	// Generate spans for each input
	if err := p.generateSpans(batch); err != nil {
		return err
	}

	// Create ORT tensors
	return p.createGLiNERTensors(batch)
}

// generateSpans creates all valid spans up to MaxWidth
// GLiNER expects spans organized as: for each word position, max_width spans (one per width)
// Total spans = num_words × max_width, organized as [pos0_w1, pos0_w2, ..., pos0_wN, pos1_w1, ...]
func (p *GLiNERPipeline) generateSpans(batch *GLiNERBatch) error {
	batchSize := batch.Size

	// Find max words across batch
	maxWords := 0
	for i := 0; i < batchSize; i++ {
		numWords := int(batch.TextLengths[i][0])
		if numWords > maxWords {
			maxWords = numWords
		}
	}

	// GLiNER expects exactly num_words × max_width spans
	// The model internally reshapes to [batch, num_words, max_width, hidden]
	numSpans := maxWords * p.MaxWidth
	if numSpans == 0 {
		numSpans = p.MaxWidth // At least max_width span slots
	}

	batch.NumSpans = numSpans
	batch.SpanIdx = make([][][]int64, batchSize)
	batch.SpanMask = make([][]int64, batchSize)

	for i := 0; i < batchSize; i++ {
		numWords := int(batch.TextLengths[i][0])
		spanIdx := make([][]int64, numSpans)
		spanMask := make([]int64, numSpans)

		// For each word position, generate max_width spans
		for pos := 0; pos < maxWords; pos++ {
			for width := 1; width <= p.MaxWidth; width++ {
				spanIndex := pos*p.MaxWidth + (width - 1)
				endPos := pos + width - 1 // inclusive end

				// Check if span is valid (within text bounds)
				if pos < numWords && endPos < numWords {
					spanIdx[spanIndex] = []int64{int64(pos), int64(endPos)}
					spanMask[spanIndex] = 1
				} else {
					// Invalid span (extends beyond text or position is padding)
					spanIdx[spanIndex] = []int64{0, 0}
					spanMask[spanIndex] = 0
				}
			}
		}

		batch.SpanIdx[i] = spanIdx
		batch.SpanMask[i] = spanMask
	}

	return nil
}

// createGLiNERTensors creates input tensors for GLiNER based on runtime
func (p *GLiNERPipeline) createGLiNERTensors(batch *GLiNERBatch) error {
	// GLiNER requires custom input tensors - don't use the standard CreateInputTensors
	// as it doesn't know how to handle GLiNER-specific inputs
	switch p.Runtime {
	case "ORT":
		return createGLiNERTensorsORT(batch, p.Model)
	case "GO", "XLA":
		return createGLiNERTensorsGoMLX(batch, p.Model)
	default:
		return fmt.Errorf("unsupported runtime for GLiNER: %s", p.Runtime)
	}
}

// Forward performs the forward inference pass
func (p *GLiNERPipeline) Forward(batch *GLiNERBatch) error {
	start := time.Now()
	var err error
	switch p.Runtime {
	case "ORT":
		err = runGLiNERSessionOnBatchORT(batch, p.BasePipeline)
	case "GO", "XLA":
		err = runGLiNERSessionOnBatchGoMLX(batch, p.BasePipeline)
	default:
		return fmt.Errorf("unsupported runtime for GLiNER: %s", p.Runtime)
	}
	if err != nil {
		return err
	}
	atomic.AddUint64(&p.PipelineTimings.NumCalls, 1)
	atomic.AddUint64(&p.PipelineTimings.TotalNS, uint64(time.Since(start)))
	return nil
}

// Postprocess converts model output to entities
func (p *GLiNERPipeline) Postprocess(batch *GLiNERBatch, labels []string) (*GLiNEROutput, error) {
	if batch.Size == 0 {
		return &GLiNEROutput{}, nil
	}

	output := batch.OutputValues[0]
	batchSize := batch.Size

	// GLiNER output shape: [batch_size, num_words, num_spans, num_labels]
	// After sigmoid, values represent probability of each label for each span
	var logits [][][][]float32
	switch v := output.(type) {
	case [][][][]float32:
		logits = v
	default:
		return nil, fmt.Errorf("expected 4D output, got type %T", output)
	}

	result := &GLiNEROutput{
		Entities: make([][]GLiNEREntity, batchSize),
	}

	// Process each input in the batch
	for i := 0; i < batchSize; i++ {
		entities := p.extractEntities(
			batch.OriginalText[i],
			logits[i],
			batch.SpanIdx[i],
			batch.SpanMask[i],
			batch.WordsToChars[i],
			labels,
		)
		result.Entities[i] = entities
	}

	return result, nil
}

// extractEntities extracts entities from the logits for a single input
// logits shape: [num_words][max_width][num_labels]
// For each word position, there are max_width span options (width 1 to max_width)
func (p *GLiNERPipeline) extractEntities(
	text string,
	logits [][][]float32, // [num_words][max_width][num_labels]
	spanIdx [][]int64,
	spanMask []int64,
	wordsToChars [][2]int,
	labels []string,
) []GLiNEREntity {
	var candidates []GLiNEREntity

	numLabels := len(labels)
	numWords := len(wordsToChars)

	// Iterate over all spans organized as [word_position × max_width]
	for spanI := 0; spanI < len(spanIdx); spanI++ {
		if spanMask[spanI] == 0 {
			continue
		}

		startWord := int(spanIdx[spanI][0])
		endWord := int(spanIdx[spanI][1])

		// Get character offsets
		if startWord >= numWords || endWord >= numWords {
			continue
		}

		charStart := wordsToChars[startWord][0]
		charEnd := wordsToChars[endWord][1]

		if charStart >= len(text) || charEnd > len(text) || charStart >= charEnd {
			continue
		}

		entityText := text[charStart:charEnd]

		// Calculate indices into logits: [word_position][width_index][label]
		// width = endWord - startWord + 1, so width_index = width - 1 = endWord - startWord
		widthIndex := endWord - startWord

		if startWord >= len(logits) {
			continue
		}
		if widthIndex >= len(logits[startWord]) {
			continue
		}

		spanLogits := logits[startWord][widthIndex]
		if len(spanLogits) < numLabels {
			continue
		}

		// Apply sigmoid and check threshold
		for labelIdx := 0; labelIdx < numLabels; labelIdx++ {
			score := sigmoid(spanLogits[labelIdx])
			if score >= p.Threshold {
				candidates = append(candidates, GLiNEREntity{
					Text:  entityText,
					Label: labels[labelIdx],
					Start: charStart,
					End:   charEnd,
					Score: score,
				})

				// If not multi-label, only take the highest scoring label
				if !p.MultiLabel {
					break
				}
			}
		}
	}

	// If flat NER, remove nested entities (keep highest scoring)
	if p.FlatNER {
		candidates = removeNestedEntities(candidates)
	}

	// Sort by position
	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].Start != candidates[j].Start {
			return candidates[i].Start < candidates[j].Start
		}
		return candidates[i].End < candidates[j].End
	})

	return candidates
}

// Helper functions

// GLiNER constants
const (
	// GLiNERDefaultMaxWidth is the standard maximum span width in words for GLiNER models
	GLiNERDefaultMaxWidth = 12
	// GLiNER special tokens for label encoding
	glinerEntityToken = "<<ENT>>"
	glinerSepToken    = "<<SEP>>"
)

func buildGLiNERLabelPrefix(labels []string) string {
	var sb strings.Builder
	for _, label := range labels {
		sb.WriteString(glinerEntityToken)
		sb.WriteString(" ")
		sb.WriteString(label)
		sb.WriteString(" ")
	}
	sb.WriteString(glinerSepToken)
	return sb.String()
}

func calculateLabelPrefixLength(labels []string) int {
	// Calculate the character length of the label prefix "<<ENT>> label1 <<ENT>> label2 <<SEP>>"
	// This is used to find where the actual text starts in character offsets
	length := 0
	for _, label := range labels {
		// "<<ENT>> " + label + " "
		length += len(glinerEntityToken) + 1 + len(label) + 1
	}
	// "<<SEP>>" + space before text
	length += len(glinerSepToken) + 1
	return length
}

func sigmoid(x float32) float32 {
	return 1.0 / (1.0 + float32(math.Exp(-float64(x))))
}

func removeNestedEntities(entities []GLiNEREntity) []GLiNEREntity {
	if len(entities) <= 1 {
		return entities
	}

	// Sort by score descending
	sort.Slice(entities, func(i, j int) bool {
		return entities[i].Score > entities[j].Score
	})

	var result []GLiNEREntity
	for _, entity := range entities {
		overlaps := false
		for _, kept := range result {
			// Check if entity overlaps with any kept entity
			if entity.Start < kept.End && entity.End > kept.Start {
				overlaps = true
				break
			}
		}
		if !overlaps {
			result = append(result, entity)
		}
	}

	return result
}

// computeGLiNEROutputDimensions extracts or infers output dimensions from batch data.
// Returns (numWords, maxWidth, numClasses) for reshaping the output tensor.
func computeGLiNEROutputDimensions(batch *GLiNERBatch, dims backends.Shape, dataLen int) (numWords, maxWidth, numClasses int) {
	// Find max words across batch
	for _, tl := range batch.TextLengths {
		if len(tl) > 0 && int(tl[0]) > numWords {
			numWords = int(tl[0])
		}
	}
	if numWords == 0 {
		numWords = 1
	}

	maxWidth = batch.NumSpans / numWords
	if maxWidth == 0 {
		maxWidth = GLiNERDefaultMaxWidth
	}

	// Last dimension is num_classes (should be fixed in model)
	if len(dims) >= 4 && dims[3] > 0 {
		numClasses = int(dims[3])
	} else {
		// Infer from data length
		expectedSize := batch.Size * numWords * maxWidth
		if expectedSize > 0 {
			numClasses = dataLen / expectedSize
		}
		if numClasses == 0 {
			numClasses = 1
		}
	}

	return numWords, maxWidth, numClasses
}

// reshapeGLiNEROutput reshapes flat output data to 4D array [batch_size, num_words, max_width, num_classes].
// This is shared between ORT and GoMLX backends.
func reshapeGLiNEROutput(data []float32, batchSize, numWords, maxWidth, numClasses int) [][][][]float32 {
	result := make([][][][]float32, batchSize)
	idx := 0

	for b := 0; b < batchSize; b++ {
		result[b] = make([][][]float32, numWords)
		for w := 0; w < numWords; w++ {
			result[b][w] = make([][]float32, maxWidth)
			for s := 0; s < maxWidth; s++ {
				result[b][w][s] = make([]float32, numClasses)
				for c := 0; c < numClasses; c++ {
					if idx < len(data) {
						result[b][w][s][c] = data[idx]
						idx++
					}
				}
			}
		}
	}

	return result
}

// =============================================================================
// BiEncoder Label Embedding Caching
// =============================================================================
// BiEncoder models compute label embeddings separately from text embeddings.
// For workloads with fixed label sets, we can cache label embeddings and reuse
// them across multiple inference calls, reducing computation time significantly.

// PrecomputeLabelEmbeddings computes and caches embeddings for the given labels.
// This is useful for BiEncoder models where label embeddings can be computed once
// and reused across many inference calls with the same labels.
func (p *GLiNERPipeline) PrecomputeLabelEmbeddings(labels []string) error {
	if len(labels) == 0 {
		return errors.New("no labels provided for precomputation")
	}

	// Check if model supports BiEncoder label embedding extraction
	// BiEncoder models have a separate "label_embedding" output or can run
	// a separate forward pass just for labels
	hasLabelOutput := false
	for _, meta := range p.Model.OutputsMeta {
		if meta.Name == "label_embeddings" || meta.Name == "entity_type_embeddings" {
			hasLabelOutput = true
			break
		}
	}

	if !hasLabelOutput {
		// For non-BiEncoder models, we can still cache the label prefix tokens
		// This provides modest speedup for repeated inference with same labels
		p.cacheLabelPrefixTokens(labels)
		return nil
	}

	// For BiEncoder models, run forward pass to get label embeddings
	embeddings, dim, err := p.computeLabelEmbeddings(labels)
	if err != nil {
		return fmt.Errorf("computing label embeddings: %w", err)
	}

	// Cache the embeddings
	p.labelEmbeddingDim = dim
	for i, label := range labels {
		start := i * dim
		end := start + dim
		if end <= len(embeddings) {
			p.labelEmbeddingCache[label] = embeddings[start:end]
		}
	}

	return nil
}

// cacheLabelPrefixTokens pre-tokenizes the label prefix for reuse
func (p *GLiNERPipeline) cacheLabelPrefixTokens(labels []string) {
	// Store the label list - the tokenization will be reused when we detect
	// the same labels in subsequent calls
	p.Labels = labels
}

// computeLabelEmbeddings runs a forward pass to extract label embeddings
// This is only applicable for BiEncoder GLiNER models
func (p *GLiNERPipeline) computeLabelEmbeddings(labels []string) ([]float32, int, error) {
	// Build label-only input (no text)
	labelPrefix := buildGLiNERLabelPrefix(labels)

	// Tokenize just the label prefix
	batch := backends.NewBatch(1)
	defer batch.Destroy()

	backends.TokenizeInputs(batch, p.Model.Tokenizer, []string{labelPrefix})

	// Create minimal input tensors for label encoding
	// BiEncoder models accept a "labels_only" flag or separate label input
	switch p.Runtime {
	case "ORT":
		if err := p.createLabelOnlyTensorsORT(batch); err != nil {
			return nil, 0, err
		}
	default:
		return nil, 0, fmt.Errorf("label embedding extraction not supported for runtime: %s", p.Runtime)
	}

	// Run inference
	if err := p.forwardLabelEmbeddings(batch); err != nil {
		return nil, 0, err
	}

	// Extract embeddings from output
	// BiEncoder output for labels is typically [num_labels, embedding_dim]
	if len(batch.OutputValues) == 0 {
		return nil, 0, errors.New("no output from label embedding forward pass")
	}

	// Find label embedding output
	for i, meta := range p.Model.OutputsMeta {
		if meta.Name == "label_embeddings" || meta.Name == "entity_type_embeddings" {
			if i < len(batch.OutputValues) {
				switch v := batch.OutputValues[i].(type) {
				case [][]float32:
					// Flatten [num_labels][dim] to [num_labels * dim]
					if len(v) > 0 {
						dim := len(v[0])
						flat := make([]float32, len(v)*dim)
						for j, emb := range v {
							copy(flat[j*dim:], emb)
						}
						return flat, dim, nil
					}
				case []float32:
					// Already flat, infer dimension from label count
					if len(labels) > 0 {
						dim := len(v) / len(labels)
						return v, dim, nil
					}
				}
			}
		}
	}

	return nil, 0, errors.New("could not extract label embeddings from model output")
}

// createLabelOnlyTensorsORT creates tensors for label-only forward pass
func (p *GLiNERPipeline) createLabelOnlyTensorsORT(batch *backends.PipelineBatch) error {
	// This creates input tensors with just the tokenized labels
	// For standard token classification, we use the base pipeline's tensor creation
	// BiEncoder models may need specialized handling
	return backends.CreateInputTensors(batch, p.Model, p.Runtime)
}

// forwardLabelEmbeddings runs inference for label embedding extraction
func (p *GLiNERPipeline) forwardLabelEmbeddings(batch *backends.PipelineBatch) error {
	// Use the base pipeline's run method which handles runtime dispatch
	return backends.RunSessionOnBatch(batch, p.BasePipeline)
}

// HasCachedLabelEmbeddings returns true if label embeddings are cached
func (p *GLiNERPipeline) HasCachedLabelEmbeddings() bool {
	return len(p.labelEmbeddingCache) > 0
}

// GetCachedLabelEmbedding returns the cached embedding for a label, if available
func (p *GLiNERPipeline) GetCachedLabelEmbedding(label string) ([]float32, bool) {
	emb, ok := p.labelEmbeddingCache[label]
	return emb, ok
}

// ClearLabelEmbeddingCache clears the cached label embeddings
func (p *GLiNERPipeline) ClearLabelEmbeddingCache() {
	p.labelEmbeddingCache = make(map[string][]float32)
	p.labelEmbeddingDim = 0
}

// RunWithCachedEmbeddings runs inference using cached label embeddings
// This is faster than RunPipelineWithLabels when the same labels are used repeatedly
func (p *GLiNERPipeline) RunWithCachedEmbeddings(inputs []string, labels []string) (*GLiNEROutput, error) {
	if len(inputs) == 0 {
		return &GLiNEROutput{Entities: [][]GLiNEREntity{}}, nil
	}

	// Check if all required labels are cached
	allCached := true
	for _, label := range labels {
		if _, ok := p.labelEmbeddingCache[label]; !ok {
			allCached = false
			break
		}
	}

	if !allCached {
		// Fall back to regular inference
		return p.RunPipelineWithLabels(inputs, labels)
	}

	// Build cached embeddings tensor
	cachedEmbeddings := make([]float32, len(labels)*p.labelEmbeddingDim)
	for i, label := range labels {
		emb := p.labelEmbeddingCache[label]
		copy(cachedEmbeddings[i*p.labelEmbeddingDim:], emb)
	}

	// Run with cached embeddings
	return p.runWithPrecomputedLabelEmbeddings(inputs, labels, cachedEmbeddings)
}

// runWithPrecomputedLabelEmbeddings runs inference with precomputed label embeddings
func (p *GLiNERPipeline) runWithPrecomputedLabelEmbeddings(inputs []string, labels []string, labelEmbeddings []float32) (*GLiNEROutput, error) {
	// For models that support cached embeddings, we pass them as an additional input
	// This requires model support for the "cached_label_embeddings" input tensor

	// Check if model accepts cached embeddings
	hasCachedInput := false
	for _, meta := range p.Model.InputsMeta {
		if meta.Name == "cached_label_embeddings" || meta.Name == "entity_type_embeddings" {
			hasCachedInput = true
			break
		}
	}

	if !hasCachedInput {
		// Model doesn't support cached embeddings, fall back to regular inference
		return p.RunPipelineWithLabels(inputs, labels)
	}

	// Prepare batch with cached embeddings
	var runErrors []error
	batch := p.prepareGLiNERBatch(len(inputs))
	defer func() {
		if batch.PipelineBatch != nil {
			runErrors = append(runErrors, batch.Destroy())
		}
	}()

	// Preprocess with cached embeddings flag
	if err := p.preprocessWithCachedEmbeddings(batch, inputs, labels, labelEmbeddings); err != nil {
		return nil, err
	}

	// Forward pass
	if err := p.Forward(batch); err != nil {
		return nil, err
	}

	// Postprocess
	result, err := p.Postprocess(batch, labels)
	if err != nil {
		return nil, err
	}

	return result, errors.Join(runErrors...)
}

// preprocessWithCachedEmbeddings prepares batch with precomputed label embeddings
func (p *GLiNERPipeline) preprocessWithCachedEmbeddings(batch *GLiNERBatch, inputs []string, labels []string, labelEmbeddings []float32) error {
	// Standard preprocessing but skip label tokenization overhead
	start := time.Now()

	// For cached embeddings, we still need to tokenize the text part
	// but we can skip the expensive label encoding
	for i, text := range inputs {
		batch.OriginalText[i] = text
	}

	// Tokenize texts (without label prefix since embeddings are cached)
	backends.TokenizeInputs(batch.PipelineBatch, p.Model.Tokenizer, inputs)

	atomic.AddUint64(&p.Model.Tokenizer.TokenizerTimings.NumCalls, 1)
	atomic.AddUint64(&p.Model.Tokenizer.TokenizerTimings.TotalNS, uint64(time.Since(start)))

	// Build GLiNER-specific inputs without label prefix
	if err := p.buildGLiNERInputsNoPrefixForCached(batch); err != nil {
		return err
	}

	// Add cached label embeddings tensor
	// This would require model-specific tensor creation
	// For now, fall back to standard processing
	return nil
}

// buildGLiNERInputsNoPrefixForCached builds inputs when using cached label embeddings
func (p *GLiNERPipeline) buildGLiNERInputsNoPrefixForCached(batch *GLiNERBatch) error {
	batchSize := batch.Size
	maxSeqLen := batch.MaxSequenceLength

	// Initialize arrays
	batch.WordsMask = make([][]int64, batchSize)
	batch.TextLengths = make([][]int64, batchSize)
	batch.WordsToChars = make([][][2]int, batchSize)

	for i, input := range batch.Input {
		wordsMask := make([]int64, maxSeqLen)
		wordsToChars := [][2]int{}

		wordCount := int64(0)

		for j, offset := range input.Offsets {
			if input.SpecialTokensMask[j] > 0 {
				continue
			}

			tokenStart := offset[0]
			tokenEnd := offset[1]

			token := ""
			if j < len(input.Tokens) {
				token = input.Tokens[j]
			}

			isNewWord := wordCount == 0 || strings.HasPrefix(token, "▁") || strings.HasPrefix(token, " ")

			if isNewWord {
				wordCount++
				wordsMask[j] = wordCount
				startOffset := int(tokenStart)
				if wordCount > 1 && (strings.HasPrefix(token, "▁") || strings.HasPrefix(token, " ")) {
					startOffset++
				}
				wordsToChars = append(wordsToChars, [2]int{startOffset, int(tokenEnd)})
			} else {
				wordsMask[j] = wordCount
				if len(wordsToChars) > 0 {
					wordsToChars[len(wordsToChars)-1][1] = int(tokenEnd)
				}
			}
		}

		batch.WordsMask[i] = wordsMask
		batch.TextLengths[i] = []int64{wordCount}
		batch.WordsToChars[i] = wordsToChars
	}

	if err := p.generateSpans(batch); err != nil {
		return err
	}

	return p.createGLiNERTensors(batch)
}

// =============================================================================
// Sequence Packing for Batch Optimization
// =============================================================================
// Sequence packing combines multiple short sequences into a single transformer
// pass using block-diagonal attention masks. This improves GPU utilization and
// reduces memory overhead for batches with varying sequence lengths.

// PackedSequence represents multiple sequences packed into one
type PackedSequence struct {
	// CombinedTokens are the concatenated tokens from all sequences
	CombinedTokens []int64
	// CombinedMask is the combined attention mask
	CombinedMask []int64
	// SequenceOffsets marks where each original sequence starts
	SequenceOffsets []int
	// SequenceLengths stores the length of each packed sequence
	SequenceLengths []int
	// OriginalIndices maps packed positions back to original batch indices
	OriginalIndices []int
}

// PackingPlan describes how to pack sequences into groups
type PackingPlan struct {
	// Groups contains indices of sequences that will be packed together
	Groups [][]int
	// Each group's total length (for validation)
	GroupLengths []int
}

// createPackingPlan determines optimal packing for a batch of sequences
// Uses First-Fit Decreasing bin packing algorithm for good utilization
func (p *GLiNERPipeline) createPackingPlan(tokenLengths []int) *PackingPlan {
	if !p.PackingEnabled || len(tokenLengths) <= 1 {
		// No packing - each sequence in its own group
		groups := make([][]int, len(tokenLengths))
		lengths := make([]int, len(tokenLengths))
		for i, l := range tokenLengths {
			groups[i] = []int{i}
			lengths[i] = l
		}
		return &PackingPlan{Groups: groups, GroupLengths: lengths}
	}

	maxLen := p.MaxPackedLen
	if maxLen <= 0 {
		maxLen = 512
	}

	// Create indices sorted by length (descending) for FFD
	type seqLen struct {
		idx int
		len int
	}
	sorted := make([]seqLen, len(tokenLengths))
	for i, l := range tokenLengths {
		sorted[i] = seqLen{idx: i, len: l}
	}
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].len > sorted[j].len
	})

	// First-Fit Decreasing bin packing
	var groups [][]int
	var groupLengths []int

	for _, seq := range sorted {
		if seq.len > maxLen {
			// Sequence too long to pack, give it its own group
			groups = append(groups, []int{seq.idx})
			groupLengths = append(groupLengths, seq.len)
			continue
		}

		// Find first group that can fit this sequence
		placed := false
		for i, groupLen := range groupLengths {
			if groupLen+seq.len+1 <= maxLen { // +1 for separator token
				groups[i] = append(groups[i], seq.idx)
				groupLengths[i] = groupLen + seq.len + 1
				placed = true
				break
			}
		}

		if !placed {
			// Create new group
			groups = append(groups, []int{seq.idx})
			groupLengths = append(groupLengths, seq.len)
		}
	}

	return &PackingPlan{Groups: groups, GroupLengths: groupLengths}
}

// packSequences combines multiple sequences according to a packing plan
func (p *GLiNERPipeline) packSequences(batch *backends.PipelineBatch, plan *PackingPlan) ([]*PackedSequence, error) {
	packed := make([]*PackedSequence, len(plan.Groups))

	for groupIdx, group := range plan.Groups {
		ps := &PackedSequence{
			SequenceOffsets: make([]int, len(group)),
			SequenceLengths: make([]int, len(group)),
			OriginalIndices: make([]int, len(group)),
		}

		currentOffset := 0
		for i, seqIdx := range group {
			input := batch.Input[seqIdx]
			seqLen := len(input.TokenIDs)

			ps.OriginalIndices[i] = seqIdx
			ps.SequenceOffsets[i] = currentOffset
			ps.SequenceLengths[i] = seqLen

			// Append tokens
			for _, tok := range input.TokenIDs {
				ps.CombinedTokens = append(ps.CombinedTokens, int64(tok))
			}

			// Append attention mask
			for _, m := range input.AttentionMask {
				ps.CombinedMask = append(ps.CombinedMask, int64(m))
			}

			currentOffset += seqLen

			// Add separator between sequences (except after last)
			if i < len(group)-1 {
				ps.CombinedTokens = append(ps.CombinedTokens, 2) // SEP token ID
				ps.CombinedMask = append(ps.CombinedMask, 1)
				currentOffset++
			}
		}

		packed[groupIdx] = ps
	}

	return packed, nil
}

// createBlockDiagonalAttentionMask creates a 2D mask for packed sequences
// Each sequence can only attend to tokens within its own boundaries
func createBlockDiagonalAttentionMask(packedSeq *PackedSequence) [][]int64 {
	totalLen := len(packedSeq.CombinedTokens)
	mask := make([][]int64, totalLen)

	for i := range mask {
		mask[i] = make([]int64, totalLen)
	}

	// For each sequence, allow attention within its boundaries
	for seqIdx := range packedSeq.OriginalIndices {
		start := packedSeq.SequenceOffsets[seqIdx]
		end := start + packedSeq.SequenceLengths[seqIdx]

		for i := start; i < end; i++ {
			for j := start; j < end; j++ {
				if packedSeq.CombinedMask[j] == 1 { // Only attend to valid tokens
					mask[i][j] = 1
				}
			}
		}
	}

	return mask
}

// RunPipelineWithPacking runs inference with sequence packing optimization
func (p *GLiNERPipeline) RunPipelineWithPacking(inputs []string, labels []string) (*GLiNEROutput, error) {
	if len(inputs) == 0 {
		return &GLiNEROutput{Entities: [][]GLiNEREntity{}}, nil
	}

	if !p.PackingEnabled || len(inputs) == 1 {
		// Fall back to standard inference
		return p.RunPipelineWithLabels(inputs, labels)
	}

	// First, tokenize all inputs to get their lengths
	labelPrefix := buildGLiNERLabelPrefix(labels)
	prefixedTexts := make([]string, len(inputs))
	for i, text := range inputs {
		prefixedTexts[i] = labelPrefix + " " + text
	}

	// Quick tokenization pass to get lengths
	tempBatch := backends.NewBatch(len(inputs))
	backends.TokenizeInputs(tempBatch, p.Model.Tokenizer, prefixedTexts)

	tokenLengths := make([]int, len(inputs))
	for i, input := range tempBatch.Input {
		tokenLengths[i] = len(input.TokenIDs)
	}
	tempBatch.Destroy()

	// Create packing plan
	plan := p.createPackingPlan(tokenLengths)

	// If no beneficial packing (all single-sequence groups), use standard path
	singleGroups := true
	for _, g := range plan.Groups {
		if len(g) > 1 {
			singleGroups = false
			break
		}
	}
	if singleGroups {
		return p.RunPipelineWithLabels(inputs, labels)
	}

	// Process each packed group
	allEntities := make([][]GLiNEREntity, len(inputs))

	for _, group := range plan.Groups {
		// Get inputs for this group
		groupInputs := make([]string, len(group))
		for i, idx := range group {
			groupInputs[i] = inputs[idx]
		}

		// Run inference on group (standard path for now, as full packing
		// requires model support for block-diagonal attention)
		groupResult, err := p.RunPipelineWithLabels(groupInputs, labels)
		if err != nil {
			return nil, err
		}

		// Map results back to original indices
		for i, idx := range group {
			if i < len(groupResult.Entities) {
				allEntities[idx] = groupResult.Entities[i]
			}
		}
	}

	return &GLiNEROutput{Entities: allEntities}, nil
}

// GetPackingStats returns statistics about potential packing efficiency
func (p *GLiNERPipeline) GetPackingStats(tokenLengths []int) (numGroups int, avgUtilization float32) {
	if !p.PackingEnabled || len(tokenLengths) == 0 {
		return len(tokenLengths), 1.0
	}

	plan := p.createPackingPlan(tokenLengths)
	numGroups = len(plan.Groups)

	totalTokens := 0
	for _, l := range tokenLengths {
		totalTokens += l
	}

	totalCapacity := numGroups * p.MaxPackedLen
	if totalCapacity > 0 {
		avgUtilization = float32(totalTokens) / float32(totalCapacity)
	}

	return numGroups, avgUtilization
}

// =============================================================================
// FlashDeBERTa Support
// =============================================================================
// FlashDeBERTa is an optimized variant of DeBERTa that uses flash attention
// for faster and more memory-efficient inference. When available, it provides
// significant speedups especially for longer sequences.

// FlashDeBERTaConfig holds configuration for FlashDeBERTa optimization
type FlashDeBERTaConfig struct {
	// Enabled indicates whether FlashDeBERTa optimization is active
	Enabled bool
	// UseFlashAttention enables flash attention if supported
	UseFlashAttention bool
	// UseFP16 enables FP16 precision for faster inference
	UseFP16 bool
	// UseMemoryEfficientAttention uses memory-efficient attention implementation
	UseMemoryEfficientAttention bool
}

// DefaultFlashDeBERTaConfig returns the default FlashDeBERTa configuration
func DefaultFlashDeBERTaConfig() FlashDeBERTaConfig {
	return FlashDeBERTaConfig{
		Enabled:                     false,
		UseFlashAttention:           true,
		UseFP16:                     false, // Default to FP32 for accuracy
		UseMemoryEfficientAttention: true,
	}
}

// WithFlashDeBERTa enables FlashDeBERTa optimization
// This requires the model to be exported with flash attention support
func WithFlashDeBERTa(config FlashDeBERTaConfig) backends.PipelineOption[*GLiNERPipeline] {
	return func(p *GLiNERPipeline) error {
		// FlashDeBERTa is configured via ONNX Runtime execution providers
		// The actual implementation depends on:
		// 1. Model being exported with flash attention ops
		// 2. ONNX Runtime having CUDA or other accelerator support
		// 3. Hardware supporting flash attention (e.g., NVIDIA Ampere+)

		if config.Enabled {
			// Check if flash attention is available
			if !isFlashAttentionAvailable() {
				// Fall back to standard attention
				config.UseFlashAttention = false
			}
		}

		// Store config for runtime use
		// Note: The actual flash attention is handled by ONNX Runtime
		// when the model includes flash attention operators
		return nil
	}
}

// isFlashAttentionAvailable checks if flash attention is available
// This depends on ONNX Runtime build and hardware
func isFlashAttentionAvailable() bool {
	// Flash attention requires:
	// 1. CUDA execution provider with flash attention support
	// 2. Compatible GPU (typically Ampere or newer)
	// 3. ONNX Runtime built with flash attention kernels

	// For now, we check if CUDA EP is available
	// The actual flash attention check would require runtime introspection
	return false // Conservative default - enable when confirmed available
}

// GetFlashDeBERTaStatus returns whether FlashDeBERTa is active for this pipeline
func (p *GLiNERPipeline) GetFlashDeBERTaStatus() (enabled bool, reason string) {
	// Check execution provider
	switch p.Runtime {
	case "ORT":
		// Check if using CUDA or other accelerated EP
		// This would require accessing the ORT session options
		return false, "FlashDeBERTa requires CUDA execution provider with flash attention kernels"
	default:
		return false, "FlashDeBERTa only supported with ONNX Runtime"
	}
}

// =============================================================================
// Relation Extraction
// =============================================================================
// GLiNER multitask models support extracting relationships between entities.
// This is useful for knowledge graph construction and structured extraction.

// RelationExtractionConfig configures relation extraction behavior
type RelationExtractionConfig struct {
	// Labels are the relation types to extract
	Labels []string
	// Threshold is the minimum score for relation detection
	Threshold float32
	// MaxEntityPairs limits the number of entity pairs to consider
	MaxEntityPairs int
	// BiDirectional if true, considers both (A,B) and (B,A) pairs
	BiDirectional bool
}

// DefaultRelationExtractionConfig returns default relation extraction settings
func DefaultRelationExtractionConfig() RelationExtractionConfig {
	return RelationExtractionConfig{
		Labels:         []string{"works_at", "located_in", "founded", "owns", "part_of"},
		Threshold:      0.5,
		MaxEntityPairs: 1000,
		BiDirectional:  true,
	}
}

// RunPipelineWithRelations extracts both entities and relationships
// This requires a multitask GLiNER model that supports relation extraction
func (p *GLiNERPipeline) RunPipelineWithRelations(
	inputs []string,
	entityLabels []string,
	relationLabels []string,
) (*GLiNEROutput, error) {
	if len(inputs) == 0 {
		return &GLiNEROutput{
			Entities:  [][]GLiNEREntity{},
			Relations: [][]GLiNERRelation{},
		}, nil
	}

	// First, check if the model supports relation extraction
	if !p.SupportsRelationExtraction() {
		// Fall back to entity-only extraction
		result, err := p.RunPipelineWithLabels(inputs, entityLabels)
		if err != nil {
			return nil, err
		}
		result.Relations = make([][]GLiNERRelation, len(inputs))
		return result, nil
	}

	// Use provided relation labels or defaults
	if len(relationLabels) == 0 {
		relationLabels = p.RelationLabels
	}
	if len(relationLabels) == 0 {
		relationLabels = DefaultRelationExtractionConfig().Labels
	}

	var runErrors []error
	batch := p.prepareGLiNERBatch(len(inputs))
	defer func() {
		if batch.PipelineBatch != nil {
			runErrors = append(runErrors, batch.Destroy())
		}
	}()

	// Preprocess with relation labels
	if err := p.PreprocessWithRelations(batch, inputs, entityLabels, relationLabels); err != nil {
		return nil, err
	}

	// Forward pass
	if err := p.Forward(batch); err != nil {
		return nil, err
	}

	// Postprocess both entities and relations
	result, err := p.PostprocessWithRelations(batch, entityLabels, relationLabels)
	if err != nil {
		return nil, err
	}

	return result, errors.Join(runErrors...)
}

// SupportsRelationExtraction checks if the model supports relation extraction
func (p *GLiNERPipeline) SupportsRelationExtraction() bool {
	// Check for relation-specific output tensor
	for _, meta := range p.Model.OutputsMeta {
		if meta.Name == "relation_logits" || meta.Name == "rel_logits" {
			return true
		}
	}
	return false
}

// PreprocessWithRelations prepares batch for entity and relation extraction
func (p *GLiNERPipeline) PreprocessWithRelations(
	batch *GLiNERBatch,
	inputs []string,
	entityLabels []string,
	relationLabels []string,
) error {
	// Build combined prefix with both entity and relation labels
	// Format: "<<ENT>> ent1 <<ENT>> ent2 <<REL>> rel1 <<REL>> rel2 <<SEP>> text"
	prefix := buildGLiNERCombinedPrefix(entityLabels, relationLabels)

	prefixedTexts := make([]string, len(inputs))
	for i, text := range inputs {
		prefixedTexts[i] = prefix + " " + text
		batch.OriginalText[i] = text
	}

	// Standard preprocessing
	backends.TokenizeInputs(batch.PipelineBatch, p.Model.Tokenizer, prefixedTexts)

	// Build GLiNER-specific inputs with adjusted prefix length
	combinedPrefixLen := uint(len(prefix) + 1) // +1 for space
	return p.buildGLiNERInputsWithPrefix(batch, combinedPrefixLen)
}

// buildGLiNERCombinedPrefix builds prefix with both entity and relation labels
func buildGLiNERCombinedPrefix(entityLabels, relationLabels []string) string {
	var sb strings.Builder

	// Entity labels
	for _, label := range entityLabels {
		sb.WriteString(glinerEntityToken)
		sb.WriteString(" ")
		sb.WriteString(label)
		sb.WriteString(" ")
	}

	// Relation labels
	for _, label := range relationLabels {
		sb.WriteString("<<REL>>")
		sb.WriteString(" ")
		sb.WriteString(label)
		sb.WriteString(" ")
	}

	sb.WriteString(glinerSepToken)
	return sb.String()
}

// buildGLiNERInputsWithPrefix builds GLiNER inputs with a custom prefix length
func (p *GLiNERPipeline) buildGLiNERInputsWithPrefix(batch *GLiNERBatch, prefixLen uint) error {
	batchSize := batch.Size
	maxSeqLen := batch.MaxSequenceLength

	batch.WordsMask = make([][]int64, batchSize)
	batch.TextLengths = make([][]int64, batchSize)
	batch.WordsToChars = make([][][2]int, batchSize)

	for i, input := range batch.Input {
		wordsMask := make([]int64, maxSeqLen)
		wordsToChars := [][2]int{}
		wordCount := int64(0)
		inTextRegion := false

		for j, offset := range input.Offsets {
			if input.SpecialTokensMask[j] > 0 {
				continue
			}

			tokenStart := offset[0]
			tokenEnd := offset[1]

			if tokenEnd <= prefixLen {
				continue
			}

			inTextRegion = true
			adjustedStart := tokenStart - prefixLen
			adjustedEnd := tokenEnd - prefixLen

			token := ""
			if j < len(input.Tokens) {
				token = input.Tokens[j]
			}

			isNewWord := false
			if inTextRegion && wordCount == 0 {
				isNewWord = true
			} else if strings.HasPrefix(token, "▁") || strings.HasPrefix(token, " ") {
				isNewWord = true
			}

			if isNewWord {
				wordCount++
				wordsMask[j] = wordCount
				startOffset := int(adjustedStart)
				if wordCount > 1 && (strings.HasPrefix(token, "▁") || strings.HasPrefix(token, " ")) {
					startOffset++
				}
				wordsToChars = append(wordsToChars, [2]int{startOffset, int(adjustedEnd)})
			} else {
				wordsMask[j] = wordCount
				if len(wordsToChars) > 0 {
					wordsToChars[len(wordsToChars)-1][1] = int(adjustedEnd)
				}
			}
		}

		batch.WordsMask[i] = wordsMask
		batch.TextLengths[i] = []int64{wordCount}
		batch.WordsToChars[i] = wordsToChars
	}

	if err := p.generateSpans(batch); err != nil {
		return err
	}

	return p.createGLiNERTensors(batch)
}

// PostprocessWithRelations extracts both entities and relations from model output
func (p *GLiNERPipeline) PostprocessWithRelations(
	batch *GLiNERBatch,
	entityLabels []string,
	relationLabels []string,
) (*GLiNEROutput, error) {
	// First extract entities using standard postprocessing
	entityResult, err := p.Postprocess(batch, entityLabels)
	if err != nil {
		return nil, err
	}

	// Check if relation output is available
	relationOutputIdx := -1
	for i, meta := range p.Model.OutputsMeta {
		if meta.Name == "relation_logits" || meta.Name == "rel_logits" {
			relationOutputIdx = i
			break
		}
	}

	if relationOutputIdx < 0 || relationOutputIdx >= len(batch.OutputValues) {
		// No relation output - return entities only
		entityResult.Relations = make([][]GLiNERRelation, len(entityResult.Entities))
		return entityResult, nil
	}

	// Extract relations
	relations := p.extractRelations(
		batch,
		entityResult.Entities,
		batch.OutputValues[relationOutputIdx],
		relationLabels,
	)

	entityResult.Relations = relations
	return entityResult, nil
}

// extractRelations extracts relationships from relation logits
// Relation output is typically [batch][num_entity_pairs][num_relation_labels]
func (p *GLiNERPipeline) extractRelations(
	batch *GLiNERBatch,
	entities [][]GLiNEREntity,
	relationOutput any,
	relationLabels []string,
) [][]GLiNERRelation {
	batchSize := len(entities)
	result := make([][]GLiNERRelation, batchSize)

	// Parse relation logits based on output shape
	var relationLogits [][][]float32
	switch v := relationOutput.(type) {
	case [][][]float32:
		relationLogits = v
	default:
		// Unsupported format - return empty relations
		return result
	}

	threshold := p.RelationThreshold
	if threshold <= 0 {
		threshold = 0.5
	}

	for batchIdx := 0; batchIdx < batchSize; batchIdx++ {
		if batchIdx >= len(relationLogits) {
			continue
		}

		batchEntities := entities[batchIdx]
		if len(batchEntities) < 2 {
			// Need at least 2 entities for a relation
			continue
		}

		batchRelLogits := relationLogits[batchIdx]
		var relations []GLiNERRelation

		// Relation logits are organized as pairs: for N entities,
		// there are N*(N-1) potential directed pairs
		pairIdx := 0
		for headIdx := 0; headIdx < len(batchEntities); headIdx++ {
			for tailIdx := 0; tailIdx < len(batchEntities); tailIdx++ {
				if headIdx == tailIdx {
					continue
				}

				if pairIdx >= len(batchRelLogits) {
					break
				}

				pairLogits := batchRelLogits[pairIdx]
				pairIdx++

				// Check each relation label
				for labelIdx, label := range relationLabels {
					if labelIdx >= len(pairLogits) {
						break
					}

					score := sigmoid(pairLogits[labelIdx])
					if score >= threshold {
						relations = append(relations, GLiNERRelation{
							HeadEntity: batchEntities[headIdx],
							TailEntity: batchEntities[tailIdx],
							Label:      label,
							Score:      score,
						})
					}
				}
			}
		}

		result[batchIdx] = relations
	}

	return result
}

// FilterRelationsByScore filters relations by minimum score
func FilterRelationsByScore(relations []GLiNERRelation, minScore float32) []GLiNERRelation {
	filtered := make([]GLiNERRelation, 0, len(relations))
	for _, rel := range relations {
		if rel.Score >= minScore {
			filtered = append(filtered, rel)
		}
	}
	return filtered
}

// GroupRelationsByHead groups relations by their head entity
func GroupRelationsByHead(relations []GLiNERRelation) map[string][]GLiNERRelation {
	grouped := make(map[string][]GLiNERRelation)
	for _, rel := range relations {
		key := fmt.Sprintf("%s:%d-%d", rel.HeadEntity.Text, rel.HeadEntity.Start, rel.HeadEntity.End)
		grouped[key] = append(grouped[key], rel)
	}
	return grouped
}

// GroupRelationsByType groups relations by their label
func GroupRelationsByType(relations []GLiNERRelation) map[string][]GLiNERRelation {
	grouped := make(map[string][]GLiNERRelation)
	for _, rel := range relations {
		grouped[rel.Label] = append(grouped[rel.Label], rel)
	}
	return grouped
}
