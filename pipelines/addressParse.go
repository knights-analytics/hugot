package pipelines

import (
	"context"
	"errors"
	"fmt"
	"math"
	"path"
	"regexp"
	"strings"
	"sync/atomic"
	"time"
	"unicode"

	"github.com/gomlx/go-huggingface/tokenizers"
	"github.com/gomlx/go-huggingface/tokenizers/api"
	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/util/safeconv"
)

var addressParseHyphenSplit = regexp.MustCompile(`^([0-9]*[a-z]?)-([0-9]*[a-z]?) `)

type (
	AddressParseConfig = backends.PipelineConfig[*AddressParsePipeline]
	AddressParseOption = backends.PipelineOption[*AddressParsePipeline]
)

// AddressParsePipeline runs AddressParse's BPEmb model using separate embedder,
// encoder, and decoder-step ONNX graphs.
type AddressParsePipeline struct {
	*backends.BasePipeline
	Encoder *backends.Model
	Decoder *backends.Model

	HyphenSplit         bool
	ManifestFilename    string
	batchSize           int
	withProbabilities   bool
	probabilityRounding int
	preProcessors       []AddressParsePreprocessor
	manifest            addressParseManifest
	tokenizer           tokenizers.Tokenizer
	unknownTokenID      int
	tagNames            map[int]string
}

// AddressParsePreprocessor transforms an address before AddressParse's built-in
// lowercase, comma removal, whitespace normalization, and optional hyphen split.
type AddressParsePreprocessor func(string) string

type addressParseManifest struct {
	FormatVersion       int                      `json:"format_version"`
	Pipeline            string                   `json:"pipeline"`
	AddressParseVersion string                   `json:"deep_parse_version"`
	CheckpointVersion   string                   `json:"checkpoint_version"`
	Model               string                   `json:"model"`
	Architecture        addressParseArchitecture `json:"architecture"`
	Tags                []addressParseTag        `json:"tags"`
	BPEmb               addressParseBPEmb        `json:"bpemb"`
	Graphs              addressParseGraphs       `json:"graphs"`
}

type addressParseArchitecture struct {
	InputSize              int  `json:"input_size"`
	EmbeddingHiddenSize    int  `json:"embedding_hidden_size"`
	EmbeddingNumLayers     int  `json:"embedding_num_layers"`
	EmbeddingBidirectional bool `json:"embedding_bidirectional"`
	EmbeddingMaxPool       bool `json:"embedding_maxpool"`
	EncoderHiddenSize      int  `json:"encoder_hidden_size"`
	EncoderNumLayers       int  `json:"encoder_num_layers"`
	DecoderHiddenSize      int  `json:"decoder_hidden_size"`
	DecoderNumLayers       int  `json:"decoder_num_layers"`
	DecoderInputSize       int  `json:"decoder_input_size"`
	OutputSize             int  `json:"output_size"`
	Attention              bool `json:"attention"`
}

type addressParseTag struct {
	ID   int    `json:"id"`
	Name string `json:"name"`
}

type addressParseBPEmb struct {
	Language            string `json:"language"`
	VocabularySize      int    `json:"vocabulary_size"`
	EmbeddingDimension  int    `json:"embedding_dimension"`
	SentencePieceModel  string `json:"sentencepiece_model"`
	SourceSentencePiece string `json:"source_sentencepiece_model"`
	Normalization       string `json:"normalization"`
	PaddingPieceLength  int    `json:"padding_piece_length"`
	FuseUnknownPieces   bool   `json:"fuse_unknown_pieces"`
}

type addressParseGraphs struct {
	Primary           string `json:"primary"`
	Encoder           string `json:"encoder"`
	EncoderBatching   string `json:"encoder_batching"`
	DecoderStep       string `json:"decoder_step"`
	DecoderInitialTag int    `json:"decoder_initial_tag"`
	DecoderSteps      string `json:"decoder_steps"`
}

// AddressParseWord is a word-level prediction from the address parser.
type AddressParseWord struct {
	Word  string
	Tag   string
	TagID int
	Score float32
}

// AddressParseComponent groups adjacent words with the same predicted tag.
type AddressParseComponent struct {
	Tag    string
	Value  string
	Scores []float32
}

// AddressParseAddress contains word-level predictions and adjacent parsed fields.
type AddressParseAddress struct {
	Address        string
	CleanedAddress string
	Words          []AddressParseWord
	Components     []AddressParseComponent
}

// ToFields combines adjacent components with the same tag into a convenient
// tag-to-value map. Repeated fields are joined with a space.
func (a AddressParseAddress) ToFields() map[string]string {
	fields := make(map[string]string, len(a.Components))
	for _, component := range a.Components {
		if value, exists := fields[component.Tag]; exists {
			fields[component.Tag] = value + " " + component.Value
		} else {
			fields[component.Tag] = component.Value
		}
	}
	return fields
}

// AddressParseOutput is the typed result for one batch of addresses.
type AddressParseOutput struct {
	Addresses []AddressParseAddress
}

func (o *AddressParseOutput) GetOutput() []any {
	if o == nil {
		return nil
	}
	output := make([]any, len(o.Addresses))
	for i := range o.Addresses {
		output[i] = o.Addresses[i]
	}
	return output
}

// WithAddressParseHyphenSplit enables splitting a leading unit/street-number
// pair such as "3-305 Main Street" into "3 305 Main Street".
func WithAddressParseHyphenSplit() AddressParseOption {
	return func(p *AddressParsePipeline) error {
		p.HyphenSplit = true
		return nil
	}
}

// WithAddressParseBatchSize limits the number of addresses processed in each
// inference batch. The default is 32.
func WithAddressParseBatchSize(size int) AddressParseOption {
	return func(p *AddressParsePipeline) error {
		if size <= 0 {
			return errors.New("AddressParse batch size must be positive")
		}
		p.batchSize = size
		return nil
	}
}

// WithAddressParseProbabilities controls whether word prediction scores are
// included. Scores are enabled by default for backwards compatibility.
func WithAddressParseProbabilities(enabled bool) AddressParseOption {
	return func(p *AddressParsePipeline) error {
		p.withProbabilities = enabled
		return nil
	}
}

// WithAddressParseProbabilityRounding rounds included scores to digits decimal
// places using ties-to-even rounding. A negative value disables rounding;
// values from 0 through 9 are supported.
func WithAddressParseProbabilityRounding(digits int) AddressParseOption {
	return func(p *AddressParsePipeline) error {
		if digits < -1 || digits > 9 {
			return errors.New("AddressParse probability rounding must be between -1 and 9")
		}
		p.probabilityRounding = digits
		return nil
	}
}

// WithAddressParsePreprocessors applies each processor in order before built-in
// address cleaning. Nil processors are ignored.
func WithAddressParsePreprocessors(processors ...AddressParsePreprocessor) AddressParseOption {
	return func(p *AddressParsePipeline) error {
		for _, processor := range processors {
			if processor != nil {
				p.preProcessors = append(p.preProcessors, processor)
			}
		}
		return nil
	}
}

// WithAddressParseManifestFilename selects a manifest relative to the model directory.
func WithAddressParseManifestFilename(filename string) AddressParseOption {
	return func(p *AddressParsePipeline) error {
		p.ManifestFilename = filename
		return nil
	}
}

// NewAddressParsePipeline initializes a AddressParse export bundle.
func NewAddressParsePipeline(ctx context.Context, config AddressParseConfig, model *backends.Model) (*AddressParsePipeline, error) {
	if model == nil {
		return nil, errors.New("AddressParse requires a BPEmb model")
	}
	if model.ORTModel == nil && model.GoMLXModel == nil {
		return nil, errors.New("AddressParse requires an initialized ONNX Runtime or GoMLX backend")
	}
	p := &AddressParsePipeline{
		BasePipeline: backends.NewBasePipeline(ctx, config, model), ManifestFilename: "manifest.json",
		batchSize: 32, withProbabilities: true, probabilityRounding: -1,
	}
	for _, option := range config.Options {
		if option == nil {
			continue
		}
		if err := option(p); err != nil {
			return nil, err
		}
	}

	if !validAddressParseAssetName(p.ManifestFilename, ".json") {
		return nil, errors.New("AddressParse manifest must be a relative .json path")
	}
	if err := readNativeConfig(ctx, model.Path, p.ManifestFilename, &p.manifest); err != nil {
		return nil, fmt.Errorf("read AddressParse manifest: %w", err)
	}
	if err := p.validateManifest(); err != nil {
		return nil, err
	}
	if err := p.validatePrimaryGraph(); err != nil {
		return nil, err
	}
	if model.Tokenizer == nil || model.Tokenizer.GoTokenizer == nil || model.Tokenizer.GoTokenizer.Tokenizer == nil {
		return nil, errors.New("AddressParse requires tokenizer.json; re-export the bundle with the updated AddressParse exporter")
	}
	p.tokenizer = model.Tokenizer.GoTokenizer.Tokenizer
	if p.tokenizer.VocabSize() != p.manifest.BPEmb.VocabularySize {
		return nil, errors.New("AddressParse tokenizer vocabulary size does not match the manifest")
	}
	var err error
	p.unknownTokenID, err = p.tokenizer.SpecialTokenID(api.TokUnknown)
	if err != nil && p.manifest.BPEmb.FuseUnknownPieces {
		return nil, fmt.Errorf("AddressParse tokenizer requires an unknown token: %w", err)
	}
	if err := requireTensorNames(model,
		[]string{"piece_ids", "piece_lengths", "forward_hidden", "forward_cell", "backward_hidden", "backward_cell"},
		[]string{"word_vectors"}); err != nil {
		return nil, fmt.Errorf("invalid AddressParse BPEmb embedder graph: %w", err)
	}

	p.Encoder, err = model.LoadGraph(ctx, p.manifest.Graphs.Encoder)
	if err != nil {
		return nil, fmt.Errorf("load AddressParse encoder: %w", err)
	}
	if err := requireTensorNames(p.Encoder,
		[]string{"word_vectors", "initial_hidden", "initial_cell"},
		[]string{"encoder_states", "decoder_hidden", "decoder_cell"}); err != nil {
		return nil, fmt.Errorf("invalid AddressParse encoder graph: %w", err)
	}

	p.Decoder, err = model.LoadGraph(ctx, p.manifest.Graphs.DecoderStep)
	if err != nil {
		return nil, fmt.Errorf("load AddressParse decoder: %w", err)
	}
	decoderInputs := []string{"previous_tag", "hidden", "cell"}
	if p.manifest.Architecture.Attention {
		decoderInputs = append(decoderInputs, "encoder_states", "lengths")
	}
	if err := requireTensorNames(p.Decoder,
		decoderInputs,
		[]string{"log_probabilities", "next_hidden", "next_cell"}); err != nil {
		return nil, fmt.Errorf("invalid AddressParse decoder graph: %w", err)
	}

	p.tagNames = make(map[int]string, len(p.manifest.Tags))
	for _, tag := range p.manifest.Tags {
		p.tagNames[tag.ID] = tag.Name
	}
	return p, p.Validate()
}

func (p *AddressParsePipeline) validateManifest() error {
	m := p.manifest
	if m.FormatVersion != 2 || m.Pipeline != "deepparse" {
		return fmt.Errorf("unsupported AddressParse export manifest format %d for model %q", m.FormatVersion, m.Model)
	}
	decoderInputSize := 1
	if m.Architecture.Attention {
		decoderInputSize += m.Architecture.EncoderHiddenSize
	}
	if m.Architecture.InputSize <= 0 || m.Architecture.OutputSize <= 0 ||
		m.Architecture.EmbeddingHiddenSize <= 0 || m.Architecture.EmbeddingNumLayers <= 0 ||
		m.Architecture.EncoderHiddenSize <= 0 || m.Architecture.EncoderNumLayers <= 0 ||
		m.Architecture.DecoderNumLayers <= 0 || m.Architecture.DecoderInputSize != decoderInputSize ||
		m.Architecture.DecoderHiddenSize != m.Architecture.EncoderHiddenSize ||
		m.Architecture.EncoderNumLayers != m.Architecture.DecoderNumLayers {
		return errors.New("unsupported AddressParse architecture in manifest")
	}
	if m.BPEmb.VocabularySize <= 0 || m.BPEmb.EmbeddingDimension != m.Architecture.InputSize ||
		m.BPEmb.PaddingPieceLength != 0 || m.BPEmb.EmbeddingDimension <= 0 {
		return errors.New("invalid BPEmb settings in AddressParse manifest")
	}
	if !validAddressParseGraphName(m.Graphs.Primary) ||
		!validAddressParseGraphName(m.Graphs.Encoder) || !validAddressParseGraphName(m.Graphs.DecoderStep) ||
		m.Graphs.EncoderBatching != "equal_address_lengths" ||
		m.Graphs.DecoderInitialTag < -1 || m.Graphs.DecoderInitialTag >= m.Architecture.OutputSize ||
		m.Graphs.DecoderSteps != "max_words_plus_eos" {
		return errors.New("invalid graph contract in AddressParse manifest")
	}
	if len(m.Tags) != m.Architecture.OutputSize {
		return errors.New("AddressParse tag map size does not match architecture output size")
	}
	seenIDs, seenNames := make(map[int]bool, len(m.Tags)), make(map[string]bool, len(m.Tags))
	for _, tag := range m.Tags {
		if tag.ID < 0 || tag.ID >= len(m.Tags) || strings.TrimSpace(tag.Name) == "" || seenIDs[tag.ID] || seenNames[tag.Name] {
			return errors.New("AddressParse tag IDs must be contiguous and tag names unique")
		}
		seenIDs[tag.ID], seenNames[tag.Name] = true, true
	}
	return nil
}

func validAddressParseGraphName(name string) bool {
	return validAddressParseAssetName(name, ".onnx")
}

func validAddressParseAssetName(name, extension string) bool {
	name = strings.ReplaceAll(name, `\`, "/")
	if name == "" || path.IsAbs(name) || strings.Contains(name, ":") || path.Ext(name) != extension {
		return false
	}
	for _, part := range strings.Split(name, "/") {
		if part == ".." {
			return false
		}
	}
	return true
}

func (p *AddressParsePipeline) validatePrimaryGraph() error {
	expected := path.Clean(strings.ReplaceAll(p.manifest.Graphs.Primary, `\`, "/"))
	selected := p.Model.OnnxPath
	if selected == "" {
		selected = p.Model.OnnxFilename
	}
	selected = strings.ReplaceAll(selected, `\`, "/")
	modelPath := strings.TrimSuffix(strings.ReplaceAll(p.Model.Path, `\`, "/"), "/")
	if strings.HasPrefix(selected, modelPath+"/") {
		selected = strings.TrimPrefix(selected, modelPath+"/")
	}
	selected = path.Clean(selected)
	if selected != expected {
		return fmt.Errorf("AddressParse manifest primary graph %q does not match loaded model graph %q", expected, selected)
	}
	return nil
}

func (*AddressParsePipeline) IsGenerative() bool { return false }

func (p *AddressParsePipeline) GetModel() *backends.Model { return p.Model }

func (p *AddressParsePipeline) GetMetadata() backends.PipelineMetadata {
	return nativeMetadata(p.Model)
}

func (p *AddressParsePipeline) GetStatistics() backends.PipelineStatistics {
	return nativeStatistics(p.BasePipeline)
}

func (p *AddressParsePipeline) Validate() error {
	if p == nil || p.BasePipeline == nil || p.Model == nil || p.Encoder == nil || p.Decoder == nil ||
		p.tokenizer == nil || len(p.tagNames) == 0 {
		return errors.New("AddressParse pipeline is not fully initialized")
	}
	if p.Model.ORTModel == nil && p.Model.GoMLXModel == nil {
		return errors.New("AddressParse requires an initialized ONNX Runtime or GoMLX backend")
	}
	return nil
}

// Run implements the common pipeline interface for a batch of addresses.
func (p *AddressParsePipeline) Run(ctx context.Context, addresses []string) (backends.PipelineBatchOutput, error) {
	return p.RunAddresses(ctx, addresses)
}

// RunAddresses parses a batch of raw street addresses.
func (p *AddressParsePipeline) RunAddresses(ctx context.Context, addresses []string) (*AddressParseOutput, error) {
	if ctx == nil {
		return nil, errors.New("nil inference context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := p.Validate(); err != nil {
		return nil, err
	}
	if len(addresses) == 0 {
		return nil, errors.New("at least one address is required")
	}
	batchSize := p.batchSize
	if batchSize <= 0 { // Also keeps pipelines assembled directly in Go safe to use.
		batchSize = 32
	}
	output := &AddressParseOutput{Addresses: make([]AddressParseAddress, 0, len(addresses))}
	for start := 0; start < len(addresses); start += batchSize {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		end := min(start+batchSize, len(addresses))
		batch, err := p.runAddressBatch(ctx, addresses[start:end], start)
		if err != nil {
			return nil, err
		}
		output.Addresses = append(output.Addresses, batch.Addresses...)
	}
	return output, nil
}

func (p *AddressParsePipeline) runAddressBatch(ctx context.Context, addresses []string, offset int) (*AddressParseOutput, error) {
	cleaned := make([]string, len(addresses))
	words := make([][]string, len(addresses))
	pieceIDs := make([][][]int64, len(addresses))
	maxWords, maxPieces := 0, 0
	tokenizerStart := time.Now()
	for i, address := range addresses {
		addressIndex := offset + i
		if strings.TrimSpace(address) == "" {
			return nil, fmt.Errorf("address %d is empty", addressIndex)
		}
		processed := address
		for _, processor := range p.preProcessors {
			processed = processor(processed)
		}
		cleaned[i] = cleanAddressParseAddress(processed, p.HyphenSplit)
		words[i] = strings.Fields(cleaned[i])
		if len(words[i]) == 0 {
			return nil, fmt.Errorf("address %d is empty after AddressParse cleaning", addressIndex)
		}
		pieceIDs[i] = make([][]int64, len(words[i]))
		for j, word := range words[i] {
			tokens := p.encodeWord(word)
			if len(tokens) == 0 {
				return nil, fmt.Errorf("SentencePiece produced no pieces for address %d word %q", addressIndex, word)
			}
			pieceIDs[i][j] = make([]int64, len(tokens))
			for k, tokenID := range tokens {
				if tokenID < 0 || tokenID >= p.manifest.BPEmb.VocabularySize {
					return nil, fmt.Errorf("tokenizer produced invalid piece ID %d for address %d word %q", tokenID, addressIndex, word)
				}
				pieceIDs[i][j][k] = int64(tokenID)
			}
			maxPieces = max(maxPieces, len(tokens))
		}
		maxWords = max(maxWords, len(words[i]))
	}
	atomic.AddUint64(&p.TokenizerTimings.NumCalls, 1)
	atomic.AddUint64(&p.TokenizerTimings.TotalNS, safeconv.DurationToU64(time.Since(tokenizerStart)))

	batchSize := len(addresses)
	flatIDs := make([]int64, batchSize*maxWords*maxPieces)
	flatPieceLengths := make([]int64, batchSize*maxWords)
	for i := range pieceIDs {
		for j, pieces := range pieceIDs[i] {
			wordIndex := i*maxWords + j
			flatPieceLengths[wordIndex] = int64(len(pieces))
			copy(flatIDs[wordIndex*maxPieces:wordIndex*maxPieces+len(pieces)], pieces)
		}
	}
	embeddingLayers := int64(p.manifest.Architecture.EmbeddingNumLayers)
	embeddingHiddenSize := int64(p.manifest.Architecture.EmbeddingHiddenSize)
	embeddingState := make([]float32, int(embeddingLayers*int64(batchSize*maxWords)*embeddingHiddenSize))
	embeddingStateTensor := backends.Tensor{
		Shape: []int64{embeddingLayers, int64(batchSize * maxWords), embeddingHiddenSize},
		Data:  embeddingState,
	}
	wordOutput, err := runNativeTensors(ctx, p.BasePipeline, p.Model, map[string]backends.Tensor{
		"piece_ids":       {Shape: []int64{int64(batchSize), int64(maxWords), int64(maxPieces)}, Data: flatIDs},
		"piece_lengths":   {Shape: []int64{int64(batchSize), int64(maxWords)}, Data: flatPieceLengths},
		"forward_hidden":  embeddingStateTensor,
		"forward_cell":    embeddingStateTensor,
		"backward_hidden": embeddingStateTensor,
		"backward_cell":   embeddingStateTensor,
	})
	if err != nil {
		return nil, err
	}
	wordVectors, err := addressParseFloatData(wordOutput["word_vectors"], []int64{int64(batchSize), int64(maxWords), int64(p.manifest.Architecture.InputSize)})
	if err != nil {
		return nil, fmt.Errorf("invalid AddressParse BPEmb embedder output: %w", err)
	}
	// The ONNX encoder uses a plain LSTM. Group addresses by word count so
	// every row ends at its true final word, matching AddressParse's packed LSTM.
	hiddenSize := int64(p.manifest.Architecture.EncoderHiddenSize)
	layers := int64(p.manifest.Architecture.EncoderNumLayers)
	inputSize := int64(p.manifest.Architecture.InputSize)
	hidden := make([]float32, int(layers*int64(batchSize)*hiddenSize))
	cell := make([]float32, len(hidden))
	var encoderStates []float32
	var addressLengths []int64
	if p.manifest.Architecture.Attention {
		encoderStates = make([]float32, batchSize*maxWords*int(hiddenSize))
		addressLengths = make([]int64, batchSize)
	}
	lengthGroups := make(map[int][]int)
	for i, addressWords := range words {
		lengthGroups[len(addressWords)] = append(lengthGroups[len(addressWords)], i)
	}
	for addressLength, indices := range lengthGroups {
		groupVectors := make([]float32, len(indices)*addressLength*int(inputSize))
		for groupIndex, addressIndex := range indices {
			sourceStart := addressIndex * maxWords * int(inputSize)
			targetStart := groupIndex * addressLength * int(inputSize)
			copy(groupVectors[targetStart:targetStart+addressLength*int(inputSize)],
				wordVectors[sourceStart:sourceStart+addressLength*int(inputSize)])
		}
		encoded, runErr := runNativeTensors(ctx, p.BasePipeline, p.Encoder, map[string]backends.Tensor{
			"word_vectors": {
				Shape: []int64{int64(len(indices)), int64(addressLength), inputSize},
				Data:  groupVectors,
			},
			"initial_hidden": {
				Shape: []int64{layers, int64(len(indices)), hiddenSize},
				Data:  make([]float32, int(layers*int64(len(indices))*hiddenSize)),
			},
			"initial_cell": {
				Shape: []int64{layers, int64(len(indices)), hiddenSize},
				Data:  make([]float32, int(layers*int64(len(indices))*hiddenSize)),
			},
		})
		if runErr != nil {
			return nil, runErr
		}
		groupEncoderStates, shapeErr := addressParseFloatData(encoded["encoder_states"], []int64{int64(len(indices)), int64(addressLength), hiddenSize})
		if shapeErr != nil {
			return nil, fmt.Errorf("invalid AddressParse encoder states: %w", shapeErr)
		}
		groupHidden, shapeErr := addressParseFloatData(encoded["decoder_hidden"], []int64{layers, int64(len(indices)), hiddenSize})
		if shapeErr != nil {
			return nil, fmt.Errorf("invalid AddressParse initial hidden state: %w", shapeErr)
		}
		groupCell, shapeErr := addressParseFloatData(encoded["decoder_cell"], []int64{layers, int64(len(indices)), hiddenSize})
		if shapeErr != nil {
			return nil, fmt.Errorf("invalid AddressParse initial cell state: %w", shapeErr)
		}
		if p.manifest.Architecture.Attention {
			for groupIndex, addressIndex := range indices {
				sourceStart := groupIndex * addressLength * int(hiddenSize)
				targetStart := addressIndex * maxWords * int(hiddenSize)
				copy(encoderStates[targetStart:targetStart+addressLength*int(hiddenSize)],
					groupEncoderStates[sourceStart:sourceStart+addressLength*int(hiddenSize)])
				addressLengths[addressIndex] = int64(addressLength)
			}
		}
		for layer := int64(0); layer < layers; layer++ {
			for groupIndex, addressIndex := range indices {
				sourceStart := int((layer*int64(len(indices)) + int64(groupIndex)) * hiddenSize)
				targetStart := int((layer*int64(batchSize) + int64(addressIndex)) * hiddenSize)
				copy(hidden[targetStart:targetStart+int(hiddenSize)], groupHidden[sourceStart:sourceStart+int(hiddenSize)])
				copy(cell[targetStart:targetStart+int(hiddenSize)], groupCell[sourceStart:sourceStart+int(hiddenSize)])
			}
		}
	}

	steps := maxWords + 1 // AddressParse always predicts one extra decoder step for EOS.
	predictions := make([][]AddressParseWord, batchSize)
	previousTags := make([]float32, batchSize)
	for i := range previousTags {
		previousTags[i] = float32(p.manifest.Graphs.DecoderInitialTag)
	}
	for step := 0; step < steps; step++ {
		decoderInputs := map[string]backends.Tensor{
			"previous_tag": {Shape: []int64{1, int64(batchSize), 1}, Data: previousTags},
			"hidden":       {Shape: []int64{layers, int64(batchSize), hiddenSize}, Data: hidden},
			"cell":         {Shape: []int64{layers, int64(batchSize), hiddenSize}, Data: cell},
		}
		if p.manifest.Architecture.Attention {
			// Mask the padded states so shorter addresses attend only to their words.
			decoderInputs["encoder_states"] = backends.Tensor{
				Shape: []int64{int64(batchSize), int64(maxWords), hiddenSize}, Data: encoderStates,
			}
			decoderInputs["lengths"] = backends.Tensor{Shape: []int64{int64(batchSize)}, Data: addressLengths}
		}
		decoded, runErr := runNativeTensors(ctx, p.BasePipeline, p.Decoder, decoderInputs)
		if runErr != nil {
			return nil, runErr
		}
		logProbabilities, shapeErr := addressParseFloatData(decoded["log_probabilities"], []int64{int64(batchSize), int64(p.manifest.Architecture.OutputSize)})
		if shapeErr != nil {
			return nil, fmt.Errorf("invalid AddressParse decoder probabilities: %w", shapeErr)
		}
		for i := range addresses {
			bestID, bestLogProbability := 0, float32(math.Inf(-1))
			for id := 0; id < p.manifest.Architecture.OutputSize; id++ {
				value := logProbabilities[i*p.manifest.Architecture.OutputSize+id]
				if value > bestLogProbability {
					bestID, bestLogProbability = id, value
				}
			}
			previousTags[i] = float32(bestID)
			if step < len(words[i]) {
				score := float32(math.Exp(float64(bestLogProbability)))
				if p.probabilityRounding >= 0 {
					factor := math.Pow10(p.probabilityRounding)
					score = float32(math.RoundToEven(float64(score)*factor) / factor)
				}
				if !p.withProbabilities {
					score = 0
				}
				predictions[i] = append(predictions[i], AddressParseWord{
					Word: words[i][step], Tag: p.tagNames[bestID], TagID: bestID,
					Score: score,
				})
			}
		}
		hidden, shapeErr = addressParseFloatData(decoded["next_hidden"], []int64{layers, int64(batchSize), hiddenSize})
		if shapeErr != nil {
			return nil, fmt.Errorf("invalid AddressParse next hidden state: %w", shapeErr)
		}
		cell, shapeErr = addressParseFloatData(decoded["next_cell"], []int64{layers, int64(batchSize), hiddenSize})
		if shapeErr != nil {
			return nil, fmt.Errorf("invalid AddressParse next cell state: %w", shapeErr)
		}
	}

	output := &AddressParseOutput{Addresses: make([]AddressParseAddress, batchSize)}
	for i := range addresses {
		output.Addresses[i] = AddressParseAddress{
			Address: addresses[i], CleanedAddress: cleaned[i], Words: predictions[i],
			Components: groupAddressParseComponents(predictions[i]),
		}
	}
	return output, nil
}

func (p *AddressParsePipeline) encodeWord(word string) []int {
	tokens := p.tokenizer.Encode(preprocessAddressParseWord(word))
	if !p.manifest.BPEmb.FuseUnknownPieces {
		return tokens
	}
	// SentencePiece collapses adjacent unknown symbols into one piece. The
	// HuggingFace BPE loader currently emits one unknown ID per symbol.
	pieces := tokens[:0]
	for _, id := range tokens {
		if id == p.unknownTokenID && len(pieces) > 0 && pieces[len(pieces)-1] == id {
			continue
		}
		pieces = append(pieces, id)
	}
	return pieces
}

func preprocessAddressParseWord(word string) string {
	return strings.Map(func(r rune) rune {
		if unicode.IsDigit(r) {
			return '0'
		}
		return r
	}, strings.ToLower(word))
}

func cleanAddressParseAddress(address string, splitHyphen bool) string {
	cleaned := strings.ToLower(strings.ReplaceAll(address, ",", ""))
	cleaned = strings.Join(strings.Fields(cleaned), " ")
	if splitHyphen {
		cleaned = addressParseHyphenSplit.ReplaceAllString(cleaned, "$1 $2 ")
		cleaned = strings.Join(strings.Fields(cleaned), " ")
	}
	return cleaned
}

func groupAddressParseComponents(words []AddressParseWord) []AddressParseComponent {
	components := make([]AddressParseComponent, 0, len(words))
	for _, word := range words {
		if len(components) == 0 || components[len(components)-1].Tag != word.Tag {
			components = append(components, AddressParseComponent{Tag: word.Tag, Value: word.Word, Scores: []float32{word.Score}})
			continue
		}
		last := &components[len(components)-1]
		last.Value += " " + word.Word
		last.Scores = append(last.Scores, word.Score)
	}
	return components
}

func addressParseFloatData(tensor backends.Tensor, expectedShape []int64) ([]float32, error) {
	values, ok := tensor.Data.([]float32)
	if !ok {
		return nil, fmt.Errorf("expected float32 output, got %T", tensor.Data)
	}
	if len(tensor.Shape) != len(expectedShape) {
		return nil, fmt.Errorf("expected shape %v, got %v", expectedShape, tensor.Shape)
	}
	count := int64(1)
	for axis, dimension := range expectedShape {
		if tensor.Shape[axis] != dimension {
			return nil, fmt.Errorf("expected shape %v, got %v", expectedShape, tensor.Shape)
		}
		count *= dimension
	}
	if int64(len(values)) != count {
		return nil, fmt.Errorf("shape %v has %d elements, expected %d", tensor.Shape, len(values), count)
	}
	return values, nil
}
