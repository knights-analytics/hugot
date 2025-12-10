package backends

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/util/fileutil"
)

// Seq2SeqPipelineInterface defines the interface for seq2seq pipeline access.
// This avoids import cycles between backends and pipelines packages.
type Seq2SeqPipelineInterface interface {
	GetEncoderModel() *Model
	GetDecoderInitModel() *Model
	GetDecoderModel() *Model
	GetTokenizer() *Tokenizer
	GetRuntime() string
	GetMaxNewTokens() int
	GetNumReturnSeqs() int
	GetDoSample() bool
	GetTopP() float32
	GetTemperature() float32
	GetRepetitionPenalty() float32
	GetDecoderStartTokenID() int64
	GetEosTokenIDs() map[int64]bool
	GetPadTokenID() int64
	GetNumDecoderLayers() int
	GetVocabSize() int
}

// Seq2SeqBatchInterface defines the interface for seq2seq batch access.
type Seq2SeqBatchInterface interface {
	GetSize() int
	GetInputTokenIDs() [][]int64
	GetInputAttentionMask() [][]int64
	GetMaxInputLength() int
	SetEncoderHiddenStates(states any)
	GetEncoderHiddenStates() any
	SetEncoderAttentionMask(mask any)
	GetEncoderAttentionMask() any
	SetPastKeyValues(pkv []any)
	GetPastKeyValues() []any
	SetLogits(logits any)
	GetLogits() any
	GetGeneratedTokens() [][]int64
	SetGeneratedTokens(tokens [][]int64)
	GetFinished() []bool
	SetFinished(finished []bool)
	GetFinishedCount() int
	SetFinishedCount(count int)
	SetDestroyEncoder(fn func() error)
	SetDestroyDecoder(fn func() error)
}

// Seq2SeqConfig holds configuration for seq2seq models (T5, BART, etc.).
type Seq2SeqConfig struct {
	DecoderStartTokenID int64
	EosTokenIDs         map[int64]bool
	PadTokenID          int64
	NumDecoderLayers    int
	NumHeads            int
	HeadDim             int
	DModel              int // Hidden size (d_model)
	VocabSize           int
}

// Seq2SeqTokenized holds tokenized inputs for seq2seq models.
type Seq2SeqTokenized struct {
	TokenIDs      [][]int64
	AttentionMask [][]int64
	MaxLength     int
}

// LoadSeq2SeqEncoder loads the encoder model for seq2seq inference.
// Looks for encoder.onnx or *-encoder*.onnx in the model path.
func LoadSeq2SeqEncoder(modelPath string, opts *options.Options) (*Model, error) {
	onnxFile, err := findOnnxFile(modelPath, "encoder")
	if err != nil {
		return nil, err
	}

	model := &Model{
		Path:         modelPath,
		OnnxFilename: onnxFile,
		Pipelines:    make(map[string]Pipeline),
	}

	if err := LoadOnnxModelBytes(model); err != nil {
		return nil, err
	}

	if err := CreateModelBackend(model, opts); err != nil {
		return nil, err
	}

	model.Destroy = func() error {
		switch opts.Backend {
		case "ORT":
			if model.ORTModel != nil {
				return model.ORTModel.Destroy()
			}
		case "GO", "XLA":
			if model.GoMLXModel != nil {
				model.GoMLXModel.Destroy()
			}
		}
		return nil
	}

	return model, nil
}

// LoadSeq2SeqDecoderInit loads the initial decoder model (no past_key_values).
// Looks for decoder-init.onnx or *-init-decoder*.onnx in the model path.
func LoadSeq2SeqDecoderInit(modelPath string, opts *options.Options) (*Model, error) {
	onnxFile, err := findOnnxFile(modelPath, "init-decoder", "decoder-init")
	if err != nil {
		return nil, err
	}

	model := &Model{
		Path:         modelPath,
		OnnxFilename: onnxFile,
		Pipelines:    make(map[string]Pipeline),
	}

	if err := LoadOnnxModelBytes(model); err != nil {
		return nil, err
	}

	if err := CreateModelBackend(model, opts); err != nil {
		return nil, err
	}

	model.Destroy = func() error {
		switch opts.Backend {
		case "ORT":
			if model.ORTModel != nil {
				return model.ORTModel.Destroy()
			}
		case "GO", "XLA":
			if model.GoMLXModel != nil {
				model.GoMLXModel.Destroy()
			}
		}
		return nil
	}

	return model, nil
}

// LoadSeq2SeqDecoder loads the decoder model with past_key_values support.
// Looks for decoder.onnx (but not decoder-init.onnx) in the model path.
func LoadSeq2SeqDecoder(modelPath string, opts *options.Options) (*Model, error) {
	onnxFile, err := findDecoderOnnxFile(modelPath)
	if err != nil {
		return nil, err
	}

	model := &Model{
		Path:         modelPath,
		OnnxFilename: onnxFile,
		Pipelines:    make(map[string]Pipeline),
	}

	if err := LoadOnnxModelBytes(model); err != nil {
		return nil, err
	}

	if err := CreateModelBackend(model, opts); err != nil {
		return nil, err
	}

	model.Destroy = func() error {
		switch opts.Backend {
		case "ORT":
			if model.ORTModel != nil {
				return model.ORTModel.Destroy()
			}
		case "GO", "XLA":
			if model.GoMLXModel != nil {
				model.GoMLXModel.Destroy()
			}
		}
		return nil
	}

	return model, nil
}

// LoadSeq2SeqTokenizer loads the tokenizer for seq2seq models.
func LoadSeq2SeqTokenizer(modelPath string, opts *options.Options) (*Tokenizer, error) {
	// Create a temporary model struct to use existing tokenizer loading
	tempModel := &Model{
		Path: modelPath,
	}

	if err := LoadTokenizer(tempModel, opts); err != nil {
		return nil, err
	}

	return tempModel.Tokenizer, nil
}

// LoadSeq2SeqConfig loads the model configuration from config.json.
func LoadSeq2SeqConfig(modelPath string) (*Seq2SeqConfig, error) {
	configPath := fileutil.PathJoinSafe(modelPath, "config.json")

	exists, err := fileutil.FileExists(configPath)
	if err != nil {
		return nil, err
	}
	if !exists {
		return nil, fmt.Errorf("config.json not found at %s", modelPath)
	}

	configBytes, err := fileutil.ReadFileBytes(configPath)
	if err != nil {
		return nil, err
	}

	var configMap map[string]any
	if err := json.Unmarshal(configBytes, &configMap); err != nil {
		return nil, err
	}

	config := &Seq2SeqConfig{
		EosTokenIDs: make(map[int64]bool),
	}

	// Decoder start token (typically 0 for T5)
	if v, ok := configMap["decoder_start_token_id"].(float64); ok {
		config.DecoderStartTokenID = int64(v)
	}

	// EOS token(s)
	if eosRaw, exists := configMap["eos_token_id"]; exists {
		switch v := eosRaw.(type) {
		case []any:
			for _, item := range v {
				if num, ok := item.(float64); ok {
					config.EosTokenIDs[int64(num)] = true
				}
			}
		case float64:
			config.EosTokenIDs[int64(v)] = true
		}
	}

	// Pad token
	if v, ok := configMap["pad_token_id"].(float64); ok {
		config.PadTokenID = int64(v)
	}

	// Number of decoder layers
	if v, ok := configMap["num_decoder_layers"].(float64); ok {
		config.NumDecoderLayers = int(v)
	} else if v, ok := configMap["num_layers"].(float64); ok {
		config.NumDecoderLayers = int(v)
	}

	// Number of attention heads
	if v, ok := configMap["num_heads"].(float64); ok {
		config.NumHeads = int(v)
	} else if v, ok := configMap["num_attention_heads"].(float64); ok {
		config.NumHeads = int(v)
	}

	// Head dimension
	if v, ok := configMap["d_kv"].(float64); ok {
		config.HeadDim = int(v)
	} else if v, ok := configMap["head_dim"].(float64); ok {
		config.HeadDim = int(v)
	}

	// Hidden size (d_model)
	if v, ok := configMap["d_model"].(float64); ok {
		config.DModel = int(v)
	} else if v, ok := configMap["hidden_size"].(float64); ok {
		config.DModel = int(v)
	}

	// Vocab size
	if v, ok := configMap["vocab_size"].(float64); ok {
		config.VocabSize = int(v)
	}

	return config, nil
}

// TokenizeSeq2SeqInputs tokenizes inputs for seq2seq models.
// It uses the existing TokenizeInputs infrastructure and converts to the seq2seq format.
func TokenizeSeq2SeqInputs(inputs []string, tokenizer *Tokenizer, padTokenID int64) (*Seq2SeqTokenized, error) {
	if tokenizer == nil {
		return nil, errors.New("tokenizer is nil")
	}

	// Use the standard batch tokenization
	batch := NewBatch(len(inputs))
	TokenizeInputs(batch, tokenizer, inputs)

	batchSize := len(inputs)
	tokenIDs := make([][]int64, batchSize)
	attentionMask := make([][]int64, batchSize)

	// Calculate max length from actual token IDs, not from MaxSequenceLength
	// (some tokenizers like T5 don't produce attention masks, so MaxSequenceLength may be wrong)
	maxLen := 0
	for i := 0; i < batchSize; i++ {
		if len(batch.Input[i].TokenIDs) > maxLen {
			maxLen = len(batch.Input[i].TokenIDs)
		}
	}

	if maxLen == 0 {
		return nil, errors.New("no tokens produced by tokenizer")
	}

	// Convert from TokenizedInput to int64 format and pad
	for i := 0; i < batchSize; i++ {
		curLen := len(batch.Input[i].TokenIDs)
		padded := make([]int64, maxLen)
		mask := make([]int64, maxLen)

		for j := 0; j < curLen; j++ {
			padded[j] = int64(batch.Input[i].TokenIDs[j])
			mask[j] = 1
		}
		for j := curLen; j < maxLen; j++ {
			padded[j] = padTokenID
			mask[j] = 0
		}

		tokenIDs[i] = padded
		attentionMask[i] = mask
	}

	return &Seq2SeqTokenized{
		TokenIDs:      tokenIDs,
		AttentionMask: attentionMask,
		MaxLength:     maxLen,
	}, nil
}

// findOnnxFile finds an ONNX file containing any of the given patterns.
func findOnnxFile(modelPath string, patterns ...string) (string, error) {
	onnxFiles, err := getOnnxFiles(modelPath)
	if err != nil {
		return "", err
	}

	for _, pattern := range patterns {
		for _, file := range onnxFiles {
			filename := file[1]
			if strings.Contains(strings.ToLower(filename), pattern) {
				return filename, nil
			}
		}
	}

	return "", fmt.Errorf("no ONNX file found matching patterns %v in %s", patterns, modelPath)
}

// findDecoderOnnxFile finds the decoder ONNX file (not the init decoder).
func findDecoderOnnxFile(modelPath string) (string, error) {
	onnxFiles, err := getOnnxFiles(modelPath)
	if err != nil {
		return "", err
	}

	for _, file := range onnxFiles {
		filename := strings.ToLower(file[1])
		// Match "decoder" but not "init-decoder" or "decoder-init"
		if strings.Contains(filename, "decoder") &&
			!strings.Contains(filename, "init") {
			return file[1], nil
		}
	}

	return "", fmt.Errorf("no decoder ONNX file found in %s", modelPath)
}
