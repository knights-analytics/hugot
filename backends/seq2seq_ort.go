//go:build ORT || ALL

package backends

import (
	"errors"
	"fmt"
	"math/rand"
	"sort"
	"strings"
	"time"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/knights-analytics/hugot/util/vectorutil"
)

// RunSeq2SeqEncoderORT runs the encoder model on the input tokens using ORT backend.
func RunSeq2SeqEncoder(batch Seq2SeqBatchInterface, model *Model, runtime string) error {
	if runtime != "ORT" {
		return fmt.Errorf("unsupported runtime for seq2seq encoder: %s", runtime)
	}

	return runSeq2SeqEncoderORT(batch, model)
}

func runSeq2SeqEncoderORT(batch Seq2SeqBatchInterface, model *Model) error {
	batchSize := batch.GetSize()
	seqLen := batch.GetMaxInputLength()
	inputIDs := batch.GetInputTokenIDs()
	attentionMask := batch.GetInputAttentionMask()

	// Flatten input tensors
	flatInputIDs := make([]int64, batchSize*seqLen)
	flatAttentionMask := make([]int64, batchSize*seqLen)
	for i := 0; i < batchSize; i++ {
		for j := 0; j < seqLen; j++ {
			flatInputIDs[i*seqLen+j] = inputIDs[i][j]
			flatAttentionMask[i*seqLen+j] = attentionMask[i][j]
		}
	}

	// Create input tensors
	inputIDsTensor, err := ort.NewTensor(ort.NewShape(int64(batchSize), int64(seqLen)), flatInputIDs)
	if err != nil {
		return fmt.Errorf("creating input_ids tensor: %w", err)
	}

	attentionMaskTensor, err := ort.NewTensor(ort.NewShape(int64(batchSize), int64(seqLen)), flatAttentionMask)
	if err != nil {
		inputIDsTensor.Destroy()
		return fmt.Errorf("creating attention_mask tensor: %w", err)
	}

	// Find output shape from model metadata
	// Encoder output is typically (batch, seq_len, hidden_size)
	var hiddenSize int64 = 512 // Default for T5-small
	for _, meta := range model.OutputsMeta {
		if strings.Contains(meta.Name, "hidden_states") || meta.Name == "last_hidden_state" {
			dims := meta.Dimensions
			if len(dims) >= 3 {
				hiddenSize = dims[2]
				if hiddenSize == -1 {
					hiddenSize = 512 // Dynamic, use default
				}
			}
			break
		}
	}

	// Create output tensor
	outputShape := ort.NewShape(int64(batchSize), int64(seqLen), hiddenSize)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		inputIDsTensor.Destroy()
		attentionMaskTensor.Destroy()
		return fmt.Errorf("creating output tensor: %w", err)
	}

	// Run encoder
	inputs := []ort.Value{inputIDsTensor, attentionMaskTensor}
	outputs := []ort.Value{outputTensor}

	if err := model.ORTModel.Session.Run(inputs, outputs); err != nil {
		inputIDsTensor.Destroy()
		attentionMaskTensor.Destroy()
		outputTensor.Destroy()
		return fmt.Errorf("running encoder: %w", err)
	}

	// Store encoder outputs in batch (keep tensors alive for decoder)
	batch.SetEncoderHiddenStates(outputTensor)
	batch.SetEncoderAttentionMask(attentionMaskTensor)

	// Set cleanup function
	batch.SetDestroyEncoder(func() error {
		var errs []error
		errs = append(errs, inputIDsTensor.Destroy())
		// Note: outputTensor and attentionMaskTensor are destroyed after generation
		return errors.Join(errs...)
	})

	return nil
}

// tokenSelector is a function type for selecting the next token from logits.
// This allows the generation loop to be reused for both greedy and sampling strategies.
type tokenSelector func(logits []float32, batchSize, vocabSize int) []int64

// RunSeq2SeqGenerationGreedy performs greedy decoding.
func RunSeq2SeqGenerationGreedy(batch Seq2SeqBatchInterface, pipeline Seq2SeqPipelineInterface) error {
	if pipeline.GetRuntime() != "ORT" {
		return fmt.Errorf("unsupported runtime: %s", pipeline.GetRuntime())
	}

	// Greedy selection: argmax over vocabulary
	selector := func(logits []float32, batchSize, vocabSize int) []int64 {
		return argmaxSeq2Seq(logits, batchSize, vocabSize)
	}

	return runSeq2SeqGenerationORT(batch, pipeline, selector)
}

// RunSeq2SeqGenerationSampling performs top-p sampling with temperature.
func RunSeq2SeqGenerationSampling(batch Seq2SeqBatchInterface, pipeline Seq2SeqPipelineInterface) error {
	if pipeline.GetRuntime() != "ORT" {
		return fmt.Errorf("unsupported runtime: %s", pipeline.GetRuntime())
	}

	topP := pipeline.GetTopP()
	temperature := pipeline.GetTemperature()

	// Sampling selection: top-p with temperature
	selector := func(logits []float32, batchSize, vocabSize int) []int64 {
		return sampleTopP(logits, batchSize, vocabSize, topP, temperature)
	}

	return runSeq2SeqGenerationORT(batch, pipeline, selector)
}

// runSeq2SeqGenerationORT is the unified generation loop for ORT backend.
// It uses the provided tokenSelector to choose the next token at each step.
func runSeq2SeqGenerationORT(batch Seq2SeqBatchInterface, pipeline Seq2SeqPipelineInterface, selectTokens tokenSelector) error {
	batchSize := batch.GetSize()
	maxNewTokens := pipeline.GetMaxNewTokens()
	eosTokenIDs := pipeline.GetEosTokenIDs()
	decoderStartToken := pipeline.GetDecoderStartTokenID()
	vocabSize := pipeline.GetVocabSize()

	// Initialize generation state
	generatedTokens := make([][]int64, batchSize)
	finished := make([]bool, batchSize)
	finishedCount := 0
	for i := range generatedTokens {
		generatedTokens[i] = []int64{}
	}

	// Get encoder outputs
	encoderHiddenStates := batch.GetEncoderHiddenStates().(ort.Value)
	encoderAttentionMask := batch.GetEncoderAttentionMask().(ort.Value)

	// First step: use decoder-init (no past_key_values)
	decoderInputIDs := make([]int64, batchSize)
	for i := range decoderInputIDs {
		decoderInputIDs[i] = decoderStartToken
	}

	decoderInitModel := pipeline.GetDecoderInitModel()
	decoderModel := pipeline.GetDecoderModel()

	// Encoder-decoder models have separate PKV for cross-attention (encoder) and self-attention (decoder).
	// decoder-init outputs all 32 PKV: 16 encoder + 16 decoder
	// decoder outputs only 16 PKV: decoder only (encoder PKV doesn't change)
	// decoder expects all 32 PKV as input
	// So we need to keep encoder PKV constant and only update decoder PKV.
	var encoderPKV []ort.Value // Cross-attention KV (constant after decoder-init)
	var decoderPKV []ort.Value // Self-attention KV (updated each step)

	encoderSeqLen := batch.GetMaxInputLength()

	for step := 0; step < maxNewTokens; step++ {
		if finishedCount == batchSize {
			break
		}

		var logits []float32
		var err error

		if step == 0 {
			// First step: use decoder-init, get all PKV
			var allPKV []ort.Value
			logits, allPKV, err = runDecoderInitStepORT(
				decoderInputIDs, encoderHiddenStates, encoderAttentionMask,
				decoderInitModel, batchSize, vocabSize, encoderSeqLen,
			)
			if err != nil {
				return err
			}

			// Split PKV: encoder PKV (constant) vs decoder PKV (will be updated)
			encoderPKV, decoderPKV = splitEncoderDecoderPKV(allPKV, decoderInitModel)
		} else {
			// Subsequent steps: use decoder with past_key_values
			combinedPKV := combineEncoderDecoderPKV(encoderPKV, decoderPKV, decoderModel)

			var newDecoderPKV []ort.Value
			logits, newDecoderPKV, err = runDecoderStepORT(
				decoderInputIDs, encoderAttentionMask,
				combinedPKV, decoderModel, batchSize, vocabSize, step, encoderSeqLen,
			)
			if err != nil {
				// Cleanup on error
				for _, pkv := range encoderPKV {
					pkv.Destroy()
				}
				for _, pkv := range decoderPKV {
					pkv.Destroy()
				}
				return err
			}

			// Destroy old decoder PKV (encoder PKV is kept constant)
			for _, pkv := range decoderPKV {
				pkv.Destroy()
			}
			decoderPKV = newDecoderPKV
		}

		// Select next tokens using the provided strategy (greedy or sampling)
		nextTokens := selectTokens(logits, batchSize, vocabSize)

		// Update decoder input for next step
		decoderInputIDs = nextTokens

		// Append tokens and check for EOS
		for i := 0; i < batchSize; i++ {
			if finished[i] {
				continue
			}

			tok := nextTokens[i]
			generatedTokens[i] = append(generatedTokens[i], tok)

			if eosTokenIDs[tok] {
				finished[i] = true
				finishedCount++
			}
		}
	}

	// Store results
	batch.SetGeneratedTokens(generatedTokens)
	batch.SetFinished(finished)
	batch.SetFinishedCount(finishedCount)

	// Set cleanup for decoder resources
	batch.SetDestroyDecoder(func() error {
		var errs []error
		for _, pkv := range encoderPKV {
			errs = append(errs, pkv.Destroy())
		}
		for _, pkv := range decoderPKV {
			errs = append(errs, pkv.Destroy())
		}
		// Also destroy encoder outputs now
		if hs, ok := batch.GetEncoderHiddenStates().(ort.Value); ok {
			errs = append(errs, hs.Destroy())
		}
		if am, ok := batch.GetEncoderAttentionMask().(ort.Value); ok {
			errs = append(errs, am.Destroy())
		}
		return errors.Join(errs...)
	})

	return nil
}

// runDecoderInitStepORT runs the initial decoder step (no past_key_values).
func runDecoderInitStepORT(
	decoderInputIDs []int64,
	encoderHiddenStates, encoderAttentionMask ort.Value,
	model *Model,
	batchSize, vocabSize, encoderSeqLen int,
) ([]float32, []ort.Value, error) {
	// Create decoder input tensor
	inputTensor, err := ort.NewTensor(ort.NewShape(int64(batchSize), 1), decoderInputIDs)
	if err != nil {
		return nil, nil, fmt.Errorf("creating decoder input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	// Build inputs based on model's expected inputs order:
	// [0] encoder_attention_mask, [1] input_ids (decoder), [2] encoder_hidden_states
	inputs := []ort.Value{encoderAttentionMask, inputTensor, encoderHiddenStates}

	// Create output tensors
	// Output 0: logits (batch, 1, vocab_size)
	// Outputs 1+: past_key_values (encoder and decoder)
	numOutputs := len(model.OutputsMeta)
	outputs := make([]ort.Value, numOutputs)

	// Logits output
	logitsTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(int64(batchSize), 1, int64(vocabSize)))
	if err != nil {
		return nil, nil, fmt.Errorf("creating logits tensor: %w", err)
	}
	outputs[0] = logitsTensor

	// Past key values outputs - infer shapes from model metadata
	// For encoder-decoder models:
	//   - present.X.decoder.key/value use decoder sequence length (1 for init)
	//   - present.X.encoder.key/value use encoder sequence length
	decoderSeqLen := 1 // Initial decoder step
	for i := 1; i < numOutputs; i++ {
		meta := model.OutputsMeta[i]
		shape := inferPKVShapeWithSeqLen(meta.Dimensions, meta.Name, batchSize, decoderSeqLen, encoderSeqLen)
		pkvTensor, err := ort.NewEmptyTensor[float32](shape)
		if err != nil {
			// Cleanup created tensors
			for j := 0; j < i; j++ {
				outputs[j].Destroy()
			}
			return nil, nil, fmt.Errorf("creating pkv tensor %d (%s): %w", i, meta.Name, err)
		}
		outputs[i] = pkvTensor
	}

	// Run decoder-init
	if err := model.ORTModel.Session.Run(inputs, outputs); err != nil {
		for _, o := range outputs {
			o.Destroy()
		}
		return nil, nil, fmt.Errorf("running decoder-init: %w", err)
	}

	// Extract logits
	logits := outputs[0].(*ort.Tensor[float32]).GetData()

	// Keep past_key_values (don't destroy them)
	pastKeyValues := outputs[1:]

	// Destroy logits tensor (we copied the data)
	outputs[0].Destroy()

	return logits, pastKeyValues, nil
}

// runDecoderStepORT runs a decoder step with past_key_values.
// pastSeqLen is the current decoder sequence length (number of tokens generated so far).
// encoderSeqLen is needed to keep encoder PKV at the correct shape.
func runDecoderStepORT(
	decoderInputIDs []int64,
	encoderAttentionMask ort.Value,
	pastKeyValues []ort.Value,
	model *Model,
	batchSize, vocabSize, pastSeqLen, encoderSeqLen int,
) ([]float32, []ort.Value, error) {
	// Create decoder input tensor
	inputTensor, err := ort.NewTensor(ort.NewShape(int64(batchSize), 1), decoderInputIDs)
	if err != nil {
		return nil, nil, fmt.Errorf("creating decoder input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	// Build inputs (decoder with past): encoder_attention_mask, input_ids, pkv_0, pkv_1, ...
	// Note: encoder_hidden_states is NOT needed since encoder KV cache is in past_key_values
	inputs := make([]ort.Value, 2+len(pastKeyValues))
	inputs[0] = encoderAttentionMask
	inputs[1] = inputTensor
	for i, pkv := range pastKeyValues {
		inputs[2+i] = pkv
	}

	// Create output tensors
	// Decoder outputs: logits + 16 decoder PKV only (not encoder PKV)
	numOutputs := len(model.OutputsMeta)
	outputs := make([]ort.Value, numOutputs)

	// Logits output
	logitsTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(int64(batchSize), 1, int64(vocabSize)))
	if err != nil {
		return nil, nil, fmt.Errorf("creating logits tensor: %w", err)
	}
	outputs[0] = logitsTensor

	// Present key values outputs (decoder only)
	// Decoder output PKV seq_len = past_seq_len + 1 (we're adding one more token)
	newDecoderSeqLen := pastSeqLen + 1
	for i := 1; i < numOutputs; i++ {
		meta := model.OutputsMeta[i]
		shape := inferPKVShapeFixed(meta.Dimensions, batchSize, newDecoderSeqLen)
		pkvTensor, err := ort.NewEmptyTensor[float32](shape)
		if err != nil {
			for j := 0; j < i; j++ {
				outputs[j].Destroy()
			}
			return nil, nil, fmt.Errorf("creating present tensor %d: %w", i, err)
		}
		outputs[i] = pkvTensor
	}

	// Run decoder
	if err := model.ORTModel.Session.Run(inputs, outputs); err != nil {
		for _, o := range outputs {
			o.Destroy()
		}
		return nil, nil, fmt.Errorf("running decoder: %w", err)
	}

	// Extract logits
	logits := outputs[0].(*ort.Tensor[float32]).GetData()

	// Keep present key values (decoder only, will be combined with encoder PKV later)
	presentKeyValues := outputs[1:]

	outputs[0].Destroy()

	return logits, presentKeyValues, nil
}

// inferPKVShape infers the shape for past/present key values tensors.
// NOTE: This is the legacy function that assumes seq_len=1. Use inferPKVShapeWithSeqLen for encoder-decoder models.
func inferPKVShape(dims Shape, batchSize int) ort.Shape {
	shape := make([]int64, len(dims))
	for i, d := range dims {
		if d == -1 {
			// Dynamic dimension - assume batch size for first, seq len for later
			if i == 0 {
				shape[i] = int64(batchSize)
			} else {
				shape[i] = 1 // Sequence length starts at 1
			}
		} else {
			shape[i] = d
		}
	}
	return ort.NewShape(shape...)
}

// inferPKVShapeWithSeqLen infers the shape for encoder-decoder PKV tensors.
// For encoder-decoder models (T5, BART, etc.), the output names indicate whether
// the tensor is for cross-attention (encoder) or self-attention (decoder):
//   - present.X.encoder.key/value -> use encoderSeqLen
//   - present.X.decoder.key/value -> use decoderSeqLen
func inferPKVShapeWithSeqLen(dims Shape, outputName string, batchSize, decoderSeqLen, encoderSeqLen int) ort.Shape {
	// Determine which sequence length to use based on output name
	isEncoderKV := strings.Contains(outputName, ".encoder.")
	seqLen := decoderSeqLen
	if isEncoderKV {
		seqLen = encoderSeqLen
	}

	shape := make([]int64, len(dims))
	for i, d := range dims {
		if d == -1 {
			// Dynamic dimension
			if i == 0 {
				shape[i] = int64(batchSize)
			} else if i == 2 {
				// Sequence length dimension (typically position 2 in [batch, heads, seq, head_dim])
				shape[i] = int64(seqLen)
			} else {
				// Unknown dynamic dim - use 1 as fallback
				shape[i] = 1
			}
		} else {
			shape[i] = d
		}
	}
	return ort.NewShape(shape...)
}

// inferPKVShapeFixed infers PKV shape with a fixed sequence length.
// Used for decoder-only outputs where we know the exact sequence length.
func inferPKVShapeFixed(dims Shape, batchSize, seqLen int) ort.Shape {
	shape := make([]int64, len(dims))
	for i, d := range dims {
		if d == -1 {
			// Dynamic dimension
			if i == 0 {
				shape[i] = int64(batchSize)
			} else if i == 2 {
				// Sequence length dimension
				shape[i] = int64(seqLen)
			} else {
				// Unknown dynamic dim - use 1 as fallback
				shape[i] = 1
			}
		} else {
			shape[i] = d
		}
	}
	return ort.NewShape(shape...)
}

// splitEncoderDecoderPKV splits the PKV outputs from decoder-init into encoder and decoder PKV.
// decoder-init outputs are ordered as: present.0.decoder.key, present.0.decoder.value,
// present.0.encoder.key, present.0.encoder.value, present.1.decoder.key, ...
// Returns (encoderPKV, decoderPKV) where:
// - encoderPKV contains cross-attention KV (constant throughout generation)
// - decoderPKV contains self-attention KV (updated each step)
func splitEncoderDecoderPKV(allPKV []ort.Value, decoderInitModel *Model) (encoderPKV, decoderPKV []ort.Value) {
	// Analyze output names to determine which are encoder vs decoder PKV
	// Skip output 0 (logits)
	for i, pkv := range allPKV {
		// Output index in model is i+1 (since we skipped logits at index 0)
		outputIdx := i + 1
		if outputIdx >= len(decoderInitModel.OutputsMeta) {
			continue
		}

		outputName := decoderInitModel.OutputsMeta[outputIdx].Name
		if strings.Contains(outputName, ".encoder.") {
			encoderPKV = append(encoderPKV, pkv)
		} else {
			decoderPKV = append(decoderPKV, pkv)
		}
	}

	return encoderPKV, decoderPKV
}

// combineEncoderDecoderPKV combines encoder and decoder PKV in the order expected by decoder input.
// The decoder expects PKV in the order matching its input names:
// past_key_values.0.decoder.key, past_key_values.0.decoder.value,
// past_key_values.0.encoder.key, past_key_values.0.encoder.value, ...
func combineEncoderDecoderPKV(encoderPKV, decoderPKV []ort.Value, decoderModel *Model) []ort.Value {
	// Decoder inputs (after encoder_attention_mask and input_ids) are the PKV tensors
	// We need to match them in the exact order the model expects
	numPKVInputs := len(decoderModel.InputsMeta) - 2 // Subtract encoder_attention_mask and input_ids
	result := make([]ort.Value, numPKVInputs)

	// Track which encoder/decoder PKV we've used
	encIdx := 0
	decIdx := 0

	// Skip the first 2 inputs (encoder_attention_mask, input_ids)
	for i := 2; i < len(decoderModel.InputsMeta); i++ {
		inputName := decoderModel.InputsMeta[i].Name
		resultIdx := i - 2

		if strings.Contains(inputName, ".encoder.") {
			if encIdx < len(encoderPKV) {
				result[resultIdx] = encoderPKV[encIdx]
				encIdx++
			}
		} else {
			if decIdx < len(decoderPKV) {
				result[resultIdx] = decoderPKV[decIdx]
				decIdx++
			}
		}
	}

	return result
}

// argmaxSeq2Seq performs argmax over the last dimension of logits.
func argmaxSeq2Seq(logits []float32, batchSize, vocabSize int) []int64 {
	tokens := make([]int64, batchSize)

	for b := 0; b < batchSize; b++ {
		// Logits are (batch, 1, vocab_size), so take last position
		offset := b * vocabSize
		maxIdx := 0
		maxVal := logits[offset]

		for v := 1; v < vocabSize; v++ {
			if logits[offset+v] > maxVal {
				maxVal = logits[offset+v]
				maxIdx = v
			}
		}
		tokens[b] = int64(maxIdx)
	}

	return tokens
}

// sampleTopP performs nucleus (top-p) sampling with temperature.
// Creates a thread-local RNG seeded with current time for non-deterministic sampling.
func sampleTopP(logits []float32, batchSize, vocabSize int, topP, temperature float32) []int64 {
	tokens := make([]int64, batchSize)

	// Create a thread-local RNG for this sampling operation
	rng := rand.New(rand.NewSource(time.Now().UnixNano())) // #nosec G404 -- not used for crypto

	for b := 0; b < batchSize; b++ {
		offset := b * vocabSize
		batchLogits := logits[offset : offset+vocabSize]

		// Apply temperature
		scaledLogits := make([]float32, vocabSize)
		for i := range scaledLogits {
			scaledLogits[i] = batchLogits[i] / temperature
		}

		// Softmax
		probs := vectorutil.SoftMax(scaledLogits)

		// Sort by probability (descending)
		type tokenProb struct {
			idx  int
			prob float32
		}
		sorted := make([]tokenProb, vocabSize)
		for i := range sorted {
			sorted[i] = tokenProb{i, probs[i]}
		}
		sort.Slice(sorted, func(i, j int) bool {
			return sorted[i].prob > sorted[j].prob
		})

		// Find cutoff for top-p
		cumSum := float32(0)
		cutoff := 0
		for i, tp := range sorted {
			cumSum += tp.prob
			if cumSum >= topP {
				cutoff = i + 1
				break
			}
		}
		if cutoff == 0 {
			cutoff = 1
		}

		// Renormalize
		topTokens := sorted[:cutoff]
		totalProb := float32(0)
		for _, tp := range topTokens {
			totalProb += tp.prob
		}

		// Sample using thread-safe random source
		r := rng.Float32() * totalProb
		cumSum = 0
		selectedIdx := topTokens[0].idx
		for _, tp := range topTokens {
			cumSum += tp.prob
			if cumSum >= r {
				selectedIdx = tp.idx
				break
			}
		}

		tokens[b] = int64(selectedIdx)
	}

	return tokens
}
