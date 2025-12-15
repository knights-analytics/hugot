//go:build !ORT && !ALL

package backends

import (
	"errors"
	"fmt"
	"math/rand"
	"strings"
	"time"

	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gopjrt/dtypes"

	"github.com/knights-analytics/hugot/util/vectorutil"
)

// RunSeq2SeqEncoder runs the encoder model on the input batch using GoMLX backend.
func RunSeq2SeqEncoder(batch Seq2SeqBatchInterface, model *Model, runtime string) error {
	if model == nil || model.GoMLXModel == nil {
		return errors.New("encoder model not loaded for GoMLX backend")
	}

	batchSize := batch.GetSize()
	inputTokenIDs := batch.GetInputTokenIDs()
	inputAttentionMask := batch.GetInputAttentionMask()
	maxLen := batch.GetMaxInputLength()

	// Create input tensors
	inputIDsFlat := make([]int64, batchSize*maxLen)
	attentionMaskFlat := make([]int64, batchSize*maxLen)

	for i := 0; i < batchSize; i++ {
		for j := 0; j < maxLen; j++ {
			idx := i*maxLen + j
			if j < len(inputTokenIDs[i]) {
				inputIDsFlat[idx] = inputTokenIDs[i][j]
				attentionMaskFlat[idx] = inputAttentionMask[i][j]
			}
		}
	}

	inputIDsTensor := tensors.FromFlatDataAndDimensions(inputIDsFlat, batchSize, maxLen)
	attentionMaskTensor := tensors.FromFlatDataAndDimensions(attentionMaskFlat, batchSize, maxLen)

	// Execute encoder
	outputs, err := model.GoMLXModel.Exec.Exec(inputIDsTensor, attentionMaskTensor)
	if err != nil {
		inputIDsTensor.FinalizeAll()
		attentionMaskTensor.FinalizeAll()
		return fmt.Errorf("encoder execution failed: %w", err)
	}

	if len(outputs) == 0 {
		inputIDsTensor.FinalizeAll()
		attentionMaskTensor.FinalizeAll()
		return errors.New("encoder returned no outputs")
	}

	// Move encoder outputs to local storage so they can be used by decoder
	// (which has a different backend instance)
	if err := outputs[0].ToLocal(); err != nil {
		inputIDsTensor.FinalizeAll()
		attentionMaskTensor.FinalizeAll()
		return fmt.Errorf("moving encoder hidden states to local: %w", err)
	}
	if err := attentionMaskTensor.ToLocal(); err != nil {
		inputIDsTensor.FinalizeAll()
		outputs[0].FinalizeAll()
		return fmt.Errorf("moving attention mask to local: %w", err)
	}

	// Store encoder hidden states for decoder use
	batch.SetEncoderHiddenStates(outputs[0])
	batch.SetEncoderAttentionMask(attentionMaskTensor)

	// Set cleanup function for encoder outputs
	batch.SetDestroyEncoder(func() error {
		var errs []error
		errs = append(errs, inputIDsTensor.FinalizeAll())
		for _, out := range outputs {
			errs = append(errs, out.FinalizeAll())
		}
		return errors.Join(errs...)
	})

	return nil
}

// RunSeq2SeqGenerationGreedy performs greedy decoding using GoMLX backend.
func RunSeq2SeqGenerationGreedy(batch Seq2SeqBatchInterface, pipeline Seq2SeqPipelineInterface) error {
	return runSeq2SeqGenerationGoMLX(batch, pipeline, false)
}

// RunSeq2SeqGenerationSampling performs sampling-based decoding using GoMLX backend.
func RunSeq2SeqGenerationSampling(batch Seq2SeqBatchInterface, pipeline Seq2SeqPipelineInterface) error {
	return runSeq2SeqGenerationGoMLX(batch, pipeline, true)
}

// runSeq2SeqGenerationGoMLX is the main generation loop for GoMLX backend.
func runSeq2SeqGenerationGoMLX(batch Seq2SeqBatchInterface, pipeline Seq2SeqPipelineInterface, doSample bool) error {
	decoderInitModel := pipeline.GetDecoderInitModel()
	decoderModel := pipeline.GetDecoderModel()

	if decoderInitModel == nil || decoderInitModel.GoMLXModel == nil {
		return errors.New("decoder-init model not loaded for GoMLX backend")
	}
	if decoderModel == nil || decoderModel.GoMLXModel == nil {
		return errors.New("decoder model not loaded for GoMLX backend")
	}

	batchSize := batch.GetSize()
	maxNewTokens := pipeline.GetMaxNewTokens()
	decoderStartTokenID := pipeline.GetDecoderStartTokenID()
	eosTokenIDs := pipeline.GetEosTokenIDs()
	topP := pipeline.GetTopP()
	temperature := pipeline.GetTemperature()

	// Initialize generation state
	generatedTokens := make([][]int64, batchSize)
	for i := range generatedTokens {
		generatedTokens[i] = make([]int64, 0, maxNewTokens)
	}
	finished := make([]bool, batchSize)
	finishedCount := 0

	// Get encoder outputs
	encoderHiddenStates := batch.GetEncoderHiddenStates().(*tensors.Tensor)
	encoderAttentionMask := batch.GetEncoderAttentionMask().(*tensors.Tensor)

	// Initialize decoder input with start token
	currentIDs := make([]int64, batchSize)
	for i := range currentIDs {
		currentIDs[i] = decoderStartTokenID
	}

	// Track KV cache tensors for cleanup
	// encoderPKV stays constant throughout generation (cross-attention KV)
	// decoderPKV gets updated each step (self-attention KV)
	var encoderPKV []*tensors.Tensor
	var decoderPKV []*tensors.Tensor

	// Generation loop
	for step := 0; step < maxNewTokens && finishedCount < batchSize; step++ {
		// Create input tensor for this step
		inputTensor := tensors.FromFlatDataAndDimensions(currentIDs, batchSize, 1)

		var outputs []*tensors.Tensor
		var err error

		if step == 0 {
			// First step: use decoder-init (no past_key_values)
			// Input order matches ONNX model: encoder_attention_mask, input_ids, encoder_hidden_states
			outputs, err = decoderInitModel.GoMLXModel.Exec.Exec(
				encoderAttentionMask,
				inputTensor,
				encoderHiddenStates,
			)
			// Move attention mask back to local storage after decoder-init uses it,
			// so it can be reused by the decoder model (different backend instance)
			if err == nil {
				if localErr := encoderAttentionMask.ToLocal(); localErr != nil {
					inputTensor.FinalizeAll()
					return fmt.Errorf("moving attention mask to local after step 0: %w", localErr)
				}
			}
		} else {
			// Subsequent steps: use decoder with past_key_values
			// Input order: encoder_attention_mask, input_ids, past_key_values...
			// PKV order: decoder.key, decoder.value, encoder.key, encoder.value (per layer)
			combinedPKV := combineEncoderDecoderPKVGoMLX(encoderPKV, decoderPKV, decoderModel)
			inputs := []any{encoderAttentionMask, inputTensor}
			for _, kv := range combinedPKV {
				inputs = append(inputs, kv)
			}
			outputs, err = decoderModel.GoMLXModel.Exec.Exec(inputs...)
		}

		if err != nil {
			inputTensor.FinalizeAll()
			return fmt.Errorf("decoder step %d failed: %w", step, err)
		}

		if len(outputs) < 1 {
			inputTensor.FinalizeAll()
			return fmt.Errorf("decoder step %d returned no outputs", step)
		}

		// First output is logits - move to local for data extraction
		logits := outputs[0]
		if err := logits.ToLocal(); err != nil {
			inputTensor.FinalizeAll()
			return fmt.Errorf("moving logits to local at step %d: %w", step, err)
		}

		// Update KV cache from remaining outputs
		if len(outputs) > 1 {
			// Move all PKV outputs to local storage
			for i := 1; i < len(outputs); i++ {
				if err := outputs[i].ToLocal(); err != nil {
					inputTensor.FinalizeAll()
					logits.FinalizeAll()
					return fmt.Errorf("moving KV cache tensor %d to local at step %d: %w", i-1, step, err)
				}
			}

			if step == 0 {
				// decoder-init outputs both encoder and decoder PKV
				// Split them: outputs are ordered as [decoder.key, decoder.value, encoder.key, encoder.value] per layer
				encoderPKV, decoderPKV = splitEncoderDecoderPKVGoMLX(outputs[1:], decoderInitModel)
			} else {
				// decoder only outputs decoder PKV (self-attention)
				// Clean up old decoder PKV tensors
				for _, kv := range decoderPKV {
					kv.FinalizeAll()
				}
				// All outputs after logits are decoder PKV
				decoderPKV = make([]*tensors.Tensor, len(outputs)-1)
				for i := 1; i < len(outputs); i++ {
					decoderPKV[i-1] = outputs[i]
				}
				// encoderPKV stays the same (from step 0)
			}
		}

		// Get next tokens from logits
		var nextTokens []int64
		if doSample {
			nextTokens, err = sampleFromLogitsGoMLX(logits, topP, temperature)
		} else {
			nextTokens, err = argmaxFromLogitsGoMLX(logits)
		}
		if err != nil {
			inputTensor.FinalizeAll()
			logits.FinalizeAll()
			return fmt.Errorf("failed to get next tokens at step %d: %w", step, err)
		}

		// Update generated sequences
		for i := 0; i < batchSize; i++ {
			if finished[i] {
				continue
			}

			token := nextTokens[i]
			generatedTokens[i] = append(generatedTokens[i], token)
			currentIDs[i] = token

			// Check for EOS
			if eosTokenIDs[token] {
				finished[i] = true
				finishedCount++
			}
		}

		// Cleanup input tensor (logits kept in KV cache cleanup)
		inputTensor.FinalizeAll()
		logits.FinalizeAll()
	}

	batch.SetGeneratedTokens(generatedTokens)
	batch.SetFinished(finished)
	batch.SetFinishedCount(finishedCount)

	// Set cleanup function for decoder resources
	batch.SetDestroyDecoder(func() error {
		var errs []error
		// Clean up encoder PKV (cross-attention, constant throughout generation)
		for _, kv := range encoderPKV {
			if kv != nil {
				errs = append(errs, kv.FinalizeAll())
			}
		}
		// Clean up decoder PKV (self-attention, updated each step)
		for _, kv := range decoderPKV {
			if kv != nil {
				errs = append(errs, kv.FinalizeAll())
			}
		}
		return errors.Join(errs...)
	})

	return nil
}

// argmaxFromLogitsGoMLX extracts the argmax token ID from logits for each batch item.
func argmaxFromLogitsGoMLX(logits *tensors.Tensor) ([]int64, error) {
	shape := logits.Shape()
	if shape.Rank() < 2 || shape.Rank() > 3 {
		return nil, fmt.Errorf("expected logits rank 2 or 3, got %d", shape.Rank())
	}

	batchSize := shape.Dimensions[0]
	vocabSize := shape.Dimensions[shape.Rank()-1]

	// Extract logits as float32 slice
	var logitsData []float32
	switch shape.DType {
	case dtypes.Float32:
		logitsData = tensors.MustCopyFlatData[float32](logits)
	case dtypes.Float64:
		float64Data := tensors.MustCopyFlatData[float64](logits)
		logitsData = make([]float32, len(float64Data))
		for i, v := range float64Data {
			logitsData[i] = float32(v)
		}
	default:
		return nil, fmt.Errorf("unsupported dtype for logits: %s", shape.DType)
	}

	// Find argmax for each batch item
	tokens := make([]int64, batchSize)
	for batch := 0; batch < batchSize; batch++ {
		var offset int
		if shape.Rank() == 3 {
			seqLen := shape.Dimensions[1]
			offset = batch*seqLen*vocabSize + (seqLen-1)*vocabSize
		} else {
			offset = batch * vocabSize
		}

		maxIdx := 0
		maxVal := logitsData[offset]
		for v := 1; v < vocabSize; v++ {
			if logitsData[offset+v] > maxVal {
				maxVal = logitsData[offset+v]
				maxIdx = v
			}
		}
		tokens[batch] = int64(maxIdx)
	}

	return tokens, nil
}

// sampleFromLogitsGoMLX samples token IDs from logits with temperature and top-p.
// Creates a thread-local RNG seeded with current time for non-deterministic sampling.
func sampleFromLogitsGoMLX(logits *tensors.Tensor, topP, temperature float32) ([]int64, error) {
	shape := logits.Shape()
	if shape.Rank() < 2 || shape.Rank() > 3 {
		return nil, fmt.Errorf("expected logits rank 2 or 3, got %d", shape.Rank())
	}

	batchSize := shape.Dimensions[0]
	vocabSize := shape.Dimensions[shape.Rank()-1]

	// Extract logits as float32 slice
	var logitsData []float32
	switch shape.DType {
	case dtypes.Float32:
		logitsData = tensors.MustCopyFlatData[float32](logits)
	case dtypes.Float64:
		float64Data := tensors.MustCopyFlatData[float64](logits)
		logitsData = make([]float32, len(float64Data))
		for i, v := range float64Data {
			logitsData[i] = float32(v)
		}
	default:
		return nil, fmt.Errorf("unsupported dtype for logits: %s", shape.DType)
	}

	// Create a thread-local RNG for this sampling operation
	rng := rand.New(rand.NewSource(time.Now().UnixNano())) // #nosec G404 -- not used for crypto

	tokens := make([]int64, batchSize)
	for batch := 0; batch < batchSize; batch++ {
		var offset int
		if shape.Rank() == 3 {
			seqLen := shape.Dimensions[1]
			offset = batch*seqLen*vocabSize + (seqLen-1)*vocabSize
		} else {
			offset = batch * vocabSize
		}

		batchLogits := make([]float32, vocabSize)
		copy(batchLogits, logitsData[offset:offset+vocabSize])

		// Apply temperature
		if temperature != 1.0 && temperature > 0 {
			for i := range batchLogits {
				batchLogits[i] /= temperature
			}
		}

		// Convert to probabilities with softmax
		probs := vectorutil.SoftMax(batchLogits)

		// Apply top-p (nucleus) sampling with thread-safe RNG
		tokens[batch] = int64(sampleTopPGoMLX(probs, topP, rng))
	}

	return tokens, nil
}

// sampleTopPGoMLX implements nucleus (top-p) sampling.
// Uses the provided random source for thread-safe random number generation.
func sampleTopPGoMLX(probs []float32, topP float32, rng *rand.Rand) int {
	// Create indexed probabilities
	type indexedProb struct {
		index int
		prob  float32
	}
	indexed := make([]indexedProb, len(probs))
	for i, p := range probs {
		indexed[i] = indexedProb{i, p}
	}

	// Sort by probability descending (simple bubble sort for small vocabs)
	for i := 0; i < len(indexed)-1; i++ {
		for j := i + 1; j < len(indexed); j++ {
			if indexed[j].prob > indexed[i].prob {
				indexed[i], indexed[j] = indexed[j], indexed[i]
			}
		}
	}

	// Find the smallest set of tokens with cumulative probability >= topP
	var cumSum float32
	cutoff := 0
	for i, ip := range indexed {
		cumSum += ip.prob
		if cumSum >= topP {
			cutoff = i + 1
			break
		}
	}
	if cutoff == 0 {
		cutoff = 1
	}

	// Normalize probabilities in the nucleus
	nucleus := indexed[:cutoff]
	var nucSum float32
	for _, ip := range nucleus {
		nucSum += ip.prob
	}

	// Random sampling from nucleus using thread-safe random source
	r := rng.Float32() * nucSum
	var cumProb float32
	for _, ip := range nucleus {
		cumProb += ip.prob
		if r <= cumProb {
			return ip.index
		}
	}

	return nucleus[0].index
}

// NewSeq2SeqRNG creates a new random number generator for seq2seq sampling.
// Each generation call should create its own RNG for thread safety.
// Use a fixed seed for reproducible results, or use time-based seed for variety.
func NewSeq2SeqRNG(seed int64) *rand.Rand {
	return rand.New(rand.NewSource(seed)) // #nosec G404 -- not used for crypto
}

// splitEncoderDecoderPKVGoMLX splits the PKV outputs from decoder-init into encoder and decoder PKV.
// decoder-init outputs are ordered as: present.0.decoder.key, present.0.decoder.value,
// present.0.encoder.key, present.0.encoder.value, present.1.decoder.key, ...
// Returns (encoderPKV, decoderPKV) where:
// - encoderPKV contains cross-attention KV (constant throughout generation)
// - decoderPKV contains self-attention KV (updated each step)
func splitEncoderDecoderPKVGoMLX(allPKV []*tensors.Tensor, decoderInitModel *Model) (encoderPKV, decoderPKV []*tensors.Tensor) {
	// Analyze output names to determine which are encoder vs decoder PKV
	// Output names from decoder-init are like: present.0.decoder.key, present.0.decoder.value,
	// present.0.encoder.key, present.0.encoder.value, ...
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

// combineEncoderDecoderPKVGoMLX combines encoder and decoder PKV in the order expected by decoder input.
// Decoder expects: past_key_values.0.decoder.key, past_key_values.0.decoder.value,
// past_key_values.0.encoder.key, past_key_values.0.encoder.value, ...
func combineEncoderDecoderPKVGoMLX(encoderPKV, decoderPKV []*tensors.Tensor, decoderModel *Model) []*tensors.Tensor {
	// Number of PKV inputs = total inputs - 2 (encoder_attention_mask, input_ids)
	numPKVInputs := len(decoderModel.InputsMeta) - 2
	result := make([]*tensors.Tensor, numPKVInputs)

	encIdx := 0
	decIdx := 0

	// Iterate through decoder input metadata to determine the correct order
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
