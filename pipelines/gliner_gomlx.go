//go:build !ORT && !ALL

package pipelines

import (
	"errors"
	"fmt"

	"github.com/gomlx/gomlx/pkg/core/tensors"

	"github.com/knights-analytics/hugot/backends"
)

// Note: GLiNER GoMLX implementation exists but is not fully functional due to
// dynamic shape operations (LSTM hidden states, ConstantOfShape nodes) that
// GoMLX cannot handle. The code is kept for future development when GoMLX
// gains dynamic shape support. For production use, use ORT backend.

// createGLiNERTensorsGoMLX creates ALL tensors needed for GLiNER inference using GoMLX
func createGLiNERTensorsGoMLX(batch *GLiNERBatch, model *backends.Model) error {
	batchSize := batch.Size
	maxSeqLen := batch.MaxSequenceLength
	numSpans := batch.NumSpans

	// Prepare all input tensors in the correct order
	inputTensors := make([]*tensors.Tensor, len(model.InputsMeta))

	for i, meta := range model.InputsMeta {
		switch meta.Name {
		case "input_ids":
			// Create input_ids tensor from tokenized input
			backing := make([]int64, batchSize*maxSeqLen)
			for b := 0; b < batchSize; b++ {
				for s := 0; s < maxSeqLen; s++ {
					idx := b*maxSeqLen + s
					if b < len(batch.Input) && s < len(batch.Input[b].TokenIDs) {
						backing[idx] = int64(batch.Input[b].TokenIDs[s])
					}
				}
			}
			inputTensors[i] = tensors.FromFlatDataAndDimensions(backing, batchSize, maxSeqLen)

		case "attention_mask":
			// Create attention_mask tensor from tokenized input
			backing := make([]int64, batchSize*maxSeqLen)
			for b := 0; b < batchSize; b++ {
				for s := 0; s < maxSeqLen; s++ {
					idx := b*maxSeqLen + s
					if b < len(batch.Input) && s < len(batch.Input[b].AttentionMask) {
						backing[idx] = int64(batch.Input[b].AttentionMask[s])
					}
				}
			}
			inputTensors[i] = tensors.FromFlatDataAndDimensions(backing, batchSize, maxSeqLen)

		case "token_type_ids":
			// Create zero tensor for token_type_ids
			backing := make([]int64, batchSize*maxSeqLen)
			inputTensors[i] = tensors.FromFlatDataAndDimensions(backing, batchSize, maxSeqLen)

		case "words_mask":
			// Flatten words_mask: [batch_size, seq_len]
			backing := make([]int64, batchSize*maxSeqLen)
			for b := 0; b < batchSize; b++ {
				for s := 0; s < maxSeqLen; s++ {
					idx := b*maxSeqLen + s
					if b < len(batch.WordsMask) && s < len(batch.WordsMask[b]) {
						backing[idx] = batch.WordsMask[b][s]
					}
				}
			}
			inputTensors[i] = tensors.FromFlatDataAndDimensions(backing, batchSize, maxSeqLen)

		case "text_lengths":
			// Flatten text_lengths: [batch_size, 1]
			backing := make([]int64, batchSize)
			for b := 0; b < batchSize; b++ {
				if b < len(batch.TextLengths) && len(batch.TextLengths[b]) > 0 {
					backing[b] = batch.TextLengths[b][0]
				}
			}
			inputTensors[i] = tensors.FromFlatDataAndDimensions(backing, batchSize, 1)

		case "span_idx":
			// Flatten span_idx: [batch_size, num_spans, 2]
			backing := make([]int64, batchSize*numSpans*2)
			for b := 0; b < batchSize; b++ {
				for s := 0; s < numSpans; s++ {
					baseIdx := (b*numSpans + s) * 2
					if b < len(batch.SpanIdx) && s < len(batch.SpanIdx[b]) && len(batch.SpanIdx[b][s]) >= 2 {
						backing[baseIdx] = batch.SpanIdx[b][s][0]
						backing[baseIdx+1] = batch.SpanIdx[b][s][1]
					}
				}
			}
			inputTensors[i] = tensors.FromFlatDataAndDimensions(backing, batchSize, numSpans, 2)

		case "span_mask":
			// Flatten span_mask: [batch_size, num_spans] - GLiNER expects bool type
			backing := make([]bool, batchSize*numSpans)
			for b := 0; b < batchSize; b++ {
				for s := 0; s < numSpans; s++ {
					idx := b*numSpans + s
					if b < len(batch.SpanMask) && s < len(batch.SpanMask[b]) {
						backing[idx] = batch.SpanMask[b][s] != 0
					}
				}
			}
			inputTensors[i] = tensors.FromFlatDataAndDimensions(backing, batchSize, numSpans)

		default:
			return fmt.Errorf("unknown input meta name %s", meta.Name)
		}
	}

	// Update batch with new tensors
	batch.InputValues = inputTensors

	// Set up destroy function
	batch.DestroyInputs = func() error {
		for _, t := range inputTensors {
			t.FinalizeAll()
		}
		return nil
	}

	return nil
}

// runGLiNERSessionOnBatchGoMLX runs the GLiNER model on a batch using GoMLX (pure Go)
func runGLiNERSessionOnBatchGoMLX(batch *GLiNERBatch, p *backends.BasePipeline) error {
	if p.Model.GoMLXModel == nil || p.Model.GoMLXModel.Exec == nil {
		return errors.New("GoMLX model not initialized")
	}
	inputTensors, ok := batch.InputValues.([]*tensors.Tensor)
	if !ok {
		return errors.New("expected []*tensors.Tensor for input tensors")
	}

	// Run inference
	outputTensors, err := p.Model.GoMLXModel.Exec.Exec(inputTensors)
	if err != nil {
		return fmt.Errorf("running GLiNER session: %w", err)
	}
	defer func() {
		for _, t := range outputTensors {
			t.FinalizeAll()
		}
	}()

	// Convert output to Go slices using shared reshape function
	batch.OutputValues = make([]any, len(outputTensors))
	for i, t := range outputTensors {
		var rawOutput []float32
		tensors.ConstFlatData(t, func(flat []float32) {
			rawOutput = make([]float32, len(flat))
			copy(rawOutput, flat)
		})
		dims := p.Model.OutputsMeta[i].Dimensions

		// Compute dimensions and reshape using shared function
		numWords, maxWidth, numClasses := computeGLiNEROutputDimensions(batch, dims, len(rawOutput))
		batch.OutputValues[i] = reshapeGLiNEROutput(rawOutput, batch.Size, numWords, maxWidth, numClasses)
	}

	return nil
}
