//go:build !ORT && !ALL

package backends

import (
	"errors"
	"fmt"

	"github.com/gomlx/gomlx/pkg/core/tensors"
)

// Note: GLiNER GoMLX implementation exists but is not fully functional due to
// dynamic shape operations (LSTM hidden states, ConstantOfShape nodes) that
// GoMLX cannot handle. The code is kept for future development when GoMLX
// gains dynamic shape support. For production use, use ORT backend.

// createGLiNERTensorsGoMLX creates ALL tensors needed for GLiNER inference using GoMLX.
func createGLiNERTensorsGoMLX(batch GLiNERBatchInterface, model *Model) error {
	batchSize := batch.GetSize()
	maxSeqLen := batch.GetMaxSequenceLength()
	numSpans := batch.GetNumSpans()
	input := batch.GetInput()
	wordsMask := batch.GetWordsMask()
	textLengths := batch.GetTextLengths()
	spanIdx := batch.GetSpanIdx()
	spanMask := batch.GetSpanMask()

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
					if b < len(input) && s < len(input[b].TokenIDs) {
						backing[idx] = int64(input[b].TokenIDs[s])
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
					if b < len(input) && s < len(input[b].AttentionMask) {
						backing[idx] = int64(input[b].AttentionMask[s])
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
					if b < len(wordsMask) && s < len(wordsMask[b]) {
						backing[idx] = wordsMask[b][s]
					}
				}
			}
			inputTensors[i] = tensors.FromFlatDataAndDimensions(backing, batchSize, maxSeqLen)

		case "text_lengths":
			// Flatten text_lengths: [batch_size, 1]
			backing := make([]int64, batchSize)
			for b := 0; b < batchSize; b++ {
				if b < len(textLengths) && len(textLengths[b]) > 0 {
					backing[b] = textLengths[b][0]
				}
			}
			inputTensors[i] = tensors.FromFlatDataAndDimensions(backing, batchSize, 1)

		case "span_idx":
			// Flatten span_idx: [batch_size, num_spans, 2]
			backing := make([]int64, batchSize*numSpans*2)
			for b := 0; b < batchSize; b++ {
				for s := 0; s < numSpans; s++ {
					baseIdx := (b*numSpans + s) * 2
					if b < len(spanIdx) && s < len(spanIdx[b]) && len(spanIdx[b][s]) >= 2 {
						backing[baseIdx] = spanIdx[b][s][0]
						backing[baseIdx+1] = spanIdx[b][s][1]
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
					if b < len(spanMask) && s < len(spanMask[b]) {
						backing[idx] = spanMask[b][s] != 0
					}
				}
			}
			inputTensors[i] = tensors.FromFlatDataAndDimensions(backing, batchSize, numSpans)

		default:
			return fmt.Errorf("unknown input meta name %s", meta.Name)
		}
	}

	// Update batch with new tensors
	batch.SetInputValues(inputTensors)

	// Set up destroy function
	batch.SetDestroyInputs(func() error {
		for _, t := range inputTensors {
			t.FinalizeAll()
		}
		return nil
	})

	return nil
}

// runGLiNERSessionGoMLX runs the GLiNER model on a batch using GoMLX (pure Go).
func runGLiNERSessionGoMLX(batch GLiNERBatchInterface, p *BasePipeline) error {
	if p.Model.GoMLXModel == nil || p.Model.GoMLXModel.Exec == nil {
		return errors.New("GoMLX model not initialized")
	}
	inputTensors, ok := batch.GetInputValues().([]*tensors.Tensor)
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
	outputValues := make([]any, len(outputTensors))
	for i, t := range outputTensors {
		var rawOutput []float32
		tensors.ConstFlatData(t, func(flat []float32) {
			rawOutput = make([]float32, len(flat))
			copy(rawOutput, flat)
		})
		dims := p.Model.OutputsMeta[i].Dimensions

		// Compute dimensions and reshape using shared function
		numWords, maxWidth, numClasses := ComputeGLiNEROutputDimensions(batch, dims, len(rawOutput))
		outputValues[i] = ReshapeGLiNEROutput(rawOutput, batch.GetSize(), numWords, maxWidth, numClasses)
	}
	batch.SetOutputValues(outputValues)

	return nil
}

// Stub functions for ORT - these are only called if the wrong runtime is configured,
// which should be caught earlier, but the compiler needs them defined.

func createGLiNERTensorsORT(batch GLiNERBatchInterface, model *Model) error {
	return errors.New("ORT runtime not available: build with ORT or ALL tags")
}

func runGLiNERSessionORT(batch GLiNERBatchInterface, p *BasePipeline) error {
	return errors.New("ORT runtime not available: build with ORT or ALL tags")
}
