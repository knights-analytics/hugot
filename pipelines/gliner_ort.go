//go:build ORT || ALL

package pipelines

import (
	"errors"
	"fmt"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/knights-analytics/hugot/backends"
)

// createGLiNERTensorsORT creates ALL tensors needed for GLiNER inference
func createGLiNERTensorsORT(batch *GLiNERBatch, model *backends.Model) error {
	batchSize := int64(batch.Size)
	maxSeqLen := int64(batch.MaxSequenceLength)
	numSpans := int64(batch.NumSpans)

	// Prepare all input tensors in the correct order
	inputTensors := make([]ort.Value, len(model.InputsMeta))
	var destroyFuncs []func() error

	for i, meta := range model.InputsMeta {
		var tensor ort.Value
		var err error

		switch meta.Name {
		case "input_ids":
			// Create input_ids tensor from tokenized input
			backing := make([]int64, batchSize*maxSeqLen)
			for b := 0; b < int(batchSize); b++ {
				for s := 0; s < int(maxSeqLen); s++ {
					idx := b*int(maxSeqLen) + s
					if b < len(batch.Input) && s < len(batch.Input[b].TokenIDs) {
						backing[idx] = int64(batch.Input[b].TokenIDs[s])
					}
				}
			}
			tensor, err = ort.NewTensor(ort.NewShape(batchSize, maxSeqLen), backing)
			if err != nil {
				return fmt.Errorf("creating input_ids tensor: %w", err)
			}

		case "attention_mask":
			// Create attention_mask tensor from tokenized input
			backing := make([]int64, batchSize*maxSeqLen)
			for b := 0; b < int(batchSize); b++ {
				for s := 0; s < int(maxSeqLen); s++ {
					idx := b*int(maxSeqLen) + s
					if b < len(batch.Input) && s < len(batch.Input[b].AttentionMask) {
						backing[idx] = int64(batch.Input[b].AttentionMask[s])
					}
				}
			}
			tensor, err = ort.NewTensor(ort.NewShape(batchSize, maxSeqLen), backing)
			if err != nil {
				return fmt.Errorf("creating attention_mask tensor: %w", err)
			}

		case "token_type_ids":
			// Create zero tensor for token_type_ids (not used by GLiNER but may be required)
			backing := make([]int64, batchSize*maxSeqLen)
			tensor, err = ort.NewTensor(ort.NewShape(batchSize, maxSeqLen), backing)
			if err != nil {
				return fmt.Errorf("creating token_type_ids tensor: %w", err)
			}

		case "words_mask":
			// Flatten words_mask: [batch_size, seq_len]
			backing := make([]int64, batchSize*maxSeqLen)
			for b := 0; b < int(batchSize); b++ {
				for s := 0; s < int(maxSeqLen); s++ {
					idx := b*int(maxSeqLen) + s
					if b < len(batch.WordsMask) && s < len(batch.WordsMask[b]) {
						backing[idx] = batch.WordsMask[b][s]
					}
				}
			}
			tensor, err = ort.NewTensor(ort.NewShape(batchSize, maxSeqLen), backing)
			if err != nil {
				return fmt.Errorf("creating words_mask tensor: %w", err)
			}

		case "text_lengths":
			// Flatten text_lengths: [batch_size, 1]
			backing := make([]int64, batchSize)
			for b := 0; b < int(batchSize); b++ {
				if b < len(batch.TextLengths) && len(batch.TextLengths[b]) > 0 {
					backing[b] = batch.TextLengths[b][0]
				}
			}
			tensor, err = ort.NewTensor(ort.NewShape(batchSize, 1), backing)
			if err != nil {
				return fmt.Errorf("creating text_lengths tensor: %w", err)
			}

		case "span_idx":
			// Flatten span_idx: [batch_size, num_spans, 2]
			backing := make([]int64, batchSize*numSpans*2)
			for b := 0; b < int(batchSize); b++ {
				for s := 0; s < int(numSpans); s++ {
					baseIdx := (b*int(numSpans) + s) * 2
					if b < len(batch.SpanIdx) && s < len(batch.SpanIdx[b]) && len(batch.SpanIdx[b][s]) >= 2 {
						backing[baseIdx] = batch.SpanIdx[b][s][0]
						backing[baseIdx+1] = batch.SpanIdx[b][s][1]
					}
				}
			}
			tensor, err = ort.NewTensor(ort.NewShape(batchSize, numSpans, 2), backing)
			if err != nil {
				return fmt.Errorf("creating span_idx tensor: %w", err)
			}

		case "span_mask":
			// Flatten span_mask: [batch_size, num_spans] - GLiNER expects bool type
			backing := make([]bool, batchSize*numSpans)
			for b := 0; b < int(batchSize); b++ {
				for s := 0; s < int(numSpans); s++ {
					idx := b*int(numSpans) + s
					if b < len(batch.SpanMask) && s < len(batch.SpanMask[b]) {
						backing[idx] = batch.SpanMask[b][s] != 0
					}
				}
			}
			tensor, err = ort.NewTensor(ort.NewShape(batchSize, numSpans), backing)
			if err != nil {
				return fmt.Errorf("creating span_mask tensor: %w", err)
			}

		default:
			return fmt.Errorf("unknown input meta name %s", meta.Name)
		}

		inputTensors[i] = tensor
		destroyFuncs = append(destroyFuncs, tensor.Destroy)
	}

	// Update batch with new tensors
	batch.InputValues = inputTensors

	// Set up destroy function
	batch.DestroyInputs = func() error {
		var errs []error
		for _, f := range destroyFuncs {
			if err := f(); err != nil {
				errs = append(errs, err)
			}
		}
		return errors.Join(errs...)
	}

	return nil
}

// runGLiNERSessionOnBatchORT runs the GLiNER model on a batch using ORT
func runGLiNERSessionOnBatchORT(batch *GLiNERBatch, p *backends.BasePipeline) error {
	inputTensors, ok := batch.InputValues.([]ort.Value)
	if !ok {
		return errors.New("expected []ort.Value for input tensors")
	}

	// Find logits output index
	var logitsIndex int = -1
	for i, meta := range p.Model.OutputsMeta {
		if meta.Name == "logits" {
			logitsIndex = i
			break
		}
	}
	if logitsIndex < 0 {
		return errors.New("logits output not found in model outputs")
	}

	// Create output tensor slice - DynamicAdvancedSession will allocate tensors
	outputTensors := make([]ort.Value, len(p.Model.OutputsMeta))

	// Run inference - the dynamic session allocates output tensors automatically
	if err := p.Model.ORTModel.Session.Run(inputTensors, outputTensors); err != nil {
		for _, t := range outputTensors {
			if t != nil {
				t.Destroy()
			}
		}
		return fmt.Errorf("running GLiNER session: %w", err)
	}

	// Extract logits data - it's the tensor at logitsIndex
	logitsTensor, ok := outputTensors[logitsIndex].(*ort.Tensor[float32])
	if !ok {
		for _, t := range outputTensors {
			if t != nil {
				t.Destroy()
			}
		}
		return errors.New("logits tensor has unexpected type")
	}

	// Get the actual shape from the output tensor
	shape := logitsTensor.GetShape()
	actualMaxWords := int(shape[1])
	maxWidth := int(shape[2])
	actualNumClasses := int(shape[3])

	// Convert output to Go slices using shared reshape function
	data := logitsTensor.GetData()
	batch.OutputValues = []any{reshapeGLiNEROutput(data, int(batch.Size), actualMaxWords, maxWidth, actualNumClasses)}

	// Clean up all output tensors
	for _, t := range outputTensors {
		if t != nil {
			t.Destroy()
		}
	}

	return nil
}

// Stub functions for GoMLX - these are only called if the wrong runtime is configured,
// which should be caught earlier, but the compiler needs them defined.

func createGLiNERTensorsGoMLX(batch *GLiNERBatch, model *backends.Model) error {
	return errors.New("GoMLX runtime not available: build with GO or XLA tags instead of ORT")
}

func runGLiNERSessionOnBatchGoMLX(batch *GLiNERBatch, p *backends.BasePipeline) error {
	return errors.New("GoMLX runtime not available: build with GO or XLA tags instead of ORT")
}
