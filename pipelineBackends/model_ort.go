//go:build ORT || ALL

package pipelineBackends

import (
	"errors"
	"fmt"
	"os"
	"strings"
	"sync/atomic"
	"time"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/knights-analytics/hugot/options"
)

type ORTModel struct {
	Session        *ort.DynamicAdvancedSession
	SessionOptions *ort.SessionOptions
	Options        *options.OrtOptions
	Destroy        func() error
}

func argmax3D(logits [][][]float32) []int64 {
	batchSize := len(logits)
	if batchSize == 0 {
		return nil
	}

	output := make([]int64, batchSize)
	for i := 0; i < batchSize; i++ {
		seq := logits[i]
		if len(seq) == 0 {
			output[i] = 0
			continue
		}

		last := seq[len(seq)-1]
		if len(last) == 0 {
			output[i] = 0
			continue
		}

		maxIdx := -1
		var maxVal float32
		for j := 0; j < len(last); j++ {
			v := last[j]
			if maxIdx == -1 || v > maxVal {
				maxVal = v
				maxIdx = j
			}
		}

		if maxIdx == -1 {
			output[i] = 0
		} else {
			output[i] = int64(maxIdx)
		}
	}

	return output
}

func createORTModelBackend(model *Model, options *options.Options) error {

	// TODO: currently models with external data can only load from regular filesystems
	cwd, err := os.Getwd()
	if err != nil {
		return err
	}
	pathChanged := false
	if !strings.HasPrefix(model.Path, "s3:") {
		err = os.Chdir(model.Path)
		if err != nil {
			return err
		}
		pathChanged = true
	}

	sessionOptions := options.BackendOptions.(*ort.SessionOptions)

	inputs, outputs, err := loadInputOutputMetaORT(model.OnnxBytes)
	if err != nil {
		return err
	}

	var inputNames []string
	var outputNames []string
	for _, v := range inputs {
		inputNames = append(inputNames, v.Name)
	}
	for _, v := range outputs {
		outputNames = append(outputNames, v.Name)
	}
	session, errSession := ort.NewDynamicAdvancedSessionWithONNXData(
		model.OnnxBytes,
		inputNames,
		outputNames,
		sessionOptions,
	)
	if errSession != nil {
		return errSession
	}

	model.ORTModel = &ORTModel{
		Session:        session,
		SessionOptions: sessionOptions,
		Options:        options.ORTOptions,
		Destroy: func() error {
			return session.Destroy()
		},
	}
	model.InputsMeta = inputs
	model.OutputsMeta = outputs
	if pathChanged {
		err = os.Chdir(cwd)
	}

	// ONNX bytes no longer needed after creating the session
	model.OnnxBytes = nil
	return err
}

func loadInputOutputMetaORT(onnxBytes []byte) ([]InputOutputInfo, []InputOutputInfo, error) {
	inputs, outputs, err := ort.GetInputOutputInfoWithONNXData(onnxBytes)
	if err != nil {
		return nil, nil, err
	}
	return convertORTInputOutputs(inputs), convertORTInputOutputs(outputs), nil
}

func createInputTensorsORT(batch *PipelineBatch, model *Model) error {
	padLeft := len(model.EosTokenIDs) > 0
	batchSize := batch.Size
	maxSequenceLength := batch.MaxSequenceLength
	total := batchSize * maxSequenceLength

	// 1) prepare result containers - now we use all inputs, not filtering
	inputVals := make([]ort.Value, len(model.InputsMeta))
	masks := make([][]bool, batchSize)

	// 2) build each tensor
	for mi, meta := range model.InputsMeta {
		switch {
		case strings.HasPrefix(meta.Name, "past_key"):
			// Create key cache tensor
			cacheTensor, err := createSingleCacheTensorORT(
				batchSize,
				model.NumKeyValueHeads,
				model.FixedCacheSize,
				model.HeadDim,
			)
			if err != nil {
				return err
			}
			inputVals[mi] = cacheTensor

		default:
			// Handle regular input tensors
			backing := make([]int64, total)
			idx := 0
			switch meta.Name {
			case "input_ids":
				for bi, inp := range batch.Input {
					seqLen := len(inp.TokenIDs)
					padLen := maxSequenceLength - seqLen
					maskRow := make([]bool, maxSequenceLength)
					for pos := range maxSequenceLength {
						if padLeft {
							if pos < padLen {
								backing[idx] = model.PadToken
							} else {
								backing[idx] = int64(inp.TokenIDs[pos-padLen])
								maskRow[pos] = true
							}
						} else {
							if pos < seqLen {
								backing[idx] = int64(inp.TokenIDs[pos])
								maskRow[pos] = true
							}
						}
						idx++
					}
					masks[bi] = maskRow
				}
			case "token_type_ids":
				for _, inp := range batch.Input {
					seqLen := len(inp.TokenIDs)
					for pos := range maxSequenceLength {
						// always right-pad
						if pos < seqLen {
							backing[idx] = int64(inp.TypeIDs[pos])
						}
						idx++
					}
				}
			case "attention_mask":
				if model.IsGenerative {
					// Build mask reflecting actual padding: 0 for pads, 1 for real tokens.
					for bi, inp := range batch.Input {
						seqLen := len(inp.TokenIDs)
						padLen := maxSequenceLength - seqLen
						for pos := range maxSequenceLength {
							if padLeft { // left padding
								if pos < padLen {
									backing[idx] = 0
								} else {
									backing[idx] = 1
								}
							} else { // right padding
								if pos < seqLen {
									backing[idx] = 1
								} else {
									backing[idx] = 0
								}
							}
							idx++
						}
						// ensure masks[bi] exists if input_ids not yet processed
						if masks[bi] == nil {
							maskRow := make([]bool, maxSequenceLength)
							if padLeft {
								for pos := padLen; pos < maxSequenceLength; pos++ {
									maskRow[pos] = true
								}
							} else {
								for pos := 0; pos < seqLen; pos++ {
									maskRow[pos] = true
								}
							}
							masks[bi] = maskRow
						}
					}
				} else {
					// Non-generative: use tokenizer mask (already right padded)
					for _, inp := range batch.Input {
						for pos := 0; pos < maxSequenceLength; pos++ {
							if pos < len(inp.TokenIDs) {
								backing[idx] = int64(inp.AttentionMask[pos])
							}
							idx++
						}
					}
				}
			case "position_ids":
				// HuggingFace-style position ids based on attention_mask cumsum.
				// position_ids = cumsum(attention_mask) - 1; then masked_fill(attention_mask==0, 1)
				// We reconstruct (or infer) the same mask logic here using masks slice.
				for bi, inp := range batch.Input {
					maskRow := masks[bi]
					if maskRow == nil { // fallback build if attention_mask came after and input_ids not seen
						seqLen := len(inp.TokenIDs)
						padLen := maxSequenceLength - seqLen
						maskRow = make([]bool, maxSequenceLength)
						if padLeft {
							for pos := padLen; pos < maxSequenceLength; pos++ {
								maskRow[pos] = true
							}
						} else {
							for pos := 0; pos < seqLen; pos++ {
								maskRow[pos] = true
							}
						}
						masks[bi] = maskRow
					}
					cumulative := 0
					for pos := 0; pos < maxSequenceLength; pos++ {
						if maskRow[pos] {
							backing[idx] = int64(cumulative)
							cumulative++
						} else {
							backing[idx] = 1 // mask fill value per HF pattern described
						}
						idx++
					}
				}
			default:
				return fmt.Errorf("unrecognized input %q", meta.Name)
			}

			// create the ONNX Runtime tensor for regular inputs
			t, err := ort.NewTensor(ort.NewShape(int64(batchSize), int64(maxSequenceLength)), backing)
			if err != nil {
				return err
			}
			inputVals[mi] = t
		}
	}

	// 3) assign and prepare cleanup
	batch.InputValues = inputVals
	batch.PaddingMask = masks
	batch.DestroyInputs = func() error {
		var agg error
		if values, ok := batch.InputValues.([]ort.Value); ok {
			for _, t := range values {
				agg = errors.Join(agg, t.Destroy())
			}
		} else {
			agg = errors.Join(agg, errors.New("batch.InputValues has incorrect type"))
		}
		return agg
	}
	return nil
}

// createSingleCacheTensorORT creates a single cache tensor
func createSingleCacheTensorORT(batchSize, numKeyValueHeads, maxSeqLen, headDim int) (ort.Value, error) {
	tensorSize := batchSize * numKeyValueHeads * maxSeqLen * headDim
	slice := make([]float32, tensorSize)
	return ort.NewTensor(
		ort.NewShape(int64(batchSize), int64(numKeyValueHeads), int64(maxSeqLen), int64(headDim)),
		slice,
	)
}

func runORTSessionOnBatch(batch *PipelineBatch, p *BasePipeline) error {
	actualBatchSize := int64(batch.Size)
	maxSequenceLength := int64(batch.MaxSequenceLength)
	var err error

	// allocate vectors with right dimensions for the output
	outputTensors := make([]ort.Value, len(p.Model.OutputsMeta))
	defer func() {
		for _, output := range outputTensors {
			err = errors.Join(err, output.Destroy())
		}
	}()

	for outputIndex, meta := range p.Model.OutputsMeta {
		var batchDimSet bool
		var tokenDimSet bool
		actualDims := make([]int64, 0, len(meta.Dimensions))

		for _, dim := range meta.Dimensions {
			if dim == -1 {
				if !batchDimSet {
					actualDims = append(actualDims, actualBatchSize)
					batchDimSet = true
				} else if !tokenDimSet {
					actualDims = append(actualDims, maxSequenceLength)
					tokenDimSet = true
				} else {
					return fmt.Errorf("only two axis can be dynamic (batch size and number of tokens)")
				}
			} else {
				actualDims = append(actualDims, dim)
			}
		}
		outputShape := ort.NewShape(actualDims...)
		outputTensors[outputIndex], err = ort.NewEmptyTensor[float32](outputShape)
		if err != nil {
			return err
		}
	}

	errOnnx := p.Model.ORTModel.Session.Run(batch.InputValues.([]ort.Value), outputTensors)
	if errOnnx != nil {
		return errOnnx
	}

	convertedOutput := make([]any, len(outputTensors))
	for i, t := range outputTensors {
		switch v := t.(type) {
		case *ort.Tensor[float32]:
			convertedOutput[i] = ReshapeOutput(v.GetData(), p.Model.OutputsMeta[i], batch.Size, batch.PaddingMask, batch.MaxSequenceLength)
		case *ort.Tensor[int64]:
			convertedOutput[i] = ReshapeOutput(v.GetData(), p.Model.OutputsMeta[i], batch.Size, batch.PaddingMask, batch.MaxSequenceLength)
		}
	}
	// store resulting tensors
	batch.OutputValues = convertedOutput
	return err
}

func runGenerativeORTSessionOnBatch(batch *PipelineBatch, p *BasePipeline) error {
	batchSize := batch.Size
	batchSize64 := int64(batchSize)
	generatedTokens := make([][]int64, batchSize)
	eosTokenIDs := p.Model.EosTokenIDs

	// Map input metadata for intelligent ordering
	inputMetaMap := make(map[string]int)
	for i, inputMeta := range p.Model.InputsMeta {
		inputMetaMap[inputMeta.Name] = i
	}

	finish := make([]bool, batchSize)
	finishCount := 0

	var prefillStart time.Time
	prefillStart = time.Now()

iterations:
	for step := 0; step < batch.MaxNewTokens; step++ {
		inputTensors := batch.InputValues.([]ort.Value)
		outputTensors := make([]ort.Value, len(p.Model.OutputsMeta))
		// Track full iteration time (session run + minimal bookkeeping)
		runStart := time.Now()
		err := p.Model.ORTModel.Session.Run(inputTensors, outputTensors)
		if err != nil {
			return err
		}
		logits := outputTensors[0].(*ort.Tensor[float32]).GetData()
		var logitsReshaped [][][]float32
		if step == 0 {
			dimensions := p.Model.OutputsMeta[0].Dimensions.ValuesInt()
			logitsReshaped = flatDataTo3D(logits, batch.PaddingMask, batch.MaxSequenceLength, dimensions[len(dimensions)-1])
		} else {
			// after the first iteration, the shape of the logits is (batchSize, 1, vocabSize) so this is handled differently
			logitsReshaped = flatDataTo3DGenerativeLoop(logits, batchSize64, int64(p.Model.VocabSize))
		}
		// this matches the python implementation where it will continue to alternate between newline and
		// EOS until the longest output sequence terminates
		// should give an array of batchSize amount of tokens
		greedyTokens := argmax3D(logitsReshaped)
		// Record latency to first token (after prefill) and switch phase timing accounting
		stepDuration := time.Since(runStart)
		if step == 0 {
			// Prefill (prompt processing + first token) timing
			prefillDuration := time.Since(prefillStart)
			if len(batch.PaddingMask) > 0 {
				var promptTokens int
				for _, row := range batch.PaddingMask {
					for _, v := range row {
						if v {
							promptTokens++
						}
					}
				}
				atomicAddUint64(&p.PipelineTimings.PrefillTokens, uint64(promptTokens))
			}
			atomicAddUint64(&p.PipelineTimings.PrefillNS, uint64(prefillDuration.Nanoseconds()))
			atomicAddUint64(&p.PipelineTimings.FirstTokenLatencyNS, uint64(prefillDuration.Nanoseconds()))
			// Include first iteration run duration also in generation time so decode TPS reflects all generated tokens
			atomicAddUint64(&p.PipelineTimings.GenerationNS, uint64(stepDuration.Nanoseconds()))
		} else {
			atomicAddUint64(&p.PipelineTimings.GenerationNS, uint64(stepDuration.Nanoseconds()))
		}
		for i, greedyToken := range greedyTokens {
			if !finish[i] {
				generatedTokens[i] = append(generatedTokens[i], greedyToken)
				if eosTokenIDs[greedyToken] {
					finish[i] = true
					finishCount++
				}
				// Count generated token
				atomicAddUint64(&p.PipelineTimings.GeneratedTokens, 1)
			}
		}
		if finishCount == batchSize {
			break iterations
		}

		// initialize next loop in correct order of the input metadata
		newModelInputs := make([]ort.Value, len(p.Model.InputsMeta))
		for i, inputMeta := range p.Model.InputsMeta {
			switch inputMeta.Name {
			case "input_ids":
				generatedTokenTensor, inputErr := ort.NewTensor(
					ort.NewShape(batchSize64, 1),
					greedyTokens,
				)
				if inputErr != nil {
					return inputErr
				}
				newModelInputs[i] = generatedTokenTensor
			case "position_ids":
				// Compute next position ids without 2D reshaping: take last column per row and add 1
				positionIDs := inputTensors[inputMetaMap["position_ids"]].(*ort.Tensor[int64]).GetData()
				seqLen := len(positionIDs) / batchSize
				lastIdx := seqLen - 1
				newPositionIDs := make([]int64, batchSize)
				for j := range batchSize {
					newPositionIDs[j] = positionIDs[j*seqLen+lastIdx] + 1
				}
				newPositionIDsTensor, positionErr := ort.NewTensor(
					ort.NewShape(batchSize64, 1),
					newPositionIDs,
				)
				if positionErr != nil {
					return positionErr
				}
				newModelInputs[i] = newPositionIDsTensor
			case "attention_mask":
				// Extend each row by one "1" without reshaping to 2D
				attentionMask := inputTensors[inputMetaMap["attention_mask"]].(*ort.Tensor[int64]).GetData()
				seqLen := len(attentionMask) / batchSize
				newAttentionMask := make([]int64, batchSize*(seqLen+1))
				for j := 0; j < batchSize; j++ {
					srcBase := j * seqLen
					dstBase := j * (seqLen + 1)
					copy(newAttentionMask[dstBase:dstBase+seqLen], attentionMask[srcBase:srcBase+seqLen])
					newAttentionMask[dstBase+seqLen] = 1
				}
				newAttentionMaskTensor, attentionErr := ort.NewTensor(
					ort.NewShape(batchSize64, int64(seqLen+1)),
					newAttentionMask,
				)
				if attentionErr != nil {
					return attentionErr
				}
				newModelInputs[i] = newAttentionMaskTensor

			default:
				// handle cache inputs (past_key_values, etc.)
				if strings.HasPrefix(inputMeta.Name, "past_key") {
					cacheInputIndex := 0
					for j, meta := range p.Model.InputsMeta {
						if j < i && strings.HasPrefix(meta.Name, "past_key") {
							cacheInputIndex++
						}
					}
					newModelInputs[i] = outputTensors[1+cacheInputIndex]
				} else {
					return fmt.Errorf("unhandled input type: %s", inputMeta.Name)
				}
			}
		}

		for _, val := range batch.InputValues.([]ort.Value) {
			err = errors.Join(err, val.Destroy())
		}
		if err != nil {
			return err
		}

		batch.InputValues = newModelInputs
	}

	batch.OutputValues = make([]any, batchSize)
	for i := range generatedTokens {
		batch.OutputValues[i] = generatedTokens[i]
	}
	return nil
}

func atomicAddUint64(addr *uint64, delta uint64) {
	atomic.AddUint64(addr, delta)
}

func convertORTInputOutputs(inputOutputs []ort.InputOutputInfo) []InputOutputInfo {
	inputOutputsStandardised := make([]InputOutputInfo, len(inputOutputs))
	for i, inputOutput := range inputOutputs {
		inputOutputsStandardised[i] = InputOutputInfo{
			Name:       inputOutput.Name,
			Dimensions: Shape(inputOutput.Dimensions),
		}
	}
	return inputOutputsStandardised
}

func createImageTensorsORT(batch *PipelineBatch, preprocessed [][][][]float32) error {
	if len(preprocessed) == 0 {
		return errors.New("no preprocessed images provided")
	}
	n, c, h, w := len(preprocessed), len(preprocessed[0]), len(preprocessed[0][0]), len(preprocessed[0][0][0])
	backing := make([]float32, n*c*h*w)
	idx := 0
	for i := range n {
		for ch := range c {
			for y := range h {
				for x := range w {
					backing[idx] = preprocessed[i][ch][y][x]
					idx++
				}
			}
		}
	}
	tensor, err := ort.NewTensor(ort.NewShape(int64(n), int64(c), int64(h), int64(w)), backing)
	if err != nil {
		return err
	}
	batch.InputValues = []ort.Value{tensor}
	batch.DestroyInputs = func() error {
		return tensor.Destroy()
	}
	return nil
}
