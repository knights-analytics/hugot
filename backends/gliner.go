package backends

import "errors"

// ErrUnsupportedRuntime is returned when an operation is attempted with an unsupported runtime.
var ErrUnsupportedRuntime = errors.New("unsupported runtime")

// GLiNERBatchInterface defines the interface for GLiNER batch access.
// This allows the backends package to work with batches without importing pipelines.
type GLiNERBatchInterface interface {
	GetSize() int
	GetMaxSequenceLength() int
	GetInput() []TokenizedInput
	GetWordsMask() [][]int64
	GetTextLengths() [][]int64
	GetSpanIdx() [][][]int64
	GetSpanMask() [][]int64
	GetNumSpans() int
	GetNumLabels() int
	SetInputValues(values any)
	GetInputValues() any
	SetOutputValues(values []any)
	GetOutputValues() []any
	SetDestroyInputs(fn func() error)
}

// GLiNERDefaultMaxWidth is the standard maximum span width in words for GLiNER models.
const GLiNERDefaultMaxWidth = 12

// CreateGLiNERTensors creates input tensors for GLiNER based on runtime.
func CreateGLiNERTensors(batch GLiNERBatchInterface, model *Model, runtime string) error {
	switch runtime {
	case "ORT":
		return createGLiNERTensorsORT(batch, model)
	case "GO", "XLA":
		return createGLiNERTensorsGoMLX(batch, model)
	default:
		return ErrUnsupportedRuntime
	}
}

// RunGLiNERSession runs the GLiNER model on a batch.
func RunGLiNERSession(batch GLiNERBatchInterface, p *BasePipeline) error {
	switch p.Runtime {
	case "ORT":
		return runGLiNERSessionORT(batch, p)
	case "GO", "XLA":
		return runGLiNERSessionGoMLX(batch, p)
	default:
		return ErrUnsupportedRuntime
	}
}

// ComputeGLiNEROutputDimensions extracts or infers output dimensions from batch data.
// Returns (numWords, maxWidth, numClasses) for reshaping the output tensor.
func ComputeGLiNEROutputDimensions(batch GLiNERBatchInterface, dims Shape, dataLen int) (numWords, maxWidth, numClasses int) {
	textLengths := batch.GetTextLengths()
	numSpans := batch.GetNumSpans()

	// Find max words across batch
	for _, tl := range textLengths {
		if len(tl) > 0 && int(tl[0]) > numWords {
			numWords = int(tl[0])
		}
	}
	if numWords == 0 {
		numWords = 1
	}

	maxWidth = numSpans / numWords
	if maxWidth == 0 {
		maxWidth = GLiNERDefaultMaxWidth
	}

	// Last dimension is num_classes (should be fixed in model)
	if len(dims) >= 4 && dims[3] > 0 {
		numClasses = int(dims[3])
	} else {
		// Infer from data length
		expectedSize := batch.GetSize() * numWords * maxWidth
		if expectedSize > 0 {
			numClasses = dataLen / expectedSize
		}
		if numClasses == 0 {
			numClasses = 1
		}
	}

	return numWords, maxWidth, numClasses
}

// ReshapeGLiNEROutput reshapes flat output data to 4D array [batch_size, num_words, max_width, num_classes].
// This is shared between ORT and GoMLX backends.
func ReshapeGLiNEROutput(data []float32, batchSize, numWords, maxWidth, numClasses int) [][][][]float32 {
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
