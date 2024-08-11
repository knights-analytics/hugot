package pipelines

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/daulet/tokenizers"
	ort "github.com/yalue/onnxruntime_go"

	util "github.com/knights-analytics/hugot/utils"
)

// BasePipeline can be embedded by a pipeline.
type basePipeline struct {
	ModelPath        string
	OnnxFilename     string
	PipelineName     string
	OrtSession       *ort.DynamicAdvancedSession
	OrtOptions       *ort.SessionOptions
	Tokenizer        *tokenizers.Tokenizer
	TokenizerOptions []tokenizers.EncodeOption
	InputsMeta       []ort.InputOutputInfo
	OutputsMeta      []ort.InputOutputInfo
	TokenizerTimings *timings
	PipelineTimings  *timings
}

type OutputInfo struct {
	Name       string
	Dimensions []int64
}

type PipelineMetadata struct {
	OutputsInfo []OutputInfo
}

type PipelineBatchOutput interface {
	GetOutput() []any
}

// Pipeline is the interface that any pipeline must implement.
type Pipeline interface {
	Destroy() error                            // Destroy the pipeline along with its onnx session
	GetStats() []string                        // Get the pipeline running stats
	Validate() error                           // Validate the pipeline for correctness
	GetMetadata() PipelineMetadata             // Return metadata information for the pipeline
	Run([]string) (PipelineBatchOutput, error) // Run the pipeline on an input
}

// PipelineOption is an option for a pipeline type.
type PipelineOption[T Pipeline] func(eo T)

// PipelineConfig is a configuration for a pipeline type that can be used
// to create that pipeline.
type PipelineConfig[T Pipeline] struct {
	ModelPath    string
	Name         string
	OnnxFilename string
	Options      []PipelineOption[T]
}

type timings struct {
	NumCalls uint64
	TotalNS  uint64
}

// tokenizedInput holds the result of running tokenizer on an input.
type tokenizedInput struct {
	Raw               string
	Tokens            []string
	TokenIDs          []uint32
	TypeIDs           []uint32
	AttentionMask     []uint32
	SpecialTokensMask []uint32
	MaxAttentionIndex int
	Offsets           []tokenizers.Offset
}

// PipelineBatch represents a batch of inputs that runs through the pipeline.
type PipelineBatch struct {
	Input             []tokenizedInput
	InputTensors      []*ort.Tensor[int64]
	MaxSequenceLength int
	OutputTensors     []*ort.Tensor[float32]
}

func (b *PipelineBatch) Destroy() error {
	destroyErrors := make([]error, 0, len(b.InputTensors)+len(b.OutputTensors))

	for _, tensor := range b.InputTensors {
		destroyErrors = append(destroyErrors, tensor.Destroy())
	}

	for _, tensor := range b.OutputTensors {
		destroyErrors = append(destroyErrors, tensor.Destroy())
	}
	return errors.Join(destroyErrors...)
}

// NewBatch initializes a new batch for inference.
func NewBatch() *PipelineBatch {
	return &PipelineBatch{}
}

func loadTokenizer(modelPath string) (*tokenizers.Tokenizer, error) {
	tokenizerBytes, err := util.ReadFileBytes(util.PathJoinSafe(modelPath, "tokenizer.json"))
	if err != nil {
		return nil, err
	}

	tk, err := tokenizers.FromBytes(tokenizerBytes)
	if err != nil {
		return nil, err
	}
	return tk, nil
}

func loadOnnxModelBytes(modelPath string, modelFilename string) ([]byte, error) {
	var modelOnnxFile string
	onnxFiles, err := getOnnxFiles(modelPath)
	if err != nil {
		return nil, err
	}
	if len(onnxFiles) == 0 {
		return nil, fmt.Errorf("no .onnx file detected at %s. There should be exactly .onnx file", modelPath)
	}
	if len(onnxFiles) > 1 {
		if modelFilename == "" {
			return nil, fmt.Errorf("multiple .onnx file detected at %s and no OnnxFilename specified", modelPath)
		}
		modelNameFound := false
		for i := range onnxFiles {
			if onnxFiles[i][1] == modelFilename {
				modelNameFound = true
				modelOnnxFile = util.PathJoinSafe(onnxFiles[i]...)
			}
		}
		if !modelNameFound {
			return nil, fmt.Errorf("file %s not found at %s", modelFilename, modelPath)
		}
	} else {
		modelOnnxFile = util.PathJoinSafe(onnxFiles[0]...)
	}

	onnxBytes, err := util.ReadFileBytes(modelOnnxFile)
	if err != nil {
		return nil, err
	}
	return onnxBytes, err
}

func loadInputOutputMeta(onnxBytes []byte) ([]ort.InputOutputInfo, []ort.InputOutputInfo, error) {
	inputs, outputs, err := ort.GetInputOutputInfoWithONNXData(onnxBytes)
	if err != nil {
		return nil, nil, err
	}
	return inputs, outputs, nil
}

func createSession(onnxBytes []byte, inputs, outputs []ort.InputOutputInfo, options *ort.SessionOptions) (*ort.DynamicAdvancedSession, error) {
	var inputNames []string
	var outputNames []string
	for _, v := range inputs {
		inputNames = append(inputNames, v.Name)
	}
	for _, v := range outputs {
		outputNames = append(outputNames, v.Name)
	}
	session, err := ort.NewDynamicAdvancedSessionWithONNXData(
		onnxBytes,
		inputNames,
		outputNames,
		options,
	)
	return session, err
}

func getOnnxFiles(path string) ([][]string, error) {
	var onnxFiles [][]string
	walker := func(_ context.Context, _ string, parent string, info os.FileInfo, _ io.Reader) (toContinue bool, err error) {
		if strings.HasSuffix(info.Name(), ".onnx") {
			onnxFiles = append(onnxFiles, []string{util.PathJoinSafe(path, parent), info.Name()})
		}
		return true, nil
	}
	err := util.FileSystem.Walk(context.Background(), path, walker)
	return onnxFiles, err
}

func tokenizeInputs(batch *PipelineBatch, tk *tokenizers.Tokenizer, inputs []string, options []tokenizers.EncodeOption) {
	outputs := make([]tokenizedInput, len(inputs))
	maxSequence := 0
	for i, input := range inputs {

		output := tk.EncodeWithOptions(input,
			true,
			options...,
		)

		maxAttentionIndex := 0
		for j, attentionMaskValue := range output.AttentionMask {
			if attentionMaskValue != 0 {
				maxAttentionIndex = j
			}
		}

		outputs[i] = tokenizedInput{
			Raw:               input,
			Tokens:            output.Tokens,
			TokenIDs:          output.IDs,
			TypeIDs:           output.TypeIDs,
			AttentionMask:     output.AttentionMask,
			MaxAttentionIndex: maxAttentionIndex,
			SpecialTokensMask: output.SpecialTokensMask,
			Offsets:           output.Offsets, // we need the offsets here for postprocessing later
		}
		if maxAttentionIndex > maxSequence {
			maxSequence = maxAttentionIndex
		}
	}
	batch.Input = outputs
	batch.MaxSequenceLength = maxSequence + 1
}

// createInputTensors creates ort input tensors.
func createInputTensors(batch *PipelineBatch, inputsMeta []ort.InputOutputInfo) error {
	tensorSize := len(batch.Input) * (batch.MaxSequenceLength)
	batchSize := int64(len(batch.Input))

	inputTensors := make([]*ort.Tensor[int64], len(inputsMeta))
	var tensorCreationErr error

	for i, inputMeta := range inputsMeta {
		backingSlice := make([]int64, tensorSize)
		counter := 0

		for _, input := range batch.Input {
			length := len(input.TokenIDs)
			for j := 0; j < batch.MaxSequenceLength; j++ {
				if j+1 <= length {
					switch inputMeta.Name {
					case "input_ids":
						backingSlice[counter] = int64(input.TokenIDs[j])
					case "token_type_ids":
						backingSlice[counter] = int64(input.TypeIDs[j])
					case "attention_mask":
						backingSlice[counter] = int64(input.AttentionMask[j])
					default:
						return fmt.Errorf("input %s not recognized", inputMeta.Name)
					}
				} else {
					backingSlice[counter] = 0 // pad with zero
				}
				counter++
			}
		}
		inputTensors[i], tensorCreationErr = ort.NewTensor(ort.NewShape(batchSize, int64(batch.MaxSequenceLength)), backingSlice)
		if tensorCreationErr != nil {
			return tensorCreationErr
		}
	}
	batch.InputTensors = inputTensors
	return nil
}

func getNames(info []ort.InputOutputInfo) []string {
	names := make([]string, 0, len(info))
	for _, v := range info {
		names = append(names, v.Name)
	}
	return names
}

func getTokenizerOptions(inputs []ort.InputOutputInfo) ([]tokenizers.EncodeOption, error) {
	var encodeOptions []tokenizers.EncodeOption
	for _, input := range inputs {
		switch input.Name {
		case "input_ids":
			encodeOptions = append(encodeOptions, tokenizers.WithReturnTokens())
		case "token_type_ids":
			encodeOptions = append(encodeOptions, tokenizers.WithReturnTypeIDs())
		case "attention_mask":
			encodeOptions = append(encodeOptions, tokenizers.WithReturnAttentionMask())
		default:
			return nil, fmt.Errorf("input %s not recognized", input.Name)
		}
	}
	return encodeOptions, nil
}

func runSessionOnBatch(batch *PipelineBatch, session *ort.DynamicAdvancedSession, outputs []ort.InputOutputInfo) error {
	actualBatchSize := int64(len(batch.Input))
	maxSequenceLength := int64(batch.MaxSequenceLength)

	// allocate vectors with right dimensions for the output
	outputTensors := make([]*ort.Tensor[float32], len(outputs))
	arbitraryOutputTensors := make([]ort.ArbitraryTensor, len(outputs))
	var outputCreationErr error

	for outputIndex, meta := range outputs {
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
		outputTensors[outputIndex], outputCreationErr = ort.NewEmptyTensor[float32](outputShape)
		if outputCreationErr != nil {
			return outputCreationErr
		}
		arbitraryOutputTensors[outputIndex] = ort.ArbitraryTensor(outputTensors[outputIndex])
	}

	// Run Onnx model
	arbitraryInputTensors := make([]ort.ArbitraryTensor, len(batch.InputTensors))
	for i, t := range batch.InputTensors {
		arbitraryInputTensors[i] = ort.ArbitraryTensor(t)
	}

	errOnnx := session.Run(arbitraryInputTensors, arbitraryOutputTensors)
	if errOnnx != nil {
		return errOnnx
	}

	// store resulting tensors
	batch.OutputTensors = outputTensors
	return nil
}

func destroySession(tk *tokenizers.Tokenizer, session *ort.DynamicAdvancedSession) error {
	var finalErr error
	errTokenizer := tk.Close()
	if errTokenizer != nil {
		finalErr = errTokenizer
	}
	ortError := session.Destroy()
	if ortError != nil {
		finalErr = ortError
	}
	return finalErr
}
