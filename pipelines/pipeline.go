package pipelines

import "C"
import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/advancedclimatesystems/gonnx"
	"github.com/daulet/tokenizers"
	ort "github.com/yalue/onnxruntime_go"
	"gorgonia.org/tensor"

	util "github.com/knights-analytics/hugot/utils"
)

// BasePipeline can be embedded by a pipeline.
type basePipeline struct {
	ModelPath        string
	OnnxFilename     string
	PipelineName     string
	Type             string
	Runtime          string
	ORTSession       *ort.DynamicAdvancedSession
	ORTOptions       *ort.SessionOptions
	GoSession        *gonnx.Model
	Tokenizer        *tokenizers.Tokenizer
	TokenizerOptions []tokenizers.EncodeOption
	InputsMeta       []InputOutputInfo
	OutputsMeta      []InputOutputInfo
	TokenizerTimings *timings
	PipelineTimings  *timings
}

type InputOutputInfo struct {
	// The name of the input or output
	Name string
	// The input or output's dimensions, if it's a tensor. This should be
	// ignored for non-tensor types.
	Dimensions Shape
}

type Shape []int64

func (s Shape) String() string {
	return fmt.Sprintf("%v", []int64(s))
}

// NewShape Returns a Shape, with the given dimensions.
func NewShape(dimensions ...int64) Shape {
	return dimensions
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
	MaxSequenceLength int
	InputValuesORT    []ort.Value
	InputValuesGo     map[string]tensor.Tensor
	OutputValuesORT   []ort.Value
	OutputValuesGo    map[string]tensor.Tensor
}

func (b *PipelineBatch) Destroy() error {

	if len(b.InputValuesORT) > 0 {
		destroyErrors := make([]error, 0, len(b.InputValuesORT)+len(b.OutputValuesORT))

		for _, ortTensor := range b.InputValuesORT {
			destroyErrors = append(destroyErrors, ortTensor.Destroy())
		}

		for _, ortTensor := range b.OutputValuesORT {
			destroyErrors = append(destroyErrors, ortTensor.Destroy())
		}
		return errors.Join(destroyErrors...)
	}
	return nil
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

func getNames(info []InputOutputInfo) []string {
	names := make([]string, 0, len(info))
	for _, v := range info {
		names = append(names, v.Name)
	}
	return names
}

func getTokenizerOptions(inputs []InputOutputInfo) ([]tokenizers.EncodeOption, error) {
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

func runSessionOnBatch(batch *PipelineBatch, ortSession *ort.DynamicAdvancedSession, goSession *gonnx.Model, outputsMeta []InputOutputInfo) error {
	if ortSession != nil {
		return runORTSessionOnBatch(batch, ortSession, outputsMeta)
	} else if goSession != nil {
		return runGoSessionOnBatch(batch, goSession)
	}
	return nil
}

func createInputTensors(batch *PipelineBatch, inputsMeta []InputOutputInfo, runtime string) error {

	switch runtime {
	case "ORT":
		return createInputTensorsORT(batch, inputsMeta)
	case "GO":
		return createInputTensorsGo(batch, inputsMeta)
	}
	return nil
}
