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
	ort "github.com/yalue/onnxruntime_go"
	"gorgonia.org/tensor"

	util "github.com/knights-analytics/hugot/utils"
)

// basePipeline can be embedded by a pipeline.
type basePipeline struct {
	ModelPath       string
	OnnxFilename    string
	PipelineName    string
	Type            string
	Runtime         string
	ORTSession      *ort.DynamicAdvancedSession
	ORTOptions      *ort.SessionOptions
	GoSession       *gonnx.Model
	Tokenizer       *Tokenizer
	InputsMeta      []InputOutputInfo
	OutputsMeta     []InputOutputInfo
	PipelineTimings *timings
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
	Runtime      string
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
	Offsets           [][2]uint
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

func getNames(info []InputOutputInfo) []string {
	names := make([]string, 0, len(info))
	for _, v := range info {
		names = append(names, v.Name)
	}
	return names
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

func newBasePipeline[T Pipeline](config PipelineConfig[T], ortOptions *ort.SessionOptions) (*basePipeline, error) {
	pipeline := &basePipeline{}
	pipeline.Runtime = config.Runtime
	pipeline.ModelPath = config.ModelPath
	pipeline.PipelineName = config.Name
	pipeline.ORTOptions = ortOptions
	pipeline.OnnxFilename = config.OnnxFilename
	pipeline.PipelineTimings = &timings{}

	// onnx model init
	model, err := loadOnnxModelBytes(pipeline.ModelPath, pipeline.OnnxFilename)
	if err != nil {
		return nil, err
	}

	err = createSession(pipeline, ortOptions, model)
	if err != nil {
		return nil, err
	}

	tkErr := loadTokenizer(pipeline)
	if tkErr != nil {
		return nil, tkErr
	}

	return pipeline, nil
}

func createSession(pipeline *basePipeline, ortOptions *ort.SessionOptions, model []byte) error {
	// creation of the session. Only one output (either token or sentence embedding).
	switch pipeline.Runtime {
	case "GO":
		// creation of the session. Only one output (either token or sentence embedding).
		session, inputs, outputs, sessionErr := createGoSession(model)
		if sessionErr != nil {
			return sessionErr
		}
		pipeline.GoSession = session
		pipeline.InputsMeta = inputs
		pipeline.OutputsMeta = outputs
	case "ORT":
		session, inputs, outputs, sessionErr := createORTSession(model, ortOptions)
		if sessionErr != nil {
			return sessionErr
		}
		pipeline.ORTSession = session
		pipeline.InputsMeta = inputs
		pipeline.OutputsMeta = outputs
	}
	return nil
}
