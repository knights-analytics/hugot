package pipelines

import "C"
import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/util"
)

// BasePipeline can be embedded by a pipeline.
type BasePipeline struct {
	ModelPath       string
	OnnxFilename    string
	PipelineName    string
	Runtime         string
	ORTSession      *ORTSession
	GoSession       *GoSession
	XLASession      *XLASession
	Tokenizer       *Tokenizer
	InputsMeta      []InputOutputInfo
	OutputsMeta     []InputOutputInfo
	PipelineTimings *timings
}

func (p *BasePipeline) Destroy() error {
	finalErr := p.Tokenizer.Destroy()
	if p.ORTSession != nil {
		finalErr = errors.Join(finalErr, p.ORTSession.Destroy())
	}
	if p.XLASession != nil {
		p.XLASession.Destroy()
	}
	return finalErr
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

func (s Shape) ValuesInt() []int {
	output := make([]int, len(s))
	for i, v := range s {
		output[i] = int(v)
	}
	return output
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

// TokenizedInput holds the result of running tokenizer on an input.
type TokenizedInput struct {
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
	Input             []TokenizedInput
	MaxSequenceLength int
	InputValues       any
	DestroyInputs     func() error
	OutputValues      [][]float32
}

func (b *PipelineBatch) Destroy() error {
	return b.DestroyInputs()
}

// NewBatch initializes a new batch for inference.
func NewBatch() *PipelineBatch {
	return &PipelineBatch{DestroyInputs: func() error {
		return nil
	}}
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

func GetNames(info []InputOutputInfo) []string {
	names := make([]string, 0, len(info))
	for _, v := range info {
		names = append(names, v.Name)
	}
	return names
}

func RunSessionOnBatch(batch *PipelineBatch, p *BasePipeline) error {
	switch p.Runtime {
	case "GO":
		return runGoSessionOnBatch(batch, p)
	case "ORT":
		return runORTSessionOnBatch(batch, p)
	case "XLA":
		return runXLASessionOnBatch(batch, p)
	}
	return nil
}

func CreateInputTensors(batch *PipelineBatch, inputsMeta []InputOutputInfo, runtime string) error {

	switch runtime {
	case "ORT":
		return createInputTensorsORT(batch, inputsMeta)
	case "GO":
		return createInputTensorsGo(batch, inputsMeta)
	case "XLA":
		return createInputTensorsXLA(batch, inputsMeta)
	}
	return nil
}

func NewBasePipeline[T Pipeline](config PipelineConfig[T], s *options.Options) (*BasePipeline, error) {
	pipeline := &BasePipeline{}
	pipeline.Runtime = s.Runtime
	pipeline.ModelPath = config.ModelPath
	pipeline.PipelineName = config.Name
	pipeline.OnnxFilename = config.OnnxFilename
	pipeline.PipelineTimings = &timings{}

	// onnx model init
	model, err := loadOnnxModelBytes(pipeline.ModelPath, pipeline.OnnxFilename)
	if err != nil {
		return nil, err
	}

	err = createSession(pipeline, s, model)
	if err != nil {
		return nil, err
	}

	tkErr := loadTokenizer(pipeline)
	if tkErr != nil {
		return nil, tkErr
	}

	return pipeline, nil
}

func createSession(pipeline *BasePipeline, s *options.Options, model []byte) error {
	// creation of the session. Only one output (either token or sentence embedding).
	var err error
	switch pipeline.Runtime {
	case "GO":
		err = createGoPipeline(pipeline, model, s)
	case "ORT":
		err = createORTPipeline(pipeline, model, s)
	case "XLA":
		err = createXLAPipeline(pipeline, model, s)
	}
	return err
}
