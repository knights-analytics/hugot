package backends

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/util/fileutil"
	"github.com/knights-analytics/hugot/util/safeconv"
)

// BasePipeline can be embedded by a pipeline.
type BasePipeline struct {
	Model           *Model
	PipelineTimings *timings
	PipelineName    string
	Runtime         string
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
	GetStatistics() PipelineStatistics         // Get the pipeline running statistics
	Validate() error                           // Validate the pipeline for correctness
	GetMetadata() PipelineMetadata             // Return metadata information for the pipeline
	GetModel() *Model                          // Return the model used by the pipeline
	IsGenerative() bool                        // Return whether the pipeline is generative
	Run([]string) (PipelineBatchOutput, error) // Run the pipeline on an input
}

type PipelineStatistics struct {
	TokenizerTotalTime      time.Duration
	TokenizerExecutionCount uint64
	TokenizerAvgQueryTime   time.Duration
	OnnxTotalTime           time.Duration
	OnnxExecutionCount      uint64
	OnnxAvgQueryTime        time.Duration
	TotalQueries            uint64
	TotalDocuments          uint64
	AverageLatency          time.Duration
	AverageBatchSize        float64
	FilteredResults         uint64
}

func (p *PipelineStatistics) ComputeTokenizerStatistics(timings *timings) {
	p.TokenizerTotalTime = safeconv.U64ToDuration(timings.TotalNS)
	p.TokenizerExecutionCount = timings.NumCalls
	p.TokenizerAvgQueryTime = time.Duration(float64(timings.TotalNS) /
		math.Max(1, float64(timings.NumCalls)))
}

func (p *PipelineStatistics) ComputeOnnxStatistics(timings *timings) {
	p.OnnxTotalTime = safeconv.U64ToDuration(timings.TotalNS)
	p.OnnxExecutionCount = timings.NumCalls
	p.OnnxAvgQueryTime = time.Duration(float64(timings.TotalNS) /
		math.Max(1, float64(timings.NumCalls)))
}

func (p *PipelineStatistics) Print() {
	jsonData, err := json.MarshalIndent(p, "", "  ")
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println(string(jsonData))
}

// PipelineOption is an option for a pipeline type.
type PipelineOption[T Pipeline] func(eo T) error

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
	Offsets           [][2]uint
	WordIDs           []int
	MaxAttentionIndex int
}

// PipelineBatch represents a batch of inputs that runs through the pipeline.
type PipelineBatch struct {
	InputValues       any
	DestroyInputs     func() error
	Input             []TokenizedInput
	PaddingMask       [][]bool
	OutputValues      []any
	Size              int
	MaxSequenceLength int
	MaxNewTokens      int
	// Multimodal support
	Images            any // Will hold *ortgenai.Images for generative models
	DestroyMultimodal func() error
}

func (b *PipelineBatch) Destroy() error {
	var err error
	if b.DestroyInputs != nil {
		err = errors.Join(err, b.DestroyInputs())
	}
	if b.DestroyMultimodal != nil {
		err = errors.Join(err, b.DestroyMultimodal())
	}
	return err
}

// NewBatch initializes a new batch for inference.
func NewBatch(size int) *PipelineBatch {
	return &PipelineBatch{
		DestroyInputs: func() error {
			return nil
		},
		DestroyMultimodal: func() error {
			return nil
		},
		Size: size,
	}
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
	case "ORT":
		return runORTSessionOnBatch(batch, p)
	case "GO", "XLA":
		return runGoMLXSessionOnBatch(batch, p)
	}
	return nil
}

func RunGenerativeSessionOnBatch(ctx context.Context, batch *PipelineBatch, p *BasePipeline, maxLength int) (chan SequenceDelta, chan error, error) {
	switch p.Runtime {
	case "ORT":
		return runGenerativeORTSessionOnBatch(ctx, batch, p, maxLength)
	case "GO":
		return nil, nil, errors.New("GO backend is not yet implemented for generative models")
	case "XLA":
		return nil, nil, errors.New("XLA backend is not yet implemented for generative models")
	default:
		return nil, nil, errors.New("invalid backend")
	}
}

func CreateMessages(batch *PipelineBatch, p *BasePipeline, inputs any, systemPrompt string) error {
	switch p.Runtime {
	case "ORT":
		return CreateMessagesORT(batch, inputs, systemPrompt)
	case "GO", "XLA":
		return fmt.Errorf("not implemented")
	}
	return nil
}

// CreateInputTensorsTraining creates input tensors for training. Same as CreateInputTensors but
// we never pad the batch size as we expect regular batch sizes from the dataset.
func CreateInputTensorsTraining(batch *PipelineBatch, model *Model, runtime string) error {
	switch runtime {
	case "ORT":
		return createInputTensorsORT(batch, model)
	case "GO":
		return createInputTensorsGoMLX(batch, model, false, false)
	case "XLA":
		return createInputTensorsGoMLX(batch, model, false, true)
	}
	return nil
}

func CreateInputTensors(batch *PipelineBatch, model *Model, runtime string) error {
	switch runtime {
	case "ORT":
		return createInputTensorsORT(batch, model)
	case "GO":
		if model.GoMLXModel.MaxCache > 0 {
			// only pad the batch dimension if we have a limited cache
			return createInputTensorsGoMLX(batch, model, true, true)
		}
		return createInputTensorsGoMLX(batch, model, false, false)
	case "XLA":
		return createInputTensorsGoMLX(batch, model, true, true)
	}
	return nil
}

// CreateTabularTensors builds input tensors for classic ML/tabular models.
func CreateTabularTensors(batch *PipelineBatch, model *Model, features [][]float32, runtime string) error {
	switch runtime {
	case "ORT":
		return createTabularTensorsORT(batch, model, features)
	case "GO", "XLA":
		return createTabularTensorsGoMLX(batch, model, features)
	default:
		return fmt.Errorf("invalid runtime %s", runtime)
	}
}

func NewBasePipeline[T Pipeline](config PipelineConfig[T], s *options.Options, model *Model) (*BasePipeline, error) {
	pipeline := &BasePipeline{}
	pipeline.Runtime = s.Backend
	pipeline.PipelineName = config.Name
	pipeline.Model = model
	pipeline.PipelineTimings = &timings{}
	return pipeline, nil
}

func CreateModelBackend(model *Model, s *options.Options) error {
	err := GetOnnxModelPath(model)
	if err != nil {
		return err
	}

	loadAsBytes := strings.HasPrefix(model.Path, "s3:")
	if loadAsBytes {
		onnxBytes, readErr := fileutil.ReadFileBytes(model.Path + "/" + model.OnnxPath)
		if readErr != nil {
			return readErr
		}
		model.OnnxBytes = onnxBytes
	}

	switch s.Backend {
	case "ORT":
		err = createORTModelBackend(model, loadAsBytes, s)
	case "GO", "XLA":
		err = createGoMLXModelBackend(model, loadAsBytes, s)
	}

	// ONNX bytes no longer needed after creating the session
	model.OnnxBytes = nil
	return err
}
