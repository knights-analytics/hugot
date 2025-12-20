package backends

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"time"

	"github.com/knights-analytics/hugot/options"
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

// FixedShapeInfo contains the fixed dimensions for models exported with static shapes.
// Models exported with no_dynamic_axes=True (e.g., via Optimum) have positive dimension
// values, while dynamic models use -1 or 0 to indicate variable dimensions.
type FixedShapeInfo struct {
	BatchSize     int
	SequenceLength int
	HasFixedShape bool
}

// GetFixedShapeFromInputs examines model input metadata to detect fixed-shape models.
// It looks for the "input_ids" input (standard for transformer models) and checks if
// both batch and sequence dimensions are positive (indicating fixed shapes).
//
// Returns FixedShapeInfo with HasFixedShape=true if the model requires fixed dimensions,
// otherwise returns HasFixedShape=false and the caller should use dynamic sizing.
func GetFixedShapeFromInputs(inputs []InputOutputInfo) FixedShapeInfo {
	for _, meta := range inputs {
		if meta.Name == "input_ids" && len(meta.Dimensions) >= 2 {
			batchDim := meta.Dimensions[0]
			seqDim := meta.Dimensions[1]
			// Positive dimensions indicate fixed shapes; -1 or 0 indicate dynamic
			if batchDim > 0 && seqDim > 0 {
				return FixedShapeInfo{
					BatchSize:      int(batchDim),
					SequenceLength: int(seqDim),
					HasFixedShape:  true,
				}
			}
		}
	}
	return FixedShapeInfo{HasFixedShape: false}
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
}

func (b *PipelineBatch) Destroy() error {
	return b.DestroyInputs()
}

// NewBatch initializes a new batch for inference.
func NewBatch(size int) *PipelineBatch {
	return &PipelineBatch{
		DestroyInputs: func() error {
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

func CreateMessages(batch *PipelineBatch, p *BasePipeline, inputs any) error {
	switch p.Runtime {
	case "ORT":
		return CreateMessagesORT(batch, inputs)
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
		return createInputTensorsGoMLX(batch, model, false, false)
	case "XLA":
		return createInputTensorsGoMLX(batch, model, true, true)
	}
	return nil
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
	var err error
	switch s.Backend {
	case "ORT":
		err = createORTModelBackend(model, s)
	case "GO", "XLA":
		err = createGoMLXModelBackend(model, s)
	}
	return err
}
