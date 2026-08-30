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
	SessionContext   context.Context
	Model            *Model
	ONNXTimings      *timings
	TokenizerTimings *timings
	PipelineName     string
	Backend          Backend
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
	GetStatistics() PipelineStatistics                          // Get the pipeline running statistics
	Validate() error                                            // Validate the pipeline for correctness
	GetMetadata() PipelineMetadata                              // Return metadata information for the pipeline
	GetModel() *Model                                           // Return the model used by the pipeline
	IsGenerative() bool                                         // Return whether the pipeline is generative
	Run(context.Context, []string) (PipelineBatchOutput, error) // Run the pipeline on an input
}

type PipelineStatistics struct {
	TokenizerTotalTime             time.Duration
	TokenizerExecutionCount        uint64
	TokenizerAvgQueryTime          time.Duration
	OnnxTotalTime                  time.Duration
	OnnxExecutionCount             uint64
	OnnxAvgQueryTime               time.Duration
	TotalQueries                   uint64
	TotalDocuments                 uint64
	AverageLatency                 time.Duration
	AverageBatchSize               float64
	FilteredResults                uint64
	AvgPrefillSeconds              float64
	TokensPerSecond                float64
	CumulativePrefillSum           float64
	CumulativePrefillCount         int
	CumulativeTokens               int
	CumulativeTokenDurationSeconds float64
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
	MaxAttentionIndex int
}

// PipelineBatch represents a batch of inputs that runs through the pipeline.
type PipelineBatch struct {
	InputValues any
	// Multimodal support
	Images            any // Will hold *ortgenai.Images for generative models
	DestroyInputs     func() error
	DestroyMultimodal func() error
	Input             []TokenizedInput
	PaddingMask       [][]bool
	OutputValues      []any
	Size              int
	MaxSequenceLength int
	MaxNewTokens      int
	// PaddedBatchSize is the bucketed batch size used when XLA pads the batch dimension.
	// Zero means no batch padding was applied (ORT / GO backend).
	PaddedBatchSize int
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

// RunPipeline executes the common synchronous pipeline lifecycle. The typed
// postprocessor keeps the concrete output type visible to pipeline callers.
func RunPipeline[T any](ctx context.Context, size int, preprocess func(*PipelineBatch) error, forward func(context.Context, *PipelineBatch) error, postprocess func(*PipelineBatch) (*T, error)) (result *T, err error) {
	batch := NewBatch(size)
	defer func() {
		err = errors.Join(err, batch.Destroy())
	}()

	if err = preprocess(batch); err != nil {
		return nil, err
	}
	if err = forward(ctx, batch); err != nil {
		return nil, err
	}
	return postprocess(batch)
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

func RunSessionOnBatch(ctx context.Context, batch *PipelineBatch, p *BasePipeline) error {
	if p.Backend == nil {
		return fmt.Errorf("pipeline backend is not configured")
	}
	return p.Backend.RunSessionOnBatch(ctx, batch, p)
}

func RunGenerativeSessionOnBatch(ctx context.Context, batch *PipelineBatch, p *BasePipeline, maxLength int, stopSequences []string, temperature *float64, topP *float64, seed *int, tools []string, guidance *Guidance) (chan SequenceDelta, chan error, error) {
	if p.Backend == nil {
		return nil, nil, fmt.Errorf("pipeline backend is not configured")
	}
	return p.Backend.RunGenerativeSessionOnBatch(ctx, batch, p, maxLength, stopSequences, temperature, topP, seed, tools, guidance)
}

func CreateMessages(batch *PipelineBatch, p *BasePipeline, inputs any, systemPrompt string) error {
	if p.Backend == nil {
		return fmt.Errorf("pipeline backend is not configured")
	}
	return p.Backend.CreateMessages(batch, inputs, systemPrompt)
}

// CreateInputTensorsTraining creates input tensors for training. Same as CreateInputTensors but
// we never pad the batch size as we expect regular batch sizes from the dataset.
func CreateInputTensorsTraining(batch *PipelineBatch, model *Model) error {
	if model.Backend != nil {
		return model.Backend.CreateInputTensors(batch, model, true)
	}
	return fmt.Errorf("pipeline backend is not configured")
}

func CreateInputTensors(batch *PipelineBatch, model *Model) error {
	if model.Backend != nil {
		return model.Backend.CreateInputTensors(batch, model, false)
	}
	return fmt.Errorf("pipeline backend is not configured")
}

// CreateTabularTensors builds input tensors for classic ML/tabular models.
func CreateTabularTensors(batch *PipelineBatch, model *Model, features [][]float32) error {
	if model.Backend != nil {
		return model.Backend.CreateTabularTensors(batch, model, features)
	}
	return fmt.Errorf("pipeline backend is not configured")
}

func NewBasePipeline[T Pipeline](sessionContext context.Context, config PipelineConfig[T], model *Model) *BasePipeline {
	return &BasePipeline{
		PipelineName:     config.Name,
		Model:            model,
		Backend:          model.Backend,
		ONNXTimings:      &timings{},
		TokenizerTimings: &timings{},
		SessionContext:   sessionContext,
	}
}

func CreateModelBackend(ctx context.Context, model *Model, s *options.Options) error {
	err := GetOnnxModelPath(ctx, model)
	if err != nil {
		return err
	}

	if strings.HasPrefix(model.Path, "s3:") {
		reader, readErr := fileutil.OpenFile(ctx, fileutil.PathJoinSafe(model.Path, model.OnnxPath))
		if readErr != nil {
			return readErr
		}
		model.OnnxReader = reader
	}

	backend, backendErr := newBackend(s)
	if backendErr != nil {
		return backendErr
	}
	model.Backend = backend
	switch s.Backend {
	case options.BackendORT:
		if s.UseGoMLX {
			err = createGoMLXModelBackend(model, s)
		} else {
			err = createORTModelBackend(model, s)
		}
	case options.BackendGo, options.BackendXLA:
		err = createGoMLXModelBackend(model, s)
	}
	return err
}
