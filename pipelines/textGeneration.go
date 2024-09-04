package pipelines

import (
	"errors"
	"fmt"
	"math"
	"sync/atomic"
	"time"

	"github.com/daulet/tokenizers"
	ort "github.com/yalue/onnxruntime_go"
)

type TextGenerationPipeline struct {
	basePipeline
}

type TextGenerationOutput struct {
	// TODO fill
}

func (t *TextGenerationOutput) GetOutput() []any {
	return []any{} // TODO fill
}

func NewTextGenerationPipeline(config PipelineConfig[*TextGenerationPipeline], ortOptions *ort.SessionOptions) (*TextGenerationPipeline, error) {
	pipeline := &TextGenerationPipeline{}
	pipeline.ModelPath = config.ModelPath
	pipeline.PipelineName = config.Name
	pipeline.OrtOptions = ortOptions
	pipeline.OnnxFilename = config.OnnxFilename
	for _, o := range config.Options {
		o(pipeline)
	}

	// onnx model init
	model, err := loadOnnxModelBytes(pipeline.ModelPath, pipeline.OnnxFilename)
	if err != nil {
		return nil, err
	}

	// init of inputs and outputs
	inputs, outputs, err := loadInputOutputMeta(model)
	if err != nil {
		return nil, err
	}
	pipeline.InputsMeta = inputs
	pipeline.OutputsMeta = outputs

	pipeline.PipelineTimings = &timings{}
	pipeline.TokenizerTimings = &timings{}

	// tokenizer init
	pipeline.TokenizerOptions, err = getTokenizerOptions(inputs)
	if err != nil {
		return nil, err
	}
	// Additional options needed for postprocessing
	pipeline.TokenizerOptions = append(pipeline.TokenizerOptions,
		tokenizers.WithReturnSpecialTokensMask(),
	)
	tk, tkErr := loadTokenizer(pipeline.ModelPath)
	if tkErr != nil {
		return nil, tkErr
	}
	pipeline.Tokenizer = tk

	// creation of the session. Only one output (either token or sentence embedding).
	session, err := createSession(model, inputs, outputs, ortOptions)
	if err != nil {
		return nil, err
	}
	pipeline.OrtSession = session

	err = pipeline.Validate()
	if err != nil {
		return nil, err
	}
	return pipeline, nil
}

func (p *TextGenerationPipeline) GetMetadata() PipelineMetadata {
	return PipelineMetadata{}
}

// Destroy frees the feature extraction pipeline resources.
func (p *TextGenerationPipeline) Destroy() error {
	return destroySession(p.Tokenizer, p.OrtSession)
}

// GetStats returns the runtime statistics for the pipeline.
func (p *TextGenerationPipeline) GetStats() []string {
	return []string{
		fmt.Sprintf("Statistics for pipeline: %s", p.PipelineName),
		fmt.Sprintf("Tokenizer: Total time=%s, Execution count=%d, Average query time=%s",
			time.Duration(p.TokenizerTimings.TotalNS),
			p.TokenizerTimings.NumCalls,
			time.Duration(float64(p.TokenizerTimings.TotalNS)/math.Max(1, float64(p.TokenizerTimings.NumCalls)))),
		fmt.Sprintf("ONNX: Total time=%s, Execution count=%d, Average query time=%s",
			time.Duration(p.PipelineTimings.TotalNS),
			p.PipelineTimings.NumCalls,
			time.Duration(float64(p.PipelineTimings.TotalNS)/math.Max(1, float64(p.PipelineTimings.NumCalls)))),
	}
}

func (p *TextGenerationPipeline) Validate() error {
	// TODO add validation
	return nil
}

func (p *TextGenerationPipeline) Preprocess(batch *PipelineBatch, inputs []string) error {
	start := time.Now()
	tokenizeInputs(batch, p.Tokenizer, inputs, p.TokenizerOptions)
	atomic.AddUint64(&p.TokenizerTimings.NumCalls, 1)
	atomic.AddUint64(&p.TokenizerTimings.TotalNS, uint64(time.Since(start)))
	return createInputTensors(batch, p.InputsMeta)
}

func (p *TextGenerationPipeline) Forward(batch *PipelineBatch) error {
	start := time.Now()
	err := runSessionOnBatch(batch, p.OrtSession, p.OutputsMeta)
	if err != nil {
		return err
	}
	atomic.AddUint64(&p.PipelineTimings.NumCalls, 1)
	atomic.AddUint64(&p.PipelineTimings.TotalNS, uint64(time.Since(start)))
	return nil
}

func (p *TextGenerationPipeline) Run(inputs []string) (PipelineBatchOutput, error) {
	return p.RunPipeline(inputs)
}

// TODO for now assume that the list of strings is a list of prompts to autocomplete
// later we should also allow for list of chat histories

func (p *TextGenerationPipeline) RunPipeline(inputs []string) (*TextGenerationOutput, error) {
	var runErrors []error
	batch := NewBatch()
	defer func(*PipelineBatch) {
		runErrors = append(runErrors, batch.Destroy())
	}(batch)

	runErrors = append(runErrors, p.Preprocess(batch, inputs))
	if e := errors.Join(runErrors...); e != nil {
		return nil, e
	}

	runErrors = append(runErrors, p.Forward(batch))
	if e := errors.Join(runErrors...); e != nil {
		return nil, e
	}

	// result, postErr := p.Postprocess(batch)
	// runErrors = append(runErrors, postErr)
	// return result, errors.Join(runErrors...)
	return &TextGenerationOutput{}, nil
}
