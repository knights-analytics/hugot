package pipelines

import (
	"context"
	"errors"

	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/options"
)

type TextGenerationPipeline struct {
	*backends.BasePipeline
	SystemPrompt string
	MaxLength    int
	Streaming    bool
}

type TextGenerationOutput struct {
	TokenStream chan backends.SequenceDelta
	ErrorStream chan error
	Responses   []string
}

func (t *TextGenerationOutput) GetOutput() []any {
	if t.TokenStream == nil && t.ErrorStream == nil {
		out := make([]any, len(t.Responses))
		for i, resp := range t.Responses {
			out[i] = any(resp)
		}
		return out
	}
	return []any{t.TokenStream, t.ErrorStream}
}

// WithSystemPrompt allows the user to define a system prompt that will be prepended to every input.
func WithSystemPrompt(systemPrompt string) backends.PipelineOption[*TextGenerationPipeline] {
	return func(pipeline *TextGenerationPipeline) error {
		pipeline.SystemPrompt = systemPrompt
		return nil
	}
}

// WithMaxLength allows the user to define the maximum generated tokens.
func WithMaxLength(maxLength int) backends.PipelineOption[*TextGenerationPipeline] {
	return func(pipeline *TextGenerationPipeline) error {
		pipeline.MaxLength = maxLength
		return nil
	}
}

func WithStreaming() backends.PipelineOption[*TextGenerationPipeline] {
	return func(pipeline *TextGenerationPipeline) error {
		pipeline.Streaming = true
		return nil
	}
}

// NewTextGenerationPipeline initializes a new text generation pipeline.
func NewTextGenerationPipeline(config backends.PipelineConfig[*TextGenerationPipeline], s *options.Options, model *backends.Model) (*TextGenerationPipeline, error) {
	defaultPipeline, err := backends.NewBasePipeline(config, s, model)
	if err != nil {
		return nil, err
	}
	pipeline := &TextGenerationPipeline{BasePipeline: defaultPipeline}
	for _, o := range config.Options {
		err = o(pipeline)
		if err != nil {
			return nil, err
		}
	}
	if pipeline.MaxLength == 0 {
		pipeline.MaxLength = 1028 // Default value if not set as per Python
	}
	err = pipeline.Validate()
	if err != nil {
		return nil, err
	}
	return pipeline, nil
}

// INTERFACE IMPLEMENTATION

func (p *TextGenerationPipeline) IsGenerative() bool {
	return true
}

func (p *TextGenerationPipeline) GetMetadata() backends.PipelineMetadata {
	return backends.PipelineMetadata{}
}

func (p *TextGenerationPipeline) GetModel() *backends.Model {
	return p.Model
}

// GetStatistics returns the runtime statistics for the pipeline.
func (p *TextGenerationPipeline) GetStatistics() backends.PipelineStatistics {
	generativeStatistics := p.Model.ORTModel.GenerativeSession.GetStatistics()
	return backends.PipelineStatistics{
		AvgPrefillSeconds:              generativeStatistics.AvgPrefillSeconds,
		TokensPerSecond:                generativeStatistics.TokensPerSecond,
		CumulativePrefillSum:           generativeStatistics.CumulativePrefillSum,
		CumulativePrefillCount:         generativeStatistics.CumulativePrefillCount,
		CumulativeTokens:               generativeStatistics.CumulativeTokens,
		CumulativeTokenDurationSeconds: generativeStatistics.CumulativeTokenDurationSeconds,
	}
}

func (p *TextGenerationPipeline) Validate() error {
	var validationErrors []error
	if !p.Model.IsGenerative {
		validationErrors = append(validationErrors, errors.New("model is not generative"))
	}

	if p.MaxLength <= 0 {
		validationErrors = append(validationErrors, errors.New("max length must be greater than zero"))
	}

	return errors.Join(validationErrors...)
}

func (p *TextGenerationPipeline) Preprocess(batch *backends.PipelineBatch, inputs any) error {
	return backends.CreateMessages(batch, p.BasePipeline, inputs, p.SystemPrompt)
}

// Forward initiates the generation loop.
func (p *TextGenerationPipeline) Forward(ctx context.Context, batch *backends.PipelineBatch) (chan backends.SequenceDelta, chan error, error) {
	tokenStream, errorStream, initErr := backends.RunGenerativeSessionOnBatch(ctx, batch, p.BasePipeline, p.MaxLength)
	if initErr != nil {
		return nil, nil, initErr
	}
	return tokenStream, errorStream, nil
}

func (p *TextGenerationPipeline) Run(inputs []string) (backends.PipelineBatchOutput, error) {
	return p.RunPipeline(context.Background(), inputs)
}

// RunPipeline processes a batch of string inputs.
func (p *TextGenerationPipeline) RunPipeline(ctx context.Context, inputs []string) (*TextGenerationOutput, error) {
	var runErrors []error
	batch := backends.NewBatch(len(inputs))
	batch.MaxNewTokens = p.MaxLength
	defer func(*backends.PipelineBatch) {
		runErrors = append(runErrors, batch.Destroy())
	}(batch)
	runErrors = append(runErrors, p.Preprocess(batch, inputs))
	if e := errors.Join(runErrors...); e != nil {
		return nil, e
	}
	tokenStream, errorStream, forwardErr := p.Forward(ctx, batch)
	if forwardErr != nil {
		return nil, forwardErr
	}
	if p.Streaming {
		return &TextGenerationOutput{
			TokenStream: tokenStream,
			ErrorStream: errorStream,
		}, errors.Join(runErrors...)
	}

	// Collect responses and errors
	responses, responseErr := collectResponses(tokenStream, errorStream, len(inputs))
	return &TextGenerationOutput{
		TokenStream: nil,
		ErrorStream: nil,
		Responses:   responses,
	}, errors.Join(append(runErrors, responseErr)...)
}

// RunMessages processes a batch of message inputs.
// If multimodal, the images should be added to the messages.
func (p *TextGenerationPipeline) RunMessages(ctx context.Context, inputs [][]backends.Message) (*TextGenerationOutput, error) {
	var runErrors []error
	batch := backends.NewBatch(len(inputs))
	batch.MaxNewTokens = p.MaxLength
	defer func(*backends.PipelineBatch) {
		runErrors = append(runErrors, batch.Destroy())
	}(batch)
	runErrors = append(runErrors, p.Preprocess(batch, inputs))
	if e := errors.Join(runErrors...); e != nil {
		return nil, e
	}
	tokenStream, errorStream, forwardErr := p.Forward(ctx, batch)
	if forwardErr != nil {
		return nil, forwardErr
	}
	if p.Streaming {
		return &TextGenerationOutput{
			TokenStream: tokenStream,
			ErrorStream: errorStream,
		}, errors.Join(runErrors...)
	}

	// Collect responses and errors
	responses, responseErr := collectResponses(tokenStream, errorStream, len(inputs))
	return &TextGenerationOutput{
		TokenStream: nil,
		ErrorStream: nil,
		Responses:   responses,
	}, errors.Join(append(runErrors, responseErr)...)
}

func collectResponses(tokenStream chan backends.SequenceDelta, errorStream chan error, batchSize int) ([]string, error) {
	responses := make([]string, batchSize)
	var finalErrors []error
	for delta := range tokenStream {
		index := delta.Index
		responses[index] += delta.Token
	}
	for err := range errorStream {
		finalErrors = append(finalErrors, err)
	}
	return responses, errors.Join(finalErrors...)
}
