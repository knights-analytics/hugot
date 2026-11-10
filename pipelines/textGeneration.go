package pipelines

import (
	"context"
	"errors"

	"github.com/knights-analytics/hugot/backends"
)

type TextGenerationPipeline struct {
	*backends.BasePipeline
	Temperature   *float64
	TopP          *float64
	Seed          *int
	Guidance      *backends.Guidance
	SystemPrompt  string
	StopSequences []string
	Tools         []string
	MaxLength     int
	Streaming     bool
}

// TextGenerationCallOptions contains settings that can be changed for a single call.
type TextGenerationCallOptions struct {
	// MaxLength overrides the pipeline's configured maximum when greater than zero.
	// A zero value uses the pipeline maximum.
	MaxLength int
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

// WithEnableThinking controls the chat template's enable_thinking argument for
// every call made by this pipeline. Engine mode does not support this option.
func WithEnableThinking(enabled bool) backends.PipelineOption[*TextGenerationPipeline] {
	return func(pipeline *TextGenerationPipeline) error {
		pipeline.EnableThinking = &enabled
		return nil
	}
}

// WithStreaming allows the user to receive generated tokens as a stream instead of waiting for the entire response.
func WithStreaming() backends.PipelineOption[*TextGenerationPipeline] {
	return func(pipeline *TextGenerationPipeline) error {
		pipeline.Streaming = true
		return nil
	}
}

func WithTemperature(temperature float64) backends.PipelineOption[*TextGenerationPipeline] {
	return func(pipeline *TextGenerationPipeline) error {
		pipeline.Temperature = &temperature
		return nil
	}
}

func WithTopP(topP float64) backends.PipelineOption[*TextGenerationPipeline] {
	return func(pipeline *TextGenerationPipeline) error {
		pipeline.TopP = &topP
		return nil
	}
}

func WithSeed(seed int) backends.PipelineOption[*TextGenerationPipeline] {
	return func(pipeline *TextGenerationPipeline) error {
		pipeline.Seed = &seed
		return nil
	}
}

// WithStopSequences allows the user to define stop sequences that will end the generation when encountered.
// If the model produces any of the provided strings in the output, generation for that sequence will stop and the stop string will be excluded.
func WithStopSequences(stopSequences []string) backends.PipelineOption[*TextGenerationPipeline] {
	return func(pipeline *TextGenerationPipeline) error {
		pipeline.StopSequences = stopSequences
		return nil
	}
}

// WithTools sets the list of Hermes-style tool definition JSON strings to include in the chat template.
func WithTools(tools []string) backends.PipelineOption[*TextGenerationPipeline] {
	return func(pipeline *TextGenerationPipeline) error {
		pipeline.Tools = tools
		return nil
	}
}

// WithGuidance enables constrained (guided) generation using a lark grammar, JSON schema, or regex.
func WithGuidance(guidance *backends.Guidance) backends.PipelineOption[*TextGenerationPipeline] {
	return func(pipeline *TextGenerationPipeline) error {
		pipeline.Guidance = guidance
		return nil
	}
}

// NewTextGenerationPipeline initializes a new text generation pipeline.
func NewTextGenerationPipeline(sessionContext context.Context, config backends.PipelineConfig[*TextGenerationPipeline], model *backends.Model) (*TextGenerationPipeline, error) {
	defaultPipeline := backends.NewBasePipeline(sessionContext, config, model)
	var err error
	pipeline := &TextGenerationPipeline{BasePipeline: defaultPipeline}
	for _, o := range config.Options {
		err = o(pipeline)
		if err != nil {
			return nil, err
		}
	}
	if pipeline.MaxLength == 0 {
		pipeline.MaxLength = 4096 // Default value if not set
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
	if p.Model.ORTModel.Generative != nil {
		return p.Model.ORTModel.Generative.Statistics()
	}
	return backends.PipelineStatistics{}
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

func (p *TextGenerationPipeline) preprocess(batch *backends.PipelineBatch, inputs any) error {
	return backends.CreateMessages(batch, p.BasePipeline, inputs, p.SystemPrompt)
}

// forward initiates the generation loop with explicit tools and guidance, allowing per-call overrides.
func (p *TextGenerationPipeline) forward(ctx context.Context, batch *backends.PipelineBatch, maxLength int, tools []string, guidance *backends.Guidance) (chan backends.SequenceDelta, chan error, error) {
	effectiveTools := tools
	if len(effectiveTools) == 0 {
		effectiveTools = p.Tools
	}
	effectiveGuidance := guidance
	if effectiveGuidance == nil {
		effectiveGuidance = p.Guidance
	}
	tokenStream, errorStream, initErr := backends.RunGenerativeSessionOnBatch(ctx, batch, p.BasePipeline, maxLength, p.StopSequences, p.Temperature, p.TopP, p.Seed, effectiveTools, effectiveGuidance)
	if initErr != nil {
		return nil, nil, initErr
	}
	return tokenStream, errorStream, nil
}

func (p *TextGenerationPipeline) Run(ctx context.Context, inputs []string) (backends.PipelineBatchOutput, error) {
	return p.RunPipeline(ctx, inputs)
}

// RunWithOptions runs text prompts and returns the standard pipeline output with
// optional per-call overrides.
func (p *TextGenerationPipeline) RunWithOptions(ctx context.Context, inputs []string, options *TextGenerationCallOptions) (backends.PipelineBatchOutput, error) {
	return p.RunPipelineWithOptions(ctx, inputs, options)
}

func (p *TextGenerationPipeline) RunPipeline(ctx context.Context, inputs []string) (*TextGenerationOutput, error) {
	return p.RunPipelineWithOptions(ctx, inputs, nil)
}

// RunPipelineWithOptions runs text prompts with optional per-call overrides.
func (p *TextGenerationPipeline) RunPipelineWithOptions(ctx context.Context, inputs []string, options *TextGenerationCallOptions) (*TextGenerationOutput, error) {
	maxLength, err := p.callMaxLength(options)
	if err != nil {
		return nil, err
	}
	var runErrors []error
	batch := backends.NewBatch(len(inputs))
	batch.MaxNewTokens = maxLength
	runErrors = append(runErrors, p.preprocess(batch, inputs))
	if e := errors.Join(runErrors...); e != nil {
		return nil, errors.Join(e, batch.Destroy())
	}
	tokenStream, errorStream, forwardErr := p.forward(ctx, batch, maxLength, p.Tools, p.Guidance)
	if forwardErr != nil {
		return nil, errors.Join(forwardErr, batch.Destroy())
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

// runMessages processes a batch of message inputs.
// tools and guidance override the pipeline-level defaults for this call only.
// If the model produces any of the provided strings in the output, generation for that sequence will stop and the stop string will be excluded.
// If multimodal, the images should be added to the messages.
func (p *TextGenerationPipeline) runMessages(ctx context.Context, inputs [][]backends.Message, tools []string, guidance *backends.Guidance, options *TextGenerationCallOptions) (*TextGenerationOutput, error) {
	maxLength, err := p.callMaxLength(options)
	if err != nil {
		return nil, err
	}
	var runErrors []error
	batch := backends.NewBatch(len(inputs))
	batch.MaxNewTokens = maxLength
	runErrors = append(runErrors, p.preprocess(batch, inputs))
	if e := errors.Join(runErrors...); e != nil {
		return nil, errors.Join(e, batch.Destroy())
	}
	tokenStream, errorStream, forwardErr := p.forward(ctx, batch, maxLength, tools, guidance)
	if forwardErr != nil {
		return nil, errors.Join(forwardErr, batch.Destroy())
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

func (p *TextGenerationPipeline) RunMessages(ctx context.Context, inputs [][]backends.Message) (*TextGenerationOutput, error) {
	return p.runMessages(ctx, inputs, p.Tools, p.Guidance, nil)
}

// RunMessagesWithOptions runs message conversations with optional per-call overrides.
func (p *TextGenerationPipeline) RunMessagesWithOptions(ctx context.Context, inputs [][]backends.Message, options *TextGenerationCallOptions) (*TextGenerationOutput, error) {
	return p.runMessages(ctx, inputs, p.Tools, p.Guidance, options)
}

func (p *TextGenerationPipeline) RunMessagesWithOverrides(ctx context.Context, inputs [][]backends.Message, tools []string, guidance *backends.Guidance) (*TextGenerationOutput, error) {
	return p.runMessages(ctx, inputs, tools, guidance, nil)
}

func (p *TextGenerationPipeline) callMaxLength(options *TextGenerationCallOptions) (int, error) {
	if options == nil || options.MaxLength == 0 {
		return p.MaxLength, nil
	}
	if options.MaxLength < 0 {
		return 0, errors.New("max length must be greater than zero")
	}
	return options.MaxLength, nil
}

// GetContextLength returns the model's configured context length.
func (p *TextGenerationPipeline) GetContextLength() int {
	if p == nil || p.Model == nil {
		return 0
	}
	return p.Model.ContextLength
}

func collectResponses(tokenStream chan backends.SequenceDelta, errorStream chan error, batchSize int) ([]string, error) {
	responses := make([]string, batchSize)
	var finalErrors []error
	for delta := range tokenStream {
		responses[delta.Sequence] += delta.Token
	}
	for err := range errorStream {
		finalErrors = append(finalErrors, err)
	}
	return responses, errors.Join(finalErrors...)
}
