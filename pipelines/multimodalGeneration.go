package pipelines

import (
	"context"
	"errors"
	"fmt"

	"github.com/knights-analytics/hugot/backends"
)

const defaultMultimodalMaxLength = 4096

// ImageTextPrompt pairs an image file (or data URI) with a prompt.
//
// By default (all zero values, as before) a prompt describes a single user
// turn: the pipeline sends one message with Role "user", the Prompt as its
// content, and ImagePath as the attached image. That keeps the one-shot
// image->text and image+text->text workflows backward compatible.
//
// For multi-turn image/text conversations, set Role to the role the final
// (image) turn should carry, and add History for any previous turns in this
// conversation. They are emitted in order — History, then the final image
// turn — so the underlying backend sees a full, correctly labelled conversation.
// If a per-conversation system turn is needed, put it in History[0]; the
// pipeline-level SystemPrompt on the pipeline struct (set once at construction)
// is still applied globally by the backend.
type ImageTextPrompt struct {
	ImagePath string
	Prompt    string
	Role      string
	History   []backends.Message
}

type MultimodalTextOutput struct {
	Responses   []string
	TokenStream chan backends.SequenceDelta
	ErrorStream chan error
}

func (o *MultimodalTextOutput) GetOutput() []any {
	if o.TokenStream == nil {
		out := make([]any, len(o.Responses))
		for i, v := range o.Responses {
			out[i] = v
		}
		return out
	}
	return []any{o.TokenStream, o.ErrorStream}
}

type multimodalGeneration struct {
	*backends.BasePipeline
	MaxLength    int
	Temperature  *float64
	TopP         *float64
	Seed         *int
	SystemPrompt string
	Streaming    bool
}

func (p *multimodalGeneration) IsGenerative() bool { return true }
func (p *multimodalGeneration) Validate() error {
	if !p.Model.IsGenerative {
		return errors.New("multimodal pipeline requires a generative model")
	}
	if p.MaxLength <= 0 {
		return errors.New("max length must be greater than zero")
	}
	return nil
}
func (p *multimodalGeneration) GetModel() *backends.Model { return p.Model }
func (p *multimodalGeneration) GetMetadata() backends.PipelineMetadata {
	return backends.PipelineMetadata{}
}

func (p *multimodalGeneration) GetStatistics() backends.PipelineStatistics {
	if p.Model.ORTModel != nil && p.Model.ORTModel.Generative != nil {
		return p.Model.ORTModel.Generative.Statistics()
	}
	return backends.PipelineStatistics{}
}

func (p *multimodalGeneration) run(ctx context.Context, messages [][]backends.Message) (*MultimodalTextOutput, error) {
	if len(messages) == 0 {
		return nil, errors.New("at least one multimodal input is required")
	}
	batch := backends.NewBatch(len(messages))
	if err := backends.CreateMessages(batch, p.BasePipeline, messages, p.SystemPrompt); err != nil {
		return nil, errors.Join(err, batch.Destroy())
	}
	stream, errs, err := backends.RunGenerativeSessionOnBatch(ctx, batch, p.BasePipeline, p.MaxLength, nil, p.Temperature, p.TopP, p.Seed, nil, nil)
	if err != nil {
		return nil, errors.Join(err, batch.Destroy())
	}
	if p.Streaming {
		return &MultimodalTextOutput{TokenStream: stream, ErrorStream: errs}, nil
	}
	responses, collectErr := collectMultimodalResponses(stream, errs, len(messages))
	return &MultimodalTextOutput{Responses: responses}, collectErr
}

func collectMultimodalResponses(stream <-chan backends.SequenceDelta, errs <-chan error, batchSize int) ([]string, error) {
	responses := make([]string, batchSize)
	var collectedErrors []error
	for stream != nil || errs != nil {
		select {
		case delta, ok := <-stream:
			if !ok {
				stream = nil
				continue
			}
			if delta.Sequence >= 0 && delta.Sequence < len(responses) {
				responses[delta.Sequence] += delta.Token
			}
		case err, ok := <-errs:
			if !ok {
				errs = nil
				continue
			}
			if err != nil {
				collectedErrors = append(collectedErrors, err)
			}
		}
	}
	return responses, errors.Join(collectedErrors...)
}

func multimodalMessages(inputs []ImageTextPrompt, defaultPrompt string) ([][]backends.Message, error) {
	messages := make([][]backends.Message, len(inputs))
	for i, input := range inputs {
		if input.ImagePath == "" {
			return nil, fmt.Errorf("multimodal input %d has no image path", i)
		}
		prompt := input.Prompt
		if prompt == "" {
			prompt = defaultPrompt
		}
		role := input.Role
		if role == "" {
			role = "user"
		}
		conv := make([]backends.Message, 0, 1+len(input.History))
		conv = append(conv, input.History...)
		conv = append(conv, backends.Message{Role: role, Content: prompt, ImageURLs: []string{input.ImagePath}})
		messages[i] = conv
	}
	return messages, nil
}

type (
	ImageToTextPipeline struct {
		*multimodalGeneration
		native *nativeImageCaptioning
	}
	ImageTextToTextPipeline struct{ *multimodalGeneration }
)

func (p *ImageToTextPipeline) IsGenerative() bool   { return p == nil || p.native == nil }
func (*ImageTextToTextPipeline) IsGenerative() bool { return true }

func (p *ImageToTextPipeline) Validate() error {
	if p == nil {
		return errors.New("image to text pipeline is nil")
	}
	if p.native != nil {
		return p.validateNativeCaption()
	}
	return p.multimodalGeneration.Validate()
}

type (
	ImageToTextConfig     = backends.PipelineConfig[*ImageToTextPipeline]
	ImageToTextOption     = backends.PipelineOption[*ImageToTextPipeline]
	ImageTextToTextConfig = backends.PipelineConfig[*ImageTextToTextPipeline]
	ImageTextToTextOption = backends.PipelineOption[*ImageTextToTextPipeline]
)

func WithImageToTextMaxLength(n int) ImageToTextOption {
	return func(p *ImageToTextPipeline) error {
		if n <= 0 {
			return errors.New("max length must be greater than zero")
		}
		p.MaxLength = n
		return nil
	}
}

func WithImageTextToTextMaxLength(n int) ImageTextToTextOption {
	return func(p *ImageTextToTextPipeline) error {
		if n <= 0 {
			return errors.New("max length must be greater than zero")
		}
		p.MaxLength = n
		return nil
	}
}

func WithImageToTextStreaming() ImageToTextOption {
	return func(p *ImageToTextPipeline) error { p.Streaming = true; return nil }
}

func WithImageTextToTextStreaming() ImageTextToTextOption {
	return func(p *ImageTextToTextPipeline) error { p.Streaming = true; return nil }
}

func NewImageToTextPipeline(ctx context.Context, config ImageToTextConfig, model *backends.Model) (*ImageToTextPipeline, error) {
	if model == nil {
		return nil, errors.New("image-to-text requires a model")
	}
	p := &ImageToTextPipeline{multimodalGeneration: &multimodalGeneration{BasePipeline: backends.NewBasePipeline(ctx, config, model), MaxLength: defaultMultimodalMaxLength}}
	if !model.IsGenerative {
		var err error
		p.native, err = newNativeImageCaptioning(ctx, model)
		if err != nil {
			return nil, err
		}
		p.MaxLength = 20
	}
	for _, o := range config.Options {
		if err := o(p); err != nil {
			return nil, err
		}
	}
	return p, p.Validate()
}

func NewImageTextToTextPipeline(ctx context.Context, config ImageTextToTextConfig, model *backends.Model) (*ImageTextToTextPipeline, error) {
	p := &ImageTextToTextPipeline{&multimodalGeneration{BasePipeline: backends.NewBasePipeline(ctx, config, model), MaxLength: defaultMultimodalMaxLength}}
	for _, o := range config.Options {
		if err := o(p); err != nil {
			return nil, err
		}
	}
	return p, p.Validate()
}

func (p *ImageToTextPipeline) Run(ctx context.Context, inputs []string) (backends.PipelineBatchOutput, error) {
	values := make([]ImageTextPrompt, len(inputs))
	for i, path := range inputs {
		values[i] = ImageTextPrompt{ImagePath: path}
	}
	return p.RunWithImages(ctx, values)
}

func (p *ImageToTextPipeline) RunWithImages(ctx context.Context, inputs []ImageTextPrompt) (*MultimodalTextOutput, error) {
	if p.native != nil {
		return p.runNativeCaption(ctx, inputs)
	}
	messages, err := multimodalMessages(inputs, "Describe this image.")
	if err != nil {
		return nil, err
	}
	return p.run(ctx, messages)
}

func (p *ImageTextToTextPipeline) Run(ctx context.Context, inputs []string) (backends.PipelineBatchOutput, error) {
	values := make([]ImageTextPrompt, len(inputs))
	for i, path := range inputs {
		values[i] = ImageTextPrompt{ImagePath: path}
	}
	return p.RunWithImages(ctx, values)
}

func (p *ImageTextToTextPipeline) RunWithImages(ctx context.Context, inputs []ImageTextPrompt) (*MultimodalTextOutput, error) {
	messages, err := multimodalMessages(inputs, "")
	if err != nil {
		return nil, err
	}
	return p.run(ctx, messages)
}
