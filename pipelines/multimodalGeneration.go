package pipelines

import (
	"context"
	"errors"
	"fmt"

	"github.com/knights-analytics/hugot/backends"
)

// ImageTextPrompt pairs an image file (or data URI) with a prompt.
type ImageTextPrompt struct {
	ImagePath string
	Prompt    string
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
	responses := make([]string, len(messages))
	for {
		select {
		case delta, ok := <-stream:
			if !ok {
				return &MultimodalTextOutput{Responses: responses}, batch.Destroy()
			}
			if delta.Sequence >= 0 && delta.Sequence < len(responses) {
				responses[delta.Sequence] += delta.Token
			}
		case err := <-errs:
			if err != nil {
				return nil, errors.Join(err, batch.Destroy())
			}
		}
	}
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
		messages[i] = []backends.Message{{Role: "user", Content: prompt, ImageURLs: []string{input.ImagePath}}}
	}
	return messages, nil
}

type (
	ImageToTextPipeline     struct{ *multimodalGeneration }
	ImageTextToTextPipeline struct{ *multimodalGeneration }
)

func (*ImageToTextPipeline) IsGenerative() bool     { return true }
func (*ImageTextToTextPipeline) IsGenerative() bool { return true }

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
	p := &ImageToTextPipeline{&multimodalGeneration{BasePipeline: backends.NewBasePipeline(ctx, config, model), MaxLength: 128}}
	for _, o := range config.Options {
		if err := o(p); err != nil {
			return nil, err
		}
	}
	return p, p.Validate()
}

func NewImageTextToTextPipeline(ctx context.Context, config ImageTextToTextConfig, model *backends.Model) (*ImageTextToTextPipeline, error) {
	p := &ImageTextToTextPipeline{&multimodalGeneration{BasePipeline: backends.NewBasePipeline(ctx, config, model), MaxLength: 128}}
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
