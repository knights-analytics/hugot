package pipelines

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/knights-analytics/hugot/backends"
)

// VisualQuestionAnsweringInput associates an image with a question.
type VisualQuestionAnsweringInput struct {
	ImagePath string
	Question  string
}

// DocumentQuestionAnsweringInput associates a document image with a question.
// ImagePath may be a local image path or a data URI supported by the backend.
type DocumentQuestionAnsweringInput struct {
	DocumentPath string
	ImagePath    string
	Question     string
}

// TableQuestionAnsweringInput is one table question. The first row is treated
// as the table header when the model receives the serialized representation.
type TableQuestionAnsweringInput struct {
	Table    [][]string
	Question string
	Query    string
}

// QuestionAnsweringTextOutput contains one generated answer per input.
type QuestionAnsweringTextOutput struct {
	Responses   []string
	TokenStream chan backends.SequenceDelta
	ErrorStream chan error
}

type (
	VisualQuestionAnsweringOutput   = QuestionAnsweringTextOutput
	DocumentQuestionAnsweringOutput = QuestionAnsweringTextOutput
	TableQuestionAnsweringOutput    = QuestionAnsweringTextOutput
)

func (o *QuestionAnsweringTextOutput) GetOutput() []any {
	if o.TokenStream != nil || o.ErrorStream != nil {
		return []any{o.TokenStream, o.ErrorStream}
	}
	out := make([]any, len(o.Responses))
	for i, response := range o.Responses {
		out[i] = response
	}
	return out
}

type questionAnsweringFamily struct{ *multimodalGeneration }

func (p questionAnsweringFamily) IsGenerative() bool { return true }

func (p *questionAnsweringFamily) validate(kind string) error {
	if p == nil || p.multimodalGeneration == nil || p.Model == nil {
		return fmt.Errorf("%s pipeline requires a model", kind)
	}
	if !p.Model.IsGenerative {
		return fmt.Errorf("%s pipeline requires a generative model", kind)
	}
	if p.MaxLength <= 0 {
		return errors.New("max length must be greater than zero")
	}
	return nil
}

func (p *questionAnsweringFamily) runText(ctx context.Context, messages [][]backends.Message) (*QuestionAnsweringTextOutput, error) {
	out, err := p.run(ctx, messages)
	if err != nil {
		return nil, err
	}
	return &QuestionAnsweringTextOutput{
		Responses:   out.Responses,
		TokenStream: out.TokenStream,
		ErrorStream: out.ErrorStream,
	}, nil
}

// VisualQuestionAnsweringPipeline answers questions about images with a
// generative multimodal model.
type VisualQuestionAnsweringPipeline struct{ questionAnsweringFamily }

// DocumentQuestionAnsweringPipeline answers questions about document images.
type DocumentQuestionAnsweringPipeline struct{ questionAnsweringFamily }

// TableQuestionAnsweringPipeline answers questions about serialized tables.
type TableQuestionAnsweringPipeline struct{ questionAnsweringFamily }

func (*VisualQuestionAnsweringPipeline) IsGenerative() bool   { return true }
func (*DocumentQuestionAnsweringPipeline) IsGenerative() bool { return true }
func (*TableQuestionAnsweringPipeline) IsGenerative() bool    { return true }

type (
	VisualQuestionAnsweringConfig   = backends.PipelineConfig[*VisualQuestionAnsweringPipeline]
	VisualQuestionAnsweringOption   = backends.PipelineOption[*VisualQuestionAnsweringPipeline]
	DocumentQuestionAnsweringConfig = backends.PipelineConfig[*DocumentQuestionAnsweringPipeline]
	DocumentQuestionAnsweringOption = backends.PipelineOption[*DocumentQuestionAnsweringPipeline]
	TableQuestionAnsweringConfig    = backends.PipelineConfig[*TableQuestionAnsweringPipeline]
	TableQuestionAnsweringOption    = backends.PipelineOption[*TableQuestionAnsweringPipeline]
)

func setQAMaxLength(n int) func(*questionAnsweringFamily) error {
	return func(p *questionAnsweringFamily) error {
		if n <= 0 {
			return errors.New("max length must be greater than zero")
		}
		p.MaxLength = n
		return nil
	}
}

func WithVisualQuestionAnsweringMaxLength(n int) VisualQuestionAnsweringOption {
	return func(p *VisualQuestionAnsweringPipeline) error { return setQAMaxLength(n)(&p.questionAnsweringFamily) }
}

func WithDocumentQuestionAnsweringMaxLength(n int) DocumentQuestionAnsweringOption {
	return func(p *DocumentQuestionAnsweringPipeline) error { return setQAMaxLength(n)(&p.questionAnsweringFamily) }
}

func WithTableQuestionAnsweringMaxLength(n int) TableQuestionAnsweringOption {
	return func(p *TableQuestionAnsweringPipeline) error { return setQAMaxLength(n)(&p.questionAnsweringFamily) }
}

func WithVisualQuestionAnsweringStreaming() VisualQuestionAnsweringOption {
	return func(p *VisualQuestionAnsweringPipeline) error { p.Streaming = true; return nil }
}

func WithDocumentQuestionAnsweringStreaming() DocumentQuestionAnsweringOption {
	return func(p *DocumentQuestionAnsweringPipeline) error { p.Streaming = true; return nil }
}

func WithTableQuestionAnsweringStreaming() TableQuestionAnsweringOption {
	return func(p *TableQuestionAnsweringPipeline) error { p.Streaming = true; return nil }
}

func newQAPipeline[T backends.Pipeline](ctx context.Context, config backends.PipelineConfig[T], model *backends.Model) *multimodalGeneration {
	return &multimodalGeneration{BasePipeline: backends.NewBasePipeline(ctx, config, model), MaxLength: 128}
}

func NewVisualQuestionAnsweringPipeline(ctx context.Context, config VisualQuestionAnsweringConfig, model *backends.Model) (*VisualQuestionAnsweringPipeline, error) {
	if model == nil {
		return nil, errors.New("visual question answering pipeline requires a model")
	}
	if !model.IsGenerative {
		return nil, errors.New("visual question answering pipeline requires a generative model")
	}
	p := &VisualQuestionAnsweringPipeline{questionAnsweringFamily{newQAPipeline(ctx, config, model)}}
	for _, option := range config.Options {
		if err := option(p); err != nil {
			return nil, err
		}
	}
	return p, p.validate("visual question answering")
}

func NewDocumentQuestionAnsweringPipeline(ctx context.Context, config DocumentQuestionAnsweringConfig, model *backends.Model) (*DocumentQuestionAnsweringPipeline, error) {
	if model == nil {
		return nil, errors.New("document question answering pipeline requires a model")
	}
	if !model.IsGenerative {
		return nil, errors.New("document question answering pipeline requires a generative model")
	}
	p := &DocumentQuestionAnsweringPipeline{questionAnsweringFamily{newQAPipeline(ctx, config, model)}}
	for _, option := range config.Options {
		if err := option(p); err != nil {
			return nil, err
		}
	}
	return p, p.validate("document question answering")
}

func NewTableQuestionAnsweringPipeline(ctx context.Context, config TableQuestionAnsweringConfig, model *backends.Model) (*TableQuestionAnsweringPipeline, error) {
	if model == nil {
		return nil, errors.New("table question answering pipeline requires a model")
	}
	if !model.IsGenerative {
		return nil, errors.New("table question answering pipeline requires a generative model")
	}
	p := &TableQuestionAnsweringPipeline{questionAnsweringFamily{newQAPipeline(ctx, config, model)}}
	for _, option := range config.Options {
		if err := option(p); err != nil {
			return nil, err
		}
	}
	return p, p.validate("table question answering")
}

func (p *VisualQuestionAnsweringPipeline) Validate() error {
	return p.validate("visual question answering")
}

func (p *DocumentQuestionAnsweringPipeline) Validate() error {
	return p.validate("document question answering")
}

func (p *TableQuestionAnsweringPipeline) Validate() error {
	return p.validate("table question answering")
}

func (p *VisualQuestionAnsweringPipeline) Run(ctx context.Context, inputs []string) (backends.PipelineBatchOutput, error) {
	if len(inputs)%2 != 0 || len(inputs) == 0 {
		return nil, errors.New("visual question answering requires [image, question] pairs")
	}
	values := make([]VisualQuestionAnsweringInput, len(inputs)/2)
	for i := range values {
		values[i] = VisualQuestionAnsweringInput{ImagePath: inputs[2*i], Question: inputs[2*i+1]}
	}
	return p.RunPipeline(ctx, values)
}

func (p *DocumentQuestionAnsweringPipeline) Run(ctx context.Context, inputs []string) (backends.PipelineBatchOutput, error) {
	if len(inputs)%2 != 0 || len(inputs) == 0 {
		return nil, errors.New("document question answering requires [document, question] pairs")
	}
	values := make([]DocumentQuestionAnsweringInput, len(inputs)/2)
	for i := range values {
		values[i] = DocumentQuestionAnsweringInput{DocumentPath: inputs[2*i], Question: inputs[2*i+1]}
	}
	return p.RunPipeline(ctx, values)
}

func (p *TableQuestionAnsweringPipeline) Run(ctx context.Context, inputs []string) (backends.PipelineBatchOutput, error) {
	values := make([]TableQuestionAnsweringInput, len(inputs))
	for i, input := range inputs {
		if err := json.Unmarshal([]byte(input), &values[i]); err != nil {
			return nil, fmt.Errorf("table question answering input %d is not valid JSON: %w", i, err)
		}
	}
	return p.RunPipeline(ctx, values)
}

func (p *VisualQuestionAnsweringPipeline) RunPipeline(ctx context.Context, inputs []VisualQuestionAnsweringInput) (*QuestionAnsweringTextOutput, error) {
	messages := make([][]backends.Message, len(inputs))
	for i, input := range inputs {
		if strings.TrimSpace(input.ImagePath) == "" || strings.TrimSpace(input.Question) == "" {
			return nil, fmt.Errorf("visual question answering input %d requires image path and question", i)
		}
		messages[i] = []backends.Message{{Role: "user", Content: input.Question, ImageURLs: []string{input.ImagePath}}}
	}
	return p.runText(ctx, messages)
}

func (p *DocumentQuestionAnsweringPipeline) RunPipeline(ctx context.Context, inputs []DocumentQuestionAnsweringInput) (*QuestionAnsweringTextOutput, error) {
	values := make([]VisualQuestionAnsweringInput, len(inputs))
	for i, input := range inputs {
		path := input.DocumentPath
		if path == "" {
			path = input.ImagePath
		}
		values[i] = VisualQuestionAnsweringInput{ImagePath: path, Question: input.Question}
	}
	return (&VisualQuestionAnsweringPipeline{p.questionAnsweringFamily}).RunPipeline(ctx, values)
}

func (p *TableQuestionAnsweringPipeline) RunPipeline(ctx context.Context, inputs []TableQuestionAnsweringInput) (*QuestionAnsweringTextOutput, error) {
	messages := make([][]backends.Message, len(inputs))
	for i, input := range inputs {
		question := input.Question
		if question == "" {
			question = input.Query
		}
		if len(input.Table) == 0 || strings.TrimSpace(question) == "" {
			return nil, fmt.Errorf("table question answering input %d requires a table and question", i)
		}
		table, err := json.Marshal(input.Table)
		if err != nil {
			return nil, fmt.Errorf("serialize table %d: %w", i, err)
		}
		messages[i] = []backends.Message{{Role: "user", Content: "Answer the question using this table:\n" + string(table) + "\nQuestion: " + question}}
	}
	return p.runText(ctx, messages)
}
