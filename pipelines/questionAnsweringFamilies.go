package pipelines

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/png"
	"os"
	"strings"

	"github.com/knights-analytics/hugot/backends"
)

// VisualQuestionAnsweringInput associates an image with a question.
//
// By default (all zero values, as before) it produces a single user turn with
// the image attached — see ImageTextPrompt in multimodalGeneration.go for the
// shared conventions. Set Role and History for multi-turn conversations.
type VisualQuestionAnsweringInput struct {
	ImagePath string
	Question  string
	Role      string
	History   []backends.Message
}

// DocumentQuestionAnsweringInput associates a document image with a question.
// ImagePath may be a local image path or a data URI supported by the backend.
//
// By default it produces a single user turn with the document image attached.
// Set Role and History for multi-turn conversations.
type DocumentQuestionAnsweringInput struct {
	DocumentPath string
	ImagePath    string
	Question     string
	Role         string
	History      []backends.Message
}

// TableQuestionAnsweringInput is one table question. The first row is treated
// as the table header when the model receives the serialized representation.
// Question is the question to answer; Query is an optional fallback that is
// used when Question is empty, for callers that follow the "query" naming.
//
// By default it produces a single user turn whose content is the serialized
// table plus the question. Set Role and History for multi-turn conversations.
type TableQuestionAnsweringInput struct {
	Table    [][]string
	Question string
	Query    string
	Role     string
	History  []backends.Message
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

func (p *questionAnsweringFamily) IsGenerative() bool { return true }

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
	if err != nil || out == nil {
		return nil, err
	}
	return &QuestionAnsweringTextOutput{
		Responses:   out.Responses,
		TokenStream: out.TokenStream,
		ErrorStream: out.ErrorStream,
	}, err
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
	return &multimodalGeneration{BasePipeline: backends.NewBasePipeline(ctx, config, model), MaxLength: defaultMultimodalMaxLength}
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

// qaMessages assembles a per-input conversation: the input's History in order,
// then a final turn with the resolved role (default "user") and the question.
// This is the shared multi-turn convention used by every question-answering
// input, matching ImageTextPrompt in multimodalGeneration.go.
func qaMessages(history []backends.Message, role, content string, imagePaths ...string) []backends.Message {
	role = qaDefaultRole(role)
	conv := make([]backends.Message, 0, 1+len(history))
	conv = append(conv, history...)
	final := backends.Message{Role: role, Content: content}
	if imagePaths != nil {
		final.ImageURLs = append([]string(nil), imagePaths...)
	}
	conv = append(conv, final)
	return conv
}

func qaDefaultRole(role string) string {
	if role == "" {
		return "user"
	}
	return role
}

// writeQATempImage materializes an in-memory image.Image as a temp PNG in
// os.TempDir(). The returned cleanup function removes the file; callers should
// also call it on the error path (it is idempotent). Only called by the
// RunWithImages paths below; not part of the public pipeline API.
func writeQATempImage(src image.Image) (string, func(), error) {
	f, err := os.CreateTemp("", "QA*.png")
	if err != nil {
		return "", func() {}, fmt.Errorf("create temp image file: %w", err)
	}
	path := f.Name()
	cleanup := func() { _ = os.Remove(path); _ = f.Close() }
	if err := png.Encode(f, src); err != nil {
		cleanup()
		return "", func() {}, fmt.Errorf("encode temp image: %w", err)
	}
	if err := f.Close(); err != nil {
		cleanup()
		return "", func() {}, fmt.Errorf("close temp image file: %w", err)
	}
	return path, func() { _ = os.Remove(path) }, nil
}

// visualQuestionAnsweringRunWithImages adapts []backends.ImageTextInput to
// []VisualQuestionAnsweringInput. An in-memory image.Image is written to a
// temp PNG (cleaned up by the returned function); an ImagePath value is used
// as-is. Returns the adapted inputs, a cleanup func, and any error.
func visualQuestionAnsweringRunWithImages(inputs []backends.ImageTextInput) ([]VisualQuestionAnsweringInput, func(), error) {
	tmp := make([]string, 0, len(inputs))
	values := make([]VisualQuestionAnsweringInput, len(inputs))
	fail := func(i int, err error) ([]VisualQuestionAnsweringInput, func(), error) {
		for _, p := range tmp {
			_ = os.Remove(p)
		}
		return nil, func() {}, fmt.Errorf("visual question answering input %d: %w", i, err)
	}
	for i, in := range inputs {
		if err := in.Validate(); err != nil {
			return fail(i, err)
		}
		var path string
		if in.ImagePath != "" {
			path = in.ImagePath
		} else {
			p, _, err := writeQATempImage(in.Image)
			if err != nil {
				return fail(i, err)
			}
			path = p
			tmp = append(tmp, path)
		}
		values[i] = VisualQuestionAnsweringInput{ImagePath: path, Question: in.Text}
	}
	cleanup := func() {
		for _, p := range tmp {
			_ = os.Remove(p)
		}
	}
	return values, cleanup, nil
}

// documentQuestionAnsweringRunWithImages is the DocQA counterpart of
// visualQuestionAnsweringRunWithImages; it fills DocumentPath rather than
// ImagePath (ImagePipeline's RunPipeline prefers DocumentPath over ImagePath,
// so this is the semantically correct shape for document QA).
func documentQuestionAnsweringRunWithImages(inputs []backends.ImageTextInput) ([]DocumentQuestionAnsweringInput, func(), error) {
	tmp := make([]string, 0, len(inputs))
	values := make([]DocumentQuestionAnsweringInput, len(inputs))
	fail := func(i int, err error) ([]DocumentQuestionAnsweringInput, func(), error) {
		for _, p := range tmp {
			_ = os.Remove(p)
		}
		return nil, func() {}, fmt.Errorf("document question answering input %d: %w", i, err)
	}
	for i, in := range inputs {
		if err := in.Validate(); err != nil {
			return fail(i, err)
		}
		var path string
		if in.ImagePath != "" {
			path = in.ImagePath
		} else {
			p, _, err := writeQATempImage(in.Image)
			if err != nil {
				return fail(i, err)
			}
			path = p
			tmp = append(tmp, path)
		}
		values[i] = DocumentQuestionAnsweringInput{DocumentPath: path, Question: in.Text}
	}
	cleanup := func() {
		for _, p := range tmp {
			_ = os.Remove(p)
		}
	}
	return values, cleanup, nil
}

func visualQuestionAnsweringMessages(inputs []VisualQuestionAnsweringInput) ([][]backends.Message, error) {
	messages := make([][]backends.Message, len(inputs))
	for i, input := range inputs {
		if strings.TrimSpace(input.ImagePath) == "" || strings.TrimSpace(input.Question) == "" {
			return nil, fmt.Errorf("visual question answering input %d requires image path and question", i)
		}
		messages[i] = qaMessages(input.History, input.Role, input.Question, input.ImagePath)
	}
	return messages, nil
}

func tableQuestionAnsweringMessages(inputs []TableQuestionAnsweringInput) ([][]backends.Message, error) {
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
		content := "Answer the question using this table:\n" + string(table) + "\nQuestion: " + question
		messages[i] = qaMessages(input.History, input.Role, content)
	}
	return messages, nil
}

// RunPipeline runs the pipeline over a batch of prepared inputs, mapping each
// to a multimodal conversation whose trailing turn carries the question and
// the image path.
func (p *VisualQuestionAnsweringPipeline) RunPipeline(ctx context.Context, inputs []VisualQuestionAnsweringInput) (*QuestionAnsweringTextOutput, error) {
	messages, err := visualQuestionAnsweringMessages(inputs)
	if err != nil {
		return nil, err
	}
	return p.runText(ctx, messages)
}

// RunWithImages runs the pipeline on a batch of in-memory image + text pairs.
// An in-memory image.Image is written to a temp PNG (cleaned up after the call
// returns); an ImagePath value is used as-is. This is the in-memory
// counterpart to Run(ctx, []string) which loads from paths.
func (p *VisualQuestionAnsweringPipeline) RunWithImages(ctx context.Context, inputs []backends.ImageTextInput) (*QuestionAnsweringTextOutput, error) {
	values, cleanup, err := visualQuestionAnsweringRunWithImages(inputs)
	if err != nil {
		return nil, err
	}
	defer cleanup()
	return p.RunPipeline(ctx, values)
}

// RunWithImages runs the pipeline on a batch of in-memory image + text pairs,
// treating each image as a document. An in-memory image.Image is written to a
// temp PNG (cleaned up after the call returns); an ImagePath value is used
// as-is. This is the in-memory counterpart to Run(ctx, []string) which loads
// from paths.
func (p *DocumentQuestionAnsweringPipeline) RunWithImages(ctx context.Context, inputs []backends.ImageTextInput) (*QuestionAnsweringTextOutput, error) {
	values, cleanup, err := documentQuestionAnsweringRunWithImages(inputs)
	if err != nil {
		return nil, err
	}
	defer cleanup()
	return p.RunPipeline(ctx, values)
}

func (p *DocumentQuestionAnsweringPipeline) RunPipeline(ctx context.Context, inputs []DocumentQuestionAnsweringInput) (*QuestionAnsweringTextOutput, error) {
	values := make([]VisualQuestionAnsweringInput, len(inputs))
	for i, input := range inputs {
		path := input.DocumentPath
		if path == "" {
			path = input.ImagePath
		}
		values[i] = VisualQuestionAnsweringInput{ImagePath: path, Question: input.Question, Role: input.Role, History: input.History}
	}
	messages, err := visualQuestionAnsweringMessages(values)
	if err != nil {
		return nil, err
	}
	return p.runText(ctx, messages)
}

func (p *TableQuestionAnsweringPipeline) RunPipeline(ctx context.Context, inputs []TableQuestionAnsweringInput) (*QuestionAnsweringTextOutput, error) {
	messages, err := tableQuestionAnsweringMessages(inputs)
	if err != nil {
		return nil, err
	}
	return p.runText(ctx, messages)
}
