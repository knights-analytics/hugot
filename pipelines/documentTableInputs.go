package pipelines

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/png"
	"os"

	"github.com/knights-analytics/hugot/backends"
)

// DocumentQuestionAnsweringInput pairs a document with a question. Native
// Donut inference uses DocumentPath (or ImagePath as a fallback) and Question.
type DocumentQuestionAnsweringInput struct {
	DocumentPath string
	ImagePath    string
	Question     string
	Role         string
	History      []backends.Message
}

// TableQuestionAnsweringInput pairs a table (header first) with a question.
// Query is a fallback for an empty Question. Native TAPAS uses the table/query.
type TableQuestionAnsweringInput struct {
	Table    [][]string
	Question string
	Query    string
	Role     string
	History  []backends.Message
}

type (
	DocumentQuestionAnsweringConfig = backends.PipelineConfig[*DocumentQuestionAnsweringPipeline]
	DocumentQuestionAnsweringOption = backends.PipelineOption[*DocumentQuestionAnsweringPipeline]
	TableQuestionAnsweringConfig    = backends.PipelineConfig[*TableQuestionAnsweringPipeline]
	TableQuestionAnsweringOption    = backends.PipelineOption[*TableQuestionAnsweringPipeline]
)

func WithDocumentQuestionAnsweringMaxLength(n int) DocumentQuestionAnsweringOption {
	return func(p *DocumentQuestionAnsweringPipeline) error {
		if n <= 0 {
			return errors.New("max length must be positive")
		}
		p.MaxLength = n
		return nil
	}
}

func WithTableQuestionAnsweringMaxLength(n int) TableQuestionAnsweringOption {
	return func(p *TableQuestionAnsweringPipeline) error {
		if n <= 0 {
			return errors.New("max length must be positive")
		}
		p.MaxLength = n
		return nil
	}
}

func WithDocumentQuestionAnsweringStreaming() DocumentQuestionAnsweringOption {
	return func(_ *DocumentQuestionAnsweringPipeline) error {
		return errors.New("native document QA returns answer results and does not support streaming")
	}
}

func WithTableQuestionAnsweringStreaming() TableQuestionAnsweringOption {
	return func(_ *TableQuestionAnsweringPipeline) error {
		return errors.New("native table QA returns cell results and does not support streaming")
	}
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

func (p *DocumentQuestionAnsweringPipeline) RunWithImages(ctx context.Context, inputs []backends.ImageTextInput) (*DocumentQuestionAnsweringOutput, error) {
	values, cleanup, err := documentQuestionAnsweringRunWithImages(inputs)
	if err != nil {
		return nil, err
	}
	defer cleanup()
	return p.RunPipeline(ctx, values)
}
