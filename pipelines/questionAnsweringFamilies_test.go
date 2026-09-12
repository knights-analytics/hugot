package pipelines

import (
	"context"
	"testing"

	"github.com/knights-analytics/hugot/backends"
)

func TestQuestionAnsweringFamiliesValidateRequiresGenerativeModel(t *testing.T) {
	model := &backends.Model{}
	config := VisualQuestionAnsweringConfig{}
	pipeline, err := NewVisualQuestionAnsweringPipeline(context.Background(), config, model)
	if err == nil || pipeline != nil {
		t.Fatal("expected visual QA validation to reject a non-generative model")
	}
}

func TestQuestionAnsweringFamiliesValidateNilModel(t *testing.T) {
	if _, err := NewTableQuestionAnsweringPipeline(context.Background(), TableQuestionAnsweringConfig{}, nil); err == nil {
		t.Fatal("expected table QA validation to reject a nil model")
	}
}

func TestQuestionAnsweringFamiliesRunInputValidation(t *testing.T) {
	pipeline := &VisualQuestionAnsweringPipeline{}
	if _, err := pipeline.Run(context.Background(), []string{"image.png"}); err == nil {
		t.Fatal("expected an image/question pair validation error")
	}
	if _, err := (&TableQuestionAnsweringPipeline{}).Run(context.Background(), []string{"not-json"}); err == nil {
		t.Fatal("expected invalid table JSON to fail")
	}
}

func TestQuestionAnsweringFamiliesGenerativeStatusOnNilPipelines(t *testing.T) {
	var visual *VisualQuestionAnsweringPipeline
	var document *DocumentQuestionAnsweringPipeline
	var table *TableQuestionAnsweringPipeline
	var imageToText *ImageToTextPipeline
	var imageTextToText *ImageTextToTextPipeline

	if !visual.IsGenerative() || !document.IsGenerative() || !table.IsGenerative() ||
		!imageToText.IsGenerative() || !imageTextToText.IsGenerative() {
		t.Fatal("expected all multimodal generative pipelines to be generative")
	}
}
