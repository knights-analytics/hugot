//go:build (GO || ALL) && !TRAINING

package go_test

import (
	"testing"

	testutil "github.com/knights-analytics/hugot/tests"
)

// Visual question answering

func TestVisualQuestionAnsweringPipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.VisualQuestionAnsweringPipeline)
}

func TestVisualQuestionAnsweringPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.VisualQuestionAnsweringPipelineValidation)
}

// Document question answering

func TestDocumentQuestionAnsweringPipelineGo(t *testing.T) {
	t.Skip("Skipping test due to unimplemented ONNX op \"ConvInteger\" issue with Go pipeline")
	runGoPipeline(t, testutil.DocumentQuestionAnsweringPipeline)
}

func TestDocumentQuestionAnsweringPipelineValidationGo(t *testing.T) {
	t.Skip("Skipping test due to unimplemented ONNX op \"ConvInteger\" issue with Go pipeline")
	runGoPipelineValidation(t, testutil.DocumentQuestionAnsweringPipelineValidation)
}

// Table question answering

func TestTableQuestionAnsweringPipelineGo(t *testing.T) {
	t.Skip("Skipping test due to unimplemented ONNX op \"ScatterElements\" issue with Go pipeline")
	runGoPipeline(t, testutil.TableQuestionAnsweringPipeline)
}

func TestTableQuestionAnsweringPipelineValidationGo(t *testing.T) {
	t.Skip("Skipping test due to unimplemented ONNX op \"ScatterElements\" issue with Go pipeline")
	runGoPipelineValidation(t, testutil.TableQuestionAnsweringPipelineValidation)
}

func TestTableQuestionAnsweringAggregationPipelineGo(t *testing.T) {
	t.Skip("Skipping test due to unimplemented ONNX op \"ScatterElements\" issue with Go pipeline")
	runGoPipeline(t, testutil.TableQuestionAnsweringAggregation)
}

// Image-to-text

func TestImageToTextPipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.ImageToTextPipeline)
}

func TestImageToTextPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.ImageToTextPipelineValidation)
}

// Image-text-to-text

func TestImageTextToTextPipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.ImageTextToTextPipeline)
}

func TestImageTextToTextPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.ImageTextToTextPipelineValidation)
}
