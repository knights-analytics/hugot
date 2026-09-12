//go:build cgo && (XLA || ALL) && !TRAINING

package xla_test

import (
	"testing"

	testutil "github.com/knights-analytics/hugot/tests"
)

// Visual question answering

func TestVisualQuestionAnsweringPipelineXLA(t *testing.T) {
	runXLAPipeline(t, testutil.VisualQuestionAnsweringPipeline)
}

func TestVisualQuestionAnsweringPipelineXLACuda(t *testing.T) {
	runXLAPipelineCuda(t, testutil.VisualQuestionAnsweringPipeline)
}

func TestVisualQuestionAnsweringPipelineValidationXLA(t *testing.T) {
	runXLAPipelineValidation(t, testutil.VisualQuestionAnsweringPipelineValidation)
}

// Document question answering

func TestDocumentQuestionAnsweringPipelineXLA(t *testing.T) {
	runXLAPipeline(t, testutil.DocumentQuestionAnsweringPipeline)
}

func TestDocumentQuestionAnsweringPipelineXLACuda(t *testing.T) {
	runXLAPipelineCuda(t, testutil.DocumentQuestionAnsweringPipeline)
}

func TestDocumentQuestionAnsweringPipelineValidationXLA(t *testing.T) {
	runXLAPipelineValidation(t, testutil.DocumentQuestionAnsweringPipelineValidation)
}

// Table question answering

func TestTableQuestionAnsweringPipelineXLA(t *testing.T) {
	runXLAPipeline(t, testutil.TableQuestionAnsweringPipeline)
}

func TestTableQuestionAnsweringPipelineXLACuda(t *testing.T) {
	runXLAPipelineCuda(t, testutil.TableQuestionAnsweringPipeline)
}

func TestTableQuestionAnsweringPipelineValidationXLA(t *testing.T) {
	runXLAPipelineValidation(t, testutil.TableQuestionAnsweringPipelineValidation)
}

// Image-to-text

func TestImageToTextPipelineXLA(t *testing.T) {
	runXLAPipeline(t, testutil.ImageToTextPipeline)
}

func TestImageToTextPipelineXLACuda(t *testing.T) {
	runXLAPipelineCuda(t, testutil.ImageToTextPipeline)
}

func TestImageToTextPipelineValidationXLA(t *testing.T) {
	runXLAPipelineValidation(t, testutil.ImageToTextPipelineValidation)
}

// Image-text-to-text

func TestImageTextToTextPipelineXLA(t *testing.T) {
	runXLAPipeline(t, testutil.ImageTextToTextPipeline)
}

func TestImageTextToTextPipelineXLACuda(t *testing.T) {
	runXLAPipelineCuda(t, testutil.ImageTextToTextPipeline)
}

func TestImageTextToTextPipelineValidationXLA(t *testing.T) {
	runXLAPipelineValidation(t, testutil.ImageTextToTextPipelineValidation)
}
