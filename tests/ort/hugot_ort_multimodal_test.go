//go:build cgo && (ORT || ALL) && !TRAINING

package ort_test

import (
	"testing"

	testutil "github.com/knights-analytics/hugot/tests"
)

// Visual question answering

func TestNativeVisualQuestionAnsweringPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.NativeVisualQuestionAnsweringPipeline)
}

func TestVisualQuestionAnsweringPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.VisualQuestionAnsweringPipeline)
}

func TestVisualQuestionAnsweringPipelineORTGoMLX(t *testing.T) {
	runORTPipelineGoMLX(t, testutil.VisualQuestionAnsweringPipeline)
}

func TestVisualQuestionAnsweringPipelineORTCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.VisualQuestionAnsweringPipeline)
}

func TestVisualQuestionAnsweringPipelineValidationORT(t *testing.T) {
	runORTPipelineValidation(t, testutil.VisualQuestionAnsweringPipelineValidation)
}

// Document question answering

func TestDocumentQuestionAnsweringPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.DocumentQuestionAnsweringPipeline)
}

func TestDocumentQuestionAnsweringPipelineORTGoMLX(t *testing.T) {
	t.Skip("Skipping test due to unimplemented ONNX op \"ConvInteger\" issue with GoMLX pipeline")
	runORTPipelineGoMLX(t, testutil.DocumentQuestionAnsweringPipeline)
}

func TestDocumentQuestionAnsweringPipelineORTCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.DocumentQuestionAnsweringPipeline)
}

func TestDocumentQuestionAnsweringPipelineValidationORT(t *testing.T) {
	runORTPipelineValidation(t, testutil.DocumentQuestionAnsweringPipelineValidation)
}

// Table question answering

func TestTableQuestionAnsweringPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.TableQuestionAnsweringPipeline)
}

func TestTableQuestionAnsweringPipelineORTGoMLX(t *testing.T) {
	t.Skip("Skipping test due to unimplemented ONNX op \"ScatterElements\" issue with GoMLX pipeline")
	runORTPipelineGoMLX(t, testutil.TableQuestionAnsweringPipeline)
}

func TestTableQuestionAnsweringPipelineORTCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.TableQuestionAnsweringPipeline)
}

func TestTableQuestionAnsweringPipelineValidationORT(t *testing.T) {
	runORTPipelineValidation(t, testutil.TableQuestionAnsweringPipelineValidation)
}

// Image-to-text

func TestNativeImageToTextPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.NativeImageToTextPipeline)
}

func TestImageToTextPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.ImageToTextPipeline)
}

func TestImageToTextPipelineORTGoMLX(t *testing.T) {
	runORTPipelineGoMLX(t, testutil.ImageToTextPipeline)
}

func TestImageToTextPipelineORTCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.ImageToTextPipeline)
}

func TestImageToTextPipelineValidationORT(t *testing.T) {
	runORTPipelineValidation(t, testutil.ImageToTextPipelineValidation)
}

// Image-text-to-text

func TestImageTextToTextPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.ImageTextToTextPipeline)
}

func TestImageTextToTextPipelineORTGoMLX(t *testing.T) {
	runORTPipelineGoMLX(t, testutil.ImageTextToTextPipeline)
}

func TestImageTextToTextPipelineORTCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.ImageTextToTextPipeline)
}

func TestImageTextToTextPipelineValidationORT(t *testing.T) {
	runORTPipelineValidation(t, testutil.ImageTextToTextPipelineValidation)
}
