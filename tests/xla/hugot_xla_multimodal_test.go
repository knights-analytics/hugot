//go:build cgo && (XLA || ALL) && !TRAINING

package xla_test

import (
	"testing"

	testutil "github.com/knights-analytics/hugot/tests"
)

// Document question answering

func TestDocumentQuestionAnsweringPipelineXLA(t *testing.T) {
	t.Skip("Skipping test due to unimplemented ONNX op \"ConvInteger\" issue with XLA pipeline")
	runXLAPipeline(t, testutil.DocumentQuestionAnsweringPipeline)
}

func TestDocumentQuestionAnsweringPipelineXLACuda(t *testing.T) {
	t.Skip("Skipping test due to unimplemented ONNX op \"ConvInteger\" issue with XLA pipeline")
	runXLAPipelineCuda(t, testutil.DocumentQuestionAnsweringPipeline)
}

func TestDocumentQuestionAnsweringPipelineValidationXLA(t *testing.T) {
	t.Skip("Skipping test due to unimplemented ONNX op \"ConvInteger\" issue with XLA pipeline")
	runXLAPipelineValidation(t, testutil.DocumentQuestionAnsweringPipelineValidation)
}

// Table question answering

func TestTableQuestionAnsweringPipelineXLA(t *testing.T) {
	t.Skip("Skipping test due to unimplemented ONNX op \"ScatterElements\" issue with XLA pipeline")
	runXLAPipeline(t, testutil.TableQuestionAnsweringPipeline)
}

func TestTableQuestionAnsweringPipelineXLACuda(t *testing.T) {
	t.Skip("Skipping test due to unimplemented ONNX op \"ScatterElements\" issue with XLA pipeline")
	runXLAPipelineCuda(t, testutil.TableQuestionAnsweringPipeline)
}

func TestTableQuestionAnsweringPipelineValidationXLA(t *testing.T) {
	t.Skip("Skipping test due to unimplemented ONNX op \"ScatterElements\" issue with XLA pipeline")
	runXLAPipelineValidation(t, testutil.TableQuestionAnsweringPipelineValidation)
}

func TestTableQuestionAnsweringAggregationPipelineXLA(t *testing.T) {
	t.Skip("Skipping test due to unimplemented ONNX op \"ScatterElements\" issue with XLA pipeline")
	runXLAPipeline(t, testutil.TableQuestionAnsweringAggregation)
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

// Video classification

func TestVideoClassificationPipelineXLA(t *testing.T) {
	runXLAPipeline(t, testutil.VideoClassificationPipeline)
}

func TestVideoClassificationPipelineXLACuda(t *testing.T) {
	runXLAPipelineCuda(t, testutil.VideoClassificationPipeline)
}

func TestVideoClassificationPipelineValidationXLA(t *testing.T) {
	runXLAPipelineValidation(t, testutil.VideoClassificationPipelineValidation)
}

// Keypoint matching

func TestKeypointMatchingPipelineXLA(t *testing.T) {
	t.Skip("Skipping test due to MaxPool: support for attribute 'dilations' is not yet implemented issue with GoMLX pipeline")
	runXLAPipeline(t, testutil.KeypointMatchingPipeline)
}

func TestKeypointMatchingPipelineXLACuda(t *testing.T) {
	t.Skip("Skipping test due to MaxPool: support for attribute 'dilations' is not yet implemented issue with GoMLX pipeline")
	runXLAPipelineCuda(t, testutil.KeypointMatchingPipeline)
}

func TestKeypointMatchingPipelineValidationXLA(t *testing.T) {
	runXLAPipelineValidation(t, testutil.KeypointMatchingPipelineValidation)
}

// AnyToAny

func TestAnyToAnyPhiPipelineXLA(t *testing.T) {
	runXLAPipeline(t, testutil.AnyToAnyPhiPipeline)
}

func TestAnyToAnyPhiPipelineXLACuda(t *testing.T) {
	runXLAPipelineCuda(t, testutil.AnyToAnyPhiPipeline)
}

func TestAnyToAnyPhiPipelineValidationXLA(t *testing.T) {
	runXLAPipelineValidation(t, testutil.AnyToAnyPhiPipelineValidation)
}

func TestAnyToAnyJanusImagePipelineXLA(t *testing.T) {
	runXLAPipeline(t, testutil.AnyToAnyJanusImagePipeline)
}

func TestAnyToAnyJanusImagePipelineXLACuda(t *testing.T) {
	runXLAPipelineCuda(t, testutil.AnyToAnyJanusImagePipeline)
}

func TestAnyToAnyJanusImagePipelineValidationXLA(t *testing.T) {
	runXLAPipelineValidation(t, testutil.AnyToAnyJanusImagePipelineValidation)
}

func TestAnyToAnyLFMAudioPipelineXLA(t *testing.T) {
	runXLAPipeline(t, testutil.AnyToAnyLFMAudioPipeline)
}

func TestAnyToAnyLFMAudioPipelineXLACuda(t *testing.T) {
	runXLAPipelineCuda(t, testutil.AnyToAnyLFMAudioPipeline)
}

func TestAnyToAnyLFMAudioPipelineValidationXLA(t *testing.T) {
	runXLAPipelineValidation(t, testutil.AnyToAnyLFMAudioPipelineValidation)
}
