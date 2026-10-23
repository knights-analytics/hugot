//go:build cgo && (ORT || ALL) && !TRAINING

package ort_test

import (
	"testing"

	testutil "github.com/knights-analytics/hugot/tests"
)

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

func TestTableQuestionAnsweringAggregationPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.TableQuestionAnsweringAggregation)
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

// Video classification

func TestVideoClassificationPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.VideoClassificationPipeline)
}

func TestVideoClassificationPipelineORTGoMLX(t *testing.T) {
	runORTPipelineGoMLX(t, testutil.VideoClassificationPipeline)
}

func TestVideoClassificationPipelineORTCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.VideoClassificationPipeline)
}

func TestVideoClassificationPipelineValidationORT(t *testing.T) {
	runORTPipelineValidation(t, testutil.VideoClassificationPipelineValidation)
}

// Keypoint matching

func TestKeypointMatchingPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.KeypointMatchingPipeline)
}

func TestKeypointMatchingPipelineORTGoMLX(t *testing.T) {
	t.Skip("Skipping test due to MaxPool: support for attribute 'dilations' is not yet implemented issue with GoMLX pipeline")
	runORTPipelineGoMLX(t, testutil.KeypointMatchingPipeline)
}

func TestKeypointMatchingPipelineORTCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.KeypointMatchingPipeline)
}

func TestKeypointMatchingPipelineValidationORT(t *testing.T) {
	runORTPipelineValidation(t, testutil.KeypointMatchingPipelineValidation)
}

// AnyToAny

func TestAnyToAnyPhiPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.AnyToAnyPhiPipeline)
}

func TestAnyToAnyPhiPipelineORTCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.AnyToAnyPhiPipeline)
}

func TestAnyToAnyPhiPipelineValidationORT(t *testing.T) {
	runORTPipelineValidation(t, testutil.AnyToAnyPhiPipelineValidation)
}

func TestAnyToAnyJanusImagePipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.AnyToAnyJanusImagePipeline)
}

func TestAnyToAnyJanusImagePipelineORTCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.AnyToAnyJanusImagePipeline)
}

func TestAnyToAnyJanusImagePipelineValidationORT(t *testing.T) {
	runORTPipelineValidation(t, testutil.AnyToAnyJanusImagePipelineValidation)
}

func TestAnyToAnyLFMAudioPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.AnyToAnyLFMAudioPipeline)
}

func TestAnyToAnyLFMAudioPipelineORTCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.AnyToAnyLFMAudioPipeline)
}

func TestAnyToAnyLFMAudioPipelineValidationORT(t *testing.T) {
	runORTPipelineValidation(t, testutil.AnyToAnyLFMAudioPipelineValidation)
}
