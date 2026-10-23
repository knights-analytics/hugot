//go:build (GO || ALL) && !TRAINING

package go_test

import (
	"testing"

	testutil "github.com/knights-analytics/hugot/tests"
)

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

// Image-text-to-text

func TestImageTextToTextPipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.ImageTextToTextPipeline)
}

func TestImageTextToTextPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.ImageTextToTextPipelineValidation)
}

// Keypoint matching

func TestKeypointMatchingPipelineGo(t *testing.T) {
	t.Skip("Skipping test due to MaxPool: support for attribute 'dilations' is not yet implemented issue with GoMLX pipeline")
	runGoPipeline(t, testutil.KeypointMatchingPipeline)
}

func TestKeypointMatchingPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.KeypointMatchingPipelineValidation)
}

// AnyToAny

func TestAnyToAnyPhiPipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.AnyToAnyPhiPipeline)
}

func TestAnyToAnyPhiPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.AnyToAnyPhiPipelineValidation)
}

func TestAnyToAnyJanusImagePipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.AnyToAnyJanusImagePipeline)
}

func TestAnyToAnyJanusImagePipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.AnyToAnyJanusImagePipelineValidation)
}

func TestAnyToAnyLFMAudioPipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.AnyToAnyLFMAudioPipeline)
}

func TestAnyToAnyLFMAudioPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.AnyToAnyLFMAudioPipelineValidation)
}
