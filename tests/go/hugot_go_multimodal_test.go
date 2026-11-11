//go:build (GO || ALL) && !TRAINING

package go_test

import (
	"testing"

	testutil "github.com/knights-analytics/hugot/tests"
)

// Document question answering

func TestDocumentQuestionAnsweringPipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.DocumentQuestionAnsweringPipeline)
}

func TestDocumentQuestionAnsweringPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.DocumentQuestionAnsweringPipelineValidation)
}

// Table question answering

func TestTableQuestionAnsweringPipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.TableQuestionAnsweringPipeline)
}

func TestTableQuestionAnsweringPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.TableQuestionAnsweringPipelineValidation)
}

func TestTableQuestionAnsweringAggregationPipelineGo(t *testing.T) {
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
