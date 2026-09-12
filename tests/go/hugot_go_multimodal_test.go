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
