//go:build (GO || ALL) && !TRAINING

package go_test

import (
	"testing"

	testutil "github.com/knights-analytics/hugot/tests"
)

// Text classification

func TestTextClassificationPipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.TextClassificationPipeline)
}

func TestTextClassificationPipelineMultiGo(t *testing.T) {
	runGoPipeline(t, testutil.TextClassificationPipelineMulti)
}

func TestTextClassificationPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.TextClassificationPipelineValidation)
}

// Token classification

func TestTokenClassificationPipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.TokenClassificationPipeline)
}

func TestTokenClassificationPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.TokenClassificationPipelineValidation)
}

// Zero shot classification

func TestZeroShotClassificationPipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.ZeroShotClassificationPipeline)
}

func TestZeroShotClassificationPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.ZeroShotClassificationPipelineValidation)
}

// Cross encoder

func TestCrossEncoderPipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.CrossEncoderPipeline)
}

func TestCrossEncoderPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.CrossEncoderPipelineValidation)
}

// Feature extraction

func TestRawFeatureExtractionPipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.RawFeatureExtractionPipeline)
}

func TestFeatureExtractionPipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.FeatureExtractionPipeline)
}

func TestFeatureExtractionPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.FeatureExtractionPipelineValidation)
}

// Fill-mask

func TestFillMaskPipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.FillMaskPipeline)
}

func TestFillMaskPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.FillMaskPipelineValidation)
}

// Text generation
// These currently only run locally due to resource constraints in CI/CD

func TestTextGenerationPipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.TextGenerationPipeline)
}

func TestTextGenerationPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.TextGenerationPipelineValidation)
}

// Question answering

func TestQAPipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.QuestionAnsweringPipeline)
}

func TestQAPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.QuestionAnsweringPipelineValidation)
}
