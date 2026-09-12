//go:build cgo && (XLA || ALL) && !TRAINING

package xla_test

import (
	"testing"

	testutil "github.com/knights-analytics/hugot/tests"
)

// Feature extraction

func TestFeatureExtractionPipelineXLA(t *testing.T) {
	runXLAPipeline(t, testutil.FeatureExtractionPipeline)
}

func TestFeatureExtractionPipelineXLACuda(t *testing.T) {
	runXLAPipelineCuda(t, testutil.FeatureExtractionPipeline)
}

func TestFeatureExtractionPipelineValidationXLA(t *testing.T) {
	runXLAPipelineValidation(t, testutil.FeatureExtractionPipelineValidation)
}

// Text classification

func TestTextClassificationPipelineXLA(t *testing.T) {
	runXLAPipeline(t, testutil.TextClassificationPipeline)
}

func TestTextClassificationPipelineXLACuda(t *testing.T) {
	runXLAPipelineCuda(t, testutil.TextClassificationPipeline)
}

func TestTextClassificationPipelineMultiXLA(t *testing.T) {
	runXLAPipeline(t, testutil.TextClassificationPipelineMulti)
}

func TestTextClassificationPipelineMultiXLACuda(t *testing.T) {
	runXLAPipelineCuda(t, testutil.TextClassificationPipelineMulti)
}

func TestTextClassificationPipelineValidationXLA(t *testing.T) {
	runXLAPipelineValidation(t, testutil.TextClassificationPipelineValidation)
}

// Token classification

func TestTokenClassificationPipelineXLA(t *testing.T) {
	runXLAPipeline(t, testutil.TokenClassificationPipeline)
}

func TestTokenClassificationPipelineXLACuda(t *testing.T) {
	runXLAPipelineCuda(t, testutil.TokenClassificationPipeline)
}

func TestTokenClassificationPipelineValidationXLA(t *testing.T) {
	runXLAPipelineValidation(t, testutil.TokenClassificationPipelineValidation)
}

// Zero shot classification

func TestZeroShotClassificationPipelineXLA(t *testing.T) {
	runXLAPipeline(t, testutil.ZeroShotClassificationPipeline)
}

func TestZeroShotClassificationPipelineXLACuda(t *testing.T) {
	runXLAPipelineCuda(t, testutil.ZeroShotClassificationPipeline)
}

func TestZeroShotClassificationPipelineValidationXLA(t *testing.T) {
	runXLAPipelineValidation(t, testutil.ZeroShotClassificationPipelineValidation)
}

// Cross encoder

func TestCrossEncoderPipelineXLA(t *testing.T) {
	runXLAPipeline(t, testutil.CrossEncoderPipeline)
}

func TestCrossEncoderPipelineXLACuda(t *testing.T) {
	runXLAPipelineCuda(t, testutil.CrossEncoderPipeline)
}

func TestCrossEncoderPipelineValidationXLA(t *testing.T) {
	runXLAPipelineValidation(t, testutil.CrossEncoderPipelineValidation)
}

// Fill-mask

func TestFillMaskPipelineXLA(t *testing.T) {
	runXLAPipeline(t, testutil.FillMaskPipeline)
}

func TestFillMaskPipelineXLACuda(t *testing.T) {
	runXLAPipelineCuda(t, testutil.FillMaskPipeline)
}

func TestFillMaskPipelineValidationXLA(t *testing.T) {
	runXLAPipelineValidation(t, testutil.FillMaskPipelineValidation)
}

// Text generation
// These currently only run locally due to resource constraints in CI/CD

func TestTextGenerationPipelineXLA(t *testing.T) {
	runXLAPipeline(t, testutil.TextGenerationPipeline)
}

func TestTextGenerationPipelineXLACuda(t *testing.T) {
	runXLAPipelineCuda(t, testutil.TextGenerationPipeline)
}

func TestTextGenerationPipelineValidationXLA(t *testing.T) {
	runXLAPipelineValidation(t, testutil.TextGenerationPipelineValidation)
}

// Question answering

func TestQAPipelineXLA(t *testing.T) {
	runXLAPipeline(t, testutil.QuestionAnsweringPipeline)
}

func TestQAPipelineXLACuda(t *testing.T) {
	runXLAPipelineCuda(t, testutil.QuestionAnsweringPipeline)
}

func TestQAPipelineValidationXLA(t *testing.T) {
	runXLAPipelineValidation(t, testutil.QuestionAnsweringPipelineValidation)
}
