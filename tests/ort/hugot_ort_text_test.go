//go:build cgo && (ORT || ALL) && !TRAINING

package ort_test

import (
	"testing"

	"github.com/knights-analytics/hugot/options"
	testutil "github.com/knights-analytics/hugot/tests"
)

// Feature extraction

func TestRawFeatureExtractionPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.RawFeatureExtractionPipeline)
}

func TestRawFeatureExtractionPipelineORTGoMLX(t *testing.T) {
	runORTPipelineGoMLX(t, testutil.RawFeatureExtractionPipeline)
}

func TestFeatureExtractionPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.FeatureExtractionPipeline)
}

func TestFeatureExtractionPipelineORTGoMLX(t *testing.T) {
	runORTPipelineGoMLX(t, testutil.FeatureExtractionPipeline)
}

func TestFeatureExtractionPipelineORTCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.FeatureExtractionPipeline)
}

func TestFeatureExtractionPipelineValidationORT(t *testing.T) {
	runORTPipelineValidation(t, testutil.FeatureExtractionPipelineValidation)
}

// Text classification

func TestTextClassificationPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.TextClassificationPipeline, textClassificationORTOptions...)
}

func TestTextClassificationPipelineORTGoMLX(t *testing.T) {
	runORTPipelineGoMLX(t, testutil.TextClassificationPipeline, textClassificationORTOptions...)
}

func TestTextClassificationPipelineORTCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.TextClassificationPipeline)
}

func TestTextClassificationPipelineMultiORT(t *testing.T) {
	runORTPipeline(t, testutil.TextClassificationPipelineMulti, textClassificationORTOptions...)
}

func TestTextClassificationPipelineMultiGoMLX(t *testing.T) {
	runORTPipelineGoMLX(t, testutil.TextClassificationPipelineMulti, textClassificationORTOptions...)
}

func TestTextClassificationPipelineORTMultiCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.TextClassificationPipelineMulti)
}

func TestTextClassificationPipelineValidationORT(t *testing.T) {
	runORTPipelineValidation(t, testutil.TextClassificationPipelineValidation)
}

// Token classification

func TestTokenClassificationPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.TokenClassificationPipeline)
}

func TestTokenClassificationPipelineORTGoMLX(t *testing.T) {
	runORTPipelineGoMLX(t, testutil.TokenClassificationPipeline)
}

func TestTokenClassificationPipelineORTCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.TokenClassificationPipeline)
}

func TestTokenClassificationPipelineValidationORT(t *testing.T) {
	runORTPipelineValidation(t, testutil.TokenClassificationPipelineValidation)
}

// Zero shot classification

func TestZeroShotClassificationPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.ZeroShotClassificationPipeline)
}

func TestZeroShotClassificationPipelineORTGoMLX(t *testing.T) {
	runORTPipelineGoMLX(t, testutil.ZeroShotClassificationPipeline)
}

func TestZeroShotClassificationPipelineORTCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.ZeroShotClassificationPipeline)
}

func TestZeroShotClassificationPipelineValidationORT(t *testing.T) {
	runORTPipelineValidation(t, testutil.ZeroShotClassificationPipelineValidation)
}

// Cross encoder

func TestCrossEncoderPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.CrossEncoderPipeline)
}

func TestCrossEncoderPipelineORTGoMLX(t *testing.T) {
	runORTPipelineGoMLX(t, testutil.CrossEncoderPipeline)
}

func TestCrossEncoderPipelineORTCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.CrossEncoderPipeline)
}

func TestCrossEncoderPipelineValidationORT(t *testing.T) {
	runORTPipelineValidation(t, testutil.CrossEncoderPipelineValidation)
}

// Fill-mask

func TestFillMaskPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.FillMaskPipeline)
}

func TestFillMaskPipelineORTGoMLX(t *testing.T) {
	runORTPipelineGoMLX(t, testutil.FillMaskPipeline)
}

func TestFillMaskPipelineORTCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.FillMaskPipeline)
}

func TestFillMaskPipelineValidationORT(t *testing.T) {
	runORTPipelineValidation(t, testutil.FillMaskPipelineValidation)
}

// Text generation
// These currently only run locally due to resource constraints in CI/CD

func TestTextGenerationPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.TextGenerationPipeline)
}

func TestTextGenerationPipelineORTEngine(t *testing.T) {
	runORTPipeline(t, testutil.TextGenerationPipelineEngine, options.WithGenerativeEngine())
}

func TestTextGenerationPipelineORTGoMLX(t *testing.T) {
	runORTPipelineGoMLX(t, testutil.TextGenerationPipeline)
}

func TestTextGenerationPipelineORTCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.TextGenerationPipeline)
}

func TestTextGenerationPipelineValidationORT(t *testing.T) {
	runORTPipelineValidation(t, testutil.TextGenerationPipelineValidation)
}

// Question answering

func TestQAPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.QuestionAnsweringPipeline)
}

func TestQAPipelineORTGoMLX(t *testing.T) {
	runORTPipelineGoMLX(t, testutil.QuestionAnsweringPipeline)
}

func TestQAPipelineORTCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.QuestionAnsweringPipeline)
}

func TestQAPipelineValidationORT(t *testing.T) {
	runORTPipelineValidation(t, testutil.QuestionAnsweringPipelineValidation)
}
