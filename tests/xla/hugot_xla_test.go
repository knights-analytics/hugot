//go:build cgo && (XLA || ALL) && !TRAINING

package xla_test

import (
	"os"
	"testing"

	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/options"
	testutil "github.com/knights-analytics/hugot/tests"
)

// FEATURE EXTRACTION

func TestFeatureExtractionPipelineXLA(t *testing.T) {
	session, err := hugot.NewXLASession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.FeatureExtractionPipeline(t, session)
}

func TestFeatureExtractionPipelineXLACuda(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	session, err := hugot.NewXLASession(t.Context(), options.WithCuda(map[string]string{
		"device_id": "0",
	}))
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.FeatureExtractionPipeline(t, session)
}

func TestFeatureExtractionPipelineValidationXLA(t *testing.T) {
	session, err := hugot.NewXLASession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.FeatureExtractionPipelineValidation(t, session)
}

// Text classification

func TestTextClassificationPipelineXLA(t *testing.T) {
	session, err := hugot.NewXLASession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.TextClassificationPipeline(t, session)
}

func TestTextClassificationPipelineXLACuda(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	session, err := hugot.NewXLASession(t.Context(), options.WithCuda(map[string]string{
		"device_id": "0",
	}))
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.TextClassificationPipeline(t, session)
}

func TestTextClassificationPipelineMultiXLA(t *testing.T) {
	session, err := hugot.NewXLASession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.TextClassificationPipelineMulti(t, session)
}

func TestTextClassificationPipelineMultiXLACuda(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	session, err := hugot.NewXLASession(t.Context(), options.WithCuda(map[string]string{
		"device_id": "0",
	}))
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.TextClassificationPipelineMulti(t, session)
}

func TestTextClassificationPipelineValidationXLA(t *testing.T) {
	session, err := hugot.NewXLASession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.TextClassificationPipelineValidation(t, session)
}

// Token classification

func TestTokenClassificationPipelineXLA(t *testing.T) {
	session, err := hugot.NewXLASession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.TokenClassificationPipeline(t, session)
}

func TestTokenClassificationPipelineXLACuda(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	session, err := hugot.NewXLASession(t.Context(), options.WithCuda(map[string]string{
		"device_id": "0",
	}))
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.TokenClassificationPipeline(t, session)
}

func TestTokenClassificationPipelineValidationXLA(t *testing.T) {
	session, err := hugot.NewXLASession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.TokenClassificationPipelineValidation(t, session)
}

// Zero shot

func TestZeroShotClassificationPipelineXLA(t *testing.T) {
	session, err := hugot.NewXLASession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.ZeroShotClassificationPipeline(t, session)
}

func TestZeroShotClassificationPipelineXLACuda(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	session, err := hugot.NewXLASession(t.Context(), options.WithCuda(map[string]string{
		"device_id": "0",
	}))
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.ZeroShotClassificationPipeline(t, session)
}

func TestZeroShotClassificationPipelineValidationXLA(t *testing.T) {
	session, err := hugot.NewXLASession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.ZeroShotClassificationPipelineValidation(t, session)
}

// Cross Encoder

func TestCrossEncoderPipelineXLA(t *testing.T) {
	session, err := hugot.NewXLASession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.CrossEncoderPipeline(t, session)
}

func TestCrossEncoderPipelineXLACuda(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	session, err := hugot.NewXLASession(t.Context(), options.WithCuda(map[string]string{
		"device_id": "0",
	}))
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.CrossEncoderPipeline(t, session)
}

func TestCrossEncoderPipelineValidationXLA(t *testing.T) {
	session, err := hugot.NewXLASession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.CrossEncoderPipelineValidation(t, session)
}

// Image classification

func TestImageClassificationPipelineXLA(t *testing.T) {
	session, err := hugot.NewXLASession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.ImageClassificationPipeline(t, session)
}

func TestImageClassificationPipelineXLACuda(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	session, err := hugot.NewXLASession(t.Context(), options.WithCuda(map[string]string{
		"device_id": "0",
	}))
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.ImageClassificationPipeline(t, session)
}

func TestImageClassificationPipelineValidationXLA(t *testing.T) {
	session, err := hugot.NewXLASession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.ImageClassificationPipelineValidation(t, session)
}

// Object detection

func TestObjectDetectionPipelineXLA(t *testing.T) {
	session, err := hugot.NewXLASession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.ObjectDetectionPipeline(t, session)
}

func TestObjectDetectionPipelineXLACuda(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	session, err := hugot.NewXLASession(t.Context(), options.WithCuda(map[string]string{
		"device_id": "0",
	}))
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.ObjectDetectionPipeline(t, session)
}

func TestObjectDetectionPipelineValidationXLA(t *testing.T) {
	session, err := hugot.NewXLASession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.ObjectDetectionPipelineValidation(t, session)
}

// text generation

func TestTextGenerationPipelineXLA(t *testing.T) {
	t.Skip("Generative models are not supported yet for XLA")
	session, err := hugot.NewXLASession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.TextGenerationPipeline(t, session)
}

func TestTextGenerationPipelineXLACuda(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	t.Skip("Generative models are not supported yet for XLA")
	session, err := hugot.NewXLASession(t.Context(), options.WithCuda(map[string]string{
		"device_id": "0",
	}))
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.TextGenerationPipeline(t, session)
}

func TestTextGenerationPipelineValidationXLA(t *testing.T) {
	t.Skip("Generative models are not supported yet for XLA")
	session, err := hugot.NewXLASession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.TextGenerationPipelineValidation(t, session)
}

// Tabular

func TestTabularPipelineXLA(t *testing.T) {
	t.Skip("Currently missing TreeEnsembleClassifier ONNX operator")
	session, err := hugot.NewXLASession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.TabularPipeline(t, session)
}

func TestTabularPipelineXLACuda(t *testing.T) {
	t.Skip("Currently missing TreeEnsembleClassifier ONNX operator")
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	session, err := hugot.NewXLASession(t.Context(), options.WithCuda(map[string]string{
		"device_id": "0",
	}))
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.TabularPipeline(t, session)
}

// Question answering

func TestQuestionAnsweringPipelineXLA(t *testing.T) {
	session, err := hugot.NewXLASession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.QuestionAnsweringPipeline(t, session)
}

func TestQuestionAnsweringPipelineXLACuda(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	session, err := hugot.NewXLASession(t.Context(), options.WithCuda(map[string]string{
		"device_id": "0",
	}))
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.QuestionAnsweringPipeline(t, session)
}

// No same name

func TestNoSameNamePipelineXLA(t *testing.T) {
	session, err := hugot.NewXLASession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.NoSameNamePipeline(t, session)
}

func TestDestroyPipelineXLA(t *testing.T) {
	session, err := hugot.NewXLASession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.DestroyPipelines(t, session)
}

// Thread safety

func TestThreadSafetyXLA(t *testing.T) {
	session, err := hugot.NewXLASession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.ThreadSafety(t, session, 250)
}

func TestThreadSafetyXLACuda(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	session, err := hugot.NewXLASession(t.Context(), options.WithCuda(map[string]string{
		"device_id": "0",
	}))
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.ThreadSafety(t, session, 1000)
}
