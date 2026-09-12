//go:build cgo && (XLA || ALL) && !TRAINING

package xla_test

import (
	"os"
	"testing"

	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/options"
	testutil "github.com/knights-analytics/hugot/tests"
)

func runXLAPipeline(t *testing.T, run func(*testing.T, *hugot.Session), opts ...options.WithOption) {
	t.Helper()
	session, err := hugot.NewXLASession(t.Context(), opts...)
	testutil.CheckT(t, err)
	defer func() { testutil.CheckT(t, session.Destroy()) }()
	run(t, session)
}

func runXLAPipelineValidation(t *testing.T, run func(*testing.T, *hugot.Session), opts ...options.WithOption) {
	t.Helper()
	runXLAPipeline(t, run, opts...)
}

func runXLAPipelineCuda(t *testing.T, run func(*testing.T, *hugot.Session), opts ...options.WithOption) {
	t.Helper()
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	opts = append([]options.WithOption{options.WithCuda(map[string]string{"device_id": "0"})}, opts...)
	runXLAPipeline(t, run, opts...)
}

// FEATURE EXTRACTION

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

// Zero shot

func TestZeroShotClassificationPipelineXLA(t *testing.T) {
	runXLAPipeline(t, testutil.ZeroShotClassificationPipeline)
}

func TestZeroShotClassificationPipelineXLACuda(t *testing.T) {
	runXLAPipelineCuda(t, testutil.ZeroShotClassificationPipeline)
}

func TestZeroShotClassificationPipelineValidationXLA(t *testing.T) {
	runXLAPipelineValidation(t, testutil.ZeroShotClassificationPipelineValidation)
}

// Cross Encoder

func TestCrossEncoderPipelineXLA(t *testing.T) {
	runXLAPipeline(t, testutil.CrossEncoderPipeline)
}

func TestCrossEncoderPipelineXLACuda(t *testing.T) {
	runXLAPipelineCuda(t, testutil.CrossEncoderPipeline)
}

func TestCrossEncoderPipelineValidationXLA(t *testing.T) {
	runXLAPipelineValidation(t, testutil.CrossEncoderPipelineValidation)
}

// Image classification

func TestImageClassificationPipelineXLA(t *testing.T) {
	runXLAPipeline(t, testutil.ImageClassificationPipeline)
}

func TestImageClassificationPipelineXLACuda(t *testing.T) {
	runXLAPipelineCuda(t, testutil.ImageClassificationPipeline)
}

func TestImageClassificationPipelineValidationXLA(t *testing.T) {
	runXLAPipelineValidation(t, testutil.ImageClassificationPipelineValidation)
}

// Object detection

func TestObjectDetectionPipelineXLA(t *testing.T) {
	runXLAPipeline(t, testutil.ObjectDetectionPipeline)
}

func TestObjectDetectionPipelineXLACuda(t *testing.T) {
	runXLAPipelineCuda(t, testutil.ObjectDetectionPipeline)
}

func TestObjectDetectionPipelineValidationXLA(t *testing.T) {
	runXLAPipelineValidation(t, testutil.ObjectDetectionPipelineValidation)
}

// fill-mask

func TestFillMaskPipelineXLA(t *testing.T) {
	runXLAPipeline(t, testutil.FillMaskPipeline)
}

func TestFillMaskPipelineXLACuda(t *testing.T) {
	runXLAPipelineCuda(t, testutil.FillMaskPipeline)
}

func TestFillMaskPipelineValidationXLA(t *testing.T) {
	runXLAPipelineValidation(t, testutil.FillMaskPipelineValidation)
}

// image feature extraction

func TestImageFeatureExtractionPipelineXLA(t *testing.T) {
	runXLAPipeline(t, testutil.ImageFeatureExtractionPipeline)
}

func TestImageFeatureExtractionPipelineXLACuda(t *testing.T) {
	runXLAPipelineCuda(t, testutil.ImageFeatureExtractionPipeline)
}

func TestImageFeatureExtractionPipelineValidationXLA(t *testing.T) {
	runXLAPipelineValidation(t, testutil.ImageFeatureExtractionPipelineValidation)
}

// image segmentation

func TestImageSegmentationPipelineXLA(t *testing.T) {
	runXLAPipeline(t, testutil.ImageSegmentationPipeline)
}

func TestImageSegmentationPipelineXLACuda(t *testing.T) {
	runXLAPipelineCuda(t, testutil.ImageSegmentationPipeline)
}

func TestImageSegmentationPipelineValidationXLA(t *testing.T) {
	runXLAPipelineValidation(t, testutil.ImageSegmentationPipelineValidation)
}

// depth estimation

func TestDepthEstimationPipelineXLA(t *testing.T) {
	runXLAPipeline(t, testutil.DepthEstimationPipeline)
}

func TestDepthEstimationPipelineXLACuda(t *testing.T) {
	runXLAPipelineCuda(t, testutil.DepthEstimationPipeline)
}

func TestDepthEstimationPipelineValidationXLA(t *testing.T) {
	runXLAPipelineValidation(t, testutil.DepthEstimationPipelineValidation)
}

// summarization

func TestSummarizationPipelineXLA(t *testing.T) {
	runXLAPipeline(t, testutil.SummarizationPipeline)
}

func TestSummarizationPipelineXLACuda(t *testing.T) {
	runXLAPipelineCuda(t, testutil.SummarizationPipeline)
}

func TestSummarizationPipelineValidationXLA(t *testing.T) {
	runXLAPipelineValidation(t, testutil.SummarizationPipelineValidation)
}

// translation

func TestTranslationPipelineXLA(t *testing.T) {
	runXLAPipeline(t, testutil.TranslationPipeline)
}

func TestTranslationPipelineXLACuda(t *testing.T) {
	runXLAPipelineCuda(t, testutil.TranslationPipeline)
}

func TestTranslationPipelineValidationXLA(t *testing.T) {
	runXLAPipelineValidation(t, testutil.TranslationPipelineValidation)
}

// text-to-text generation

func TestText2TextGenerationPipelineXLA(t *testing.T) {
	runXLAPipeline(t, testutil.Text2TextGenerationPipeline)
}

func TestText2TextGenerationPipelineXLACuda(t *testing.T) {
	runXLAPipelineCuda(t, testutil.Text2TextGenerationPipeline)
}

func TestText2TextGenerationPipelineValidationXLA(t *testing.T) {
	runXLAPipelineValidation(t, testutil.Text2TextGenerationPipelineValidation)
}

// zero-shot object detection

func TestZeroShotObjectDetectionPipelineXLA(t *testing.T) {
	runXLAPipeline(t, testutil.ZeroShotObjectDetectionPipeline)
}

func TestZeroShotObjectDetectionPipelineXLACuda(t *testing.T) {
	runXLAPipelineCuda(t, testutil.ZeroShotObjectDetectionPipeline)
}

func TestZeroShotObjectDetectionPipelineValidationXLA(t *testing.T) {
	runXLAPipelineValidation(t, testutil.ZeroShotObjectDetectionPipelineValidation)
}

// mask generation

func TestMaskGenerationPipelineXLA(t *testing.T) {
	runXLAPipeline(t, testutil.MaskGenerationPipeline)
}

func TestMaskGenerationPipelineXLACuda(t *testing.T) {
	runXLAPipelineCuda(t, testutil.MaskGenerationPipeline)
}

func TestMaskGenerationPipelineValidationXLA(t *testing.T) {
	runXLAPipelineValidation(t, testutil.MaskGenerationPipelineValidation)
}

// zero-shot audio classification

func TestZeroShotAudioClassificationPipelineXLA(t *testing.T) {
	runXLAPipeline(t, testutil.ZeroShotAudioClassificationPipeline)
}

func TestZeroShotAudioClassificationPipelineXLACuda(t *testing.T) {
	runXLAPipelineCuda(t, testutil.ZeroShotAudioClassificationPipeline)
}

func TestZeroShotAudioClassificationPipelineValidationXLA(t *testing.T) {
	runXLAPipelineValidation(t, testutil.ZeroShotAudioClassificationPipelineValidation)
}

// visual question answering

func TestVisualQuestionAnsweringPipelineXLA(t *testing.T) {
	t.Skip("visual question answering requires the ORT GenAI multimodal runtime")
	runXLAPipeline(t, testutil.VisualQuestionAnsweringPipeline)
}

func TestVisualQuestionAnsweringPipelineXLACuda(t *testing.T) {
	t.Skip("visual question answering requires the ORT GenAI multimodal runtime")
	runXLAPipelineCuda(t, testutil.VisualQuestionAnsweringPipeline)
}

func TestVisualQuestionAnsweringPipelineValidationXLA(t *testing.T) {
	runXLAPipelineValidation(t, testutil.VisualQuestionAnsweringPipelineValidation)
}

// document question answering

func TestDocumentQuestionAnsweringPipelineXLA(t *testing.T) {
	runXLAPipeline(t, testutil.DocumentQuestionAnsweringPipeline)
}

func TestDocumentQuestionAnsweringPipelineXLACuda(t *testing.T) {
	runXLAPipelineCuda(t, testutil.DocumentQuestionAnsweringPipeline)
}

func TestDocumentQuestionAnsweringPipelineValidationXLA(t *testing.T) {
	runXLAPipelineValidation(t, testutil.DocumentQuestionAnsweringPipelineValidation)
}

// table question answering

func TestTableQuestionAnsweringPipelineXLA(t *testing.T) {
	runXLAPipeline(t, testutil.TableQuestionAnsweringPipeline)
}

func TestTableQuestionAnsweringPipelineXLACuda(t *testing.T) {
	runXLAPipelineCuda(t, testutil.TableQuestionAnsweringPipeline)
}

func TestTableQuestionAnsweringPipelineValidationXLA(t *testing.T) {
	runXLAPipelineValidation(t, testutil.TableQuestionAnsweringPipelineValidation)
}

// text-to-speech

func TestTextToSpeechPipelineXLA(t *testing.T) {
	runXLAPipeline(t, testutil.TextToSpeechPipeline)
}

func TestTextToSpeechPipelineXLACuda(t *testing.T) {
	runXLAPipelineCuda(t, testutil.TextToSpeechPipeline)
}

func TestTextToSpeechPipelineValidationXLA(t *testing.T) {
	runXLAPipelineValidation(t, testutil.TextToSpeechPipelineValidation)
}

// text-to-audio

func TestTextToAudioPipelineXLA(t *testing.T) {
	runXLAPipeline(t, testutil.TextToAudioPipeline)
}

func TestTextToAudioPipelineXLACuda(t *testing.T) {
	runXLAPipelineCuda(t, testutil.TextToAudioPipeline)
}

func TestTextToAudioPipelineValidationXLA(t *testing.T) {
	runXLAPipelineValidation(t, testutil.TextToAudioPipelineValidation)
}

// audio classification

func TestAudioClassificationPipelineXLA(t *testing.T) {
	runXLAPipeline(t, testutil.AudioClassificationPipeline)
}

func TestAudioClassificationPipelineXLACuda(t *testing.T) {
	runXLAPipelineCuda(t, testutil.AudioClassificationPipeline)
}

func TestAudioClassificationPipelineValidationXLA(t *testing.T) {
	runXLAPipelineValidation(t, testutil.AudioClassificationPipelineValidation)
}

// background removal

func TestBackgroundRemovalPipelineXLA(t *testing.T) {
	runXLAPipeline(t, testutil.BackgroundRemovalPipeline)
}

func TestBackgroundRemovalPipelineXLACuda(t *testing.T) {
	runXLAPipelineCuda(t, testutil.BackgroundRemovalPipeline)
}

func TestBackgroundRemovalPipelineValidationXLA(t *testing.T) {
	runXLAPipelineValidation(t, testutil.BackgroundRemovalPipelineValidation)
}

// zero-shot image classification

func TestZeroShotImageClassificationPipelineXLA(t *testing.T) {
	runXLAPipeline(t, testutil.ZeroShotImageClassificationPipeline)
}

func TestZeroShotImageClassificationPipelineXLACuda(t *testing.T) {
	runXLAPipelineCuda(t, testutil.ZeroShotImageClassificationPipeline)
}

func TestZeroShotImageClassificationPipelineValidationXLA(t *testing.T) {
	runXLAPipelineValidation(t, testutil.ZeroShotImageClassificationPipelineValidation)
}

// automatic speech recognition

func TestAutomaticSpeechRecognitionPipelineXLA(t *testing.T) {
	runXLAPipeline(t, testutil.AutomaticSpeechRecognitionPipeline)
}

func TestAutomaticSpeechRecognitionPipelineXLACuda(t *testing.T) {
	runXLAPipelineCuda(t, testutil.AutomaticSpeechRecognitionPipeline)
}

func TestAutomaticSpeechRecognitionPipelineValidationXLA(t *testing.T) {
	runXLAPipelineValidation(t, testutil.AutomaticSpeechRecognitionPipelineValidation)
}

// image-to-text

func TestImageToTextPipelineXLA(t *testing.T) {
	runXLAPipeline(t, testutil.ImageToTextPipeline)
}

func TestImageToTextPipelineXLACuda(t *testing.T) {
	runXLAPipelineCuda(t, testutil.ImageToTextPipeline)
}

func TestImageToTextPipelineValidationXLA(t *testing.T) {
	runXLAPipelineValidation(t, testutil.ImageToTextPipelineValidation)
}

// image-text-to-text

func TestImageTextToTextPipelineXLA(t *testing.T) {
	runXLAPipeline(t, testutil.ImageTextToTextPipeline)
}

func TestImageTextToTextPipelineXLACuda(t *testing.T) {
	runXLAPipelineCuda(t, testutil.ImageTextToTextPipeline)
}

func TestImageTextToTextPipelineValidationXLA(t *testing.T) {
	runXLAPipelineValidation(t, testutil.ImageTextToTextPipelineValidation)
}

// Text generation
// These currently only run locally due to resource constraints in CI/CD

func TestTextGenerationPipelineXLA(t *testing.T) {
	t.Skip("Generative models are not supported yet for XLA")
	runXLAPipeline(t, testutil.TextGenerationPipeline)
}

func TestTextGenerationPipelineXLACuda(t *testing.T) {
	t.Skip("Generative models are not supported yet for XLA")
	runXLAPipelineCuda(t, testutil.TextGenerationPipeline)
}

func TestTextGenerationPipelineValidationXLA(t *testing.T) {
	t.Skip("Generative models are not supported yet for XLA")
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
	testutil.QuestionAnsweringPipelineValidation(t, nil)
}

// Tabular pipeline

func TestTabularPipelineXLA(t *testing.T) {
	t.Skip("Currently missing TreeEnsembleClassifier ONNX operator")
	runXLAPipeline(t, testutil.TabularPipeline)
}

func TestTabularPipelineXLACuda(t *testing.T) {
	t.Skip("Currently missing TreeEnsembleClassifier ONNX operator")
	runXLAPipelineCuda(t, testutil.TabularPipeline)
}

func TestTabularPipelineValidationXLA(t *testing.T) {
	testutil.TabularPipelineValidation(t, nil)
}

// No same name

func TestNoSameNamePipelineXLA(t *testing.T) {
	runXLAPipeline(t, testutil.NoSameNamePipeline)
}

func TestNoSameNameAcrossTypesPipelineXLA(t *testing.T) {
	runXLAPipeline(t, testutil.NoSameNameAcrossTypesPipeline)
}

func TestDestroyPipelineXLA(t *testing.T) {
	runXLAPipeline(t, testutil.DestroyPipelines)
}

// Thread safety

func TestThreadSafetyXLA(t *testing.T) {
	runXLAPipeline(t, func(t *testing.T, session *hugot.Session) {
		testutil.ThreadSafety(t, session, 250)
	})
}

func TestThreadSafetyXLACuda(t *testing.T) {
	runXLAPipelineCuda(t, func(t *testing.T, session *hugot.Session) {
		testutil.ThreadSafety(t, session, 1000)
	})
}
