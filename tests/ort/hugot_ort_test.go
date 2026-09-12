//go:build cgo && (ORT || ALL) && !TRAINING

package ort_test

import (
	"os"
	"testing"

	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/options"
	testutil "github.com/knights-analytics/hugot/tests"
)

func runORTPipeline(t *testing.T, run func(*testing.T, *hugot.Session), opts ...options.WithOption) {
	t.Helper()
	session, err := hugot.NewORTSession(t.Context(), opts...)
	testutil.CheckT(t, err)
	defer func() { testutil.CheckT(t, session.Destroy()) }()
	run(t, session)
}

func runORTPipelineGoMLX(t *testing.T, run func(*testing.T, *hugot.Session), opts ...options.WithOption) {
	t.Helper()
	opts = append([]options.WithOption{options.WithGoMLX()}, opts...)
	runORTPipeline(t, run, opts...)
}

func runORTPipelineValidation(t *testing.T, run func(*testing.T, *hugot.Session), opts ...options.WithOption) {
	t.Helper()
	runORTPipeline(t, run, opts...)
}

func runORTPipelineCuda(t *testing.T, run func(*testing.T, *hugot.Session), opts ...options.WithOption) {
	t.Helper()
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	opts = append([]options.WithOption{options.WithCuda(map[string]string{"device_id": "0"})}, opts...)
	runORTPipeline(t, run, opts...)
}

var textClassificationORTOptions = []options.WithOption{
	options.WithOnnxLibraryPath("/usr/lib"),
	options.WithTelemetry(),
	options.WithCPUMemArena(true),
	options.WithMemPattern(true),
	options.WithIntraOpNumThreads(1),
	options.WithInterOpNumThreads(1),
}

// FEATURE EXTRACTION

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

// Zero shot

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

// Cross Encoder

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

// Image classification

func TestImageClassificationPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.ImageClassificationPipeline)
}

func TestImageClassificationPipelineORTGoMLX(t *testing.T) {
	runORTPipelineGoMLX(t, testutil.ImageClassificationPipeline)
}

func TestImageClassificationPipelineORTCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.ImageClassificationPipeline)
}

func TestImageClassificationPipelineValidationORT(t *testing.T) {
	runORTPipelineValidation(t, testutil.ImageClassificationPipelineValidation)
}

// Object detection

func TestObjectDetectionPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.ObjectDetectionPipeline)
}

func TestObjectDetectionPipelineORTGoMLX(t *testing.T) {
	runORTPipelineGoMLX(t, testutil.ObjectDetectionPipeline)
}

func TestObjectDetectionPipelineORTCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.ObjectDetectionPipeline)
}

func TestObjectDetectionPipelineValidationORT(t *testing.T) {
	runORTPipelineValidation(t, testutil.ObjectDetectionPipelineValidation)
}

// fill-mask

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

// image feature extraction

func TestImageFeatureExtractionPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.ImageFeatureExtractionPipeline)
}

func TestImageFeatureExtractionPipelineORTGoMLX(t *testing.T) {
	runORTPipelineGoMLX(t, testutil.ImageFeatureExtractionPipeline)
}

func TestImageFeatureExtractionPipelineORTCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.ImageFeatureExtractionPipeline)
}

func TestImageFeatureExtractionPipelineValidationORT(t *testing.T) {
	runORTPipelineValidation(t, testutil.ImageFeatureExtractionPipelineValidation)
}

// image segmentation

func TestImageSegmentationPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.ImageSegmentationPipeline)
}

func TestImageSegmentationPipelineORTGoMLX(t *testing.T) {
	runORTPipelineGoMLX(t, testutil.ImageSegmentationPipeline)
}

func TestImageSegmentationPipelineORTCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.ImageSegmentationPipeline)
}

func TestImageSegmentationPipelineValidationORT(t *testing.T) {
	runORTPipelineValidation(t, testutil.ImageSegmentationPipelineValidation)
}

// depth estimation

func TestDepthEstimationPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.DepthEstimationPipeline)
}

func TestDepthEstimationPipelineORTGoMLX(t *testing.T) {
	runORTPipelineGoMLX(t, testutil.DepthEstimationPipeline)
}

func TestDepthEstimationPipelineORTCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.DepthEstimationPipeline)
}

func TestDepthEstimationPipelineValidationORT(t *testing.T) {
	runORTPipelineValidation(t, testutil.DepthEstimationPipelineValidation)
}

// summarization

func TestSummarizationPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.SummarizationPipeline)
}

func TestSummarizationPipelineORTGoMLX(t *testing.T) {
	runORTPipelineGoMLX(t, testutil.SummarizationPipeline)
}

func TestSummarizationPipelineORTCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.SummarizationPipeline)
}

func TestSummarizationPipelineValidationORT(t *testing.T) {
	runORTPipelineValidation(t, testutil.SummarizationPipelineValidation)
}

// translation

func TestTranslationPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.TranslationPipeline)
}

func TestTranslationPipelineORTGoMLX(t *testing.T) {
	runORTPipelineGoMLX(t, testutil.TranslationPipeline)
}

func TestTranslationPipelineORTCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.TranslationPipeline)
}

func TestTranslationPipelineValidationORT(t *testing.T) {
	runORTPipelineValidation(t, testutil.TranslationPipelineValidation)
}

// text-to-text generation

func TestText2TextGenerationPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.Text2TextGenerationPipeline)
}

func TestText2TextGenerationPipelineORTGoMLX(t *testing.T) {
	runORTPipelineGoMLX(t, testutil.Text2TextGenerationPipeline)
}

func TestText2TextGenerationPipelineORTCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.Text2TextGenerationPipeline)
}

func TestText2TextGenerationPipelineValidationORT(t *testing.T) {
	runORTPipelineValidation(t, testutil.Text2TextGenerationPipelineValidation)
}

// zero-shot object detection

func TestZeroShotObjectDetectionPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.ZeroShotObjectDetectionPipeline)
}

func TestZeroShotObjectDetectionPipelineORTGoMLX(t *testing.T) {
	runORTPipelineGoMLX(t, testutil.ZeroShotObjectDetectionPipeline)
}

func TestZeroShotObjectDetectionPipelineORTCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.ZeroShotObjectDetectionPipeline)
}

func TestZeroShotObjectDetectionPipelineValidationORT(t *testing.T) {
	runORTPipelineValidation(t, testutil.ZeroShotObjectDetectionPipelineValidation)
}

// mask generation

func TestMaskGenerationPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.MaskGenerationPipeline)
}

func TestMaskGenerationPipelineORTGoMLX(t *testing.T) {
	runORTPipelineGoMLX(t, testutil.MaskGenerationPipeline)
}

func TestMaskGenerationPipelineORTCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.MaskGenerationPipeline)
}

func TestMaskGenerationPipelineValidationORT(t *testing.T) {
	runORTPipelineValidation(t, testutil.MaskGenerationPipelineValidation)
}

// zero-shot audio classification

func TestZeroShotAudioClassificationPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.ZeroShotAudioClassificationPipeline)
}

func TestZeroShotAudioClassificationPipelineORTGoMLX(t *testing.T) {
	runORTPipelineGoMLX(t, testutil.ZeroShotAudioClassificationPipeline)
}

func TestZeroShotAudioClassificationPipelineORTCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.ZeroShotAudioClassificationPipeline)
}

func TestZeroShotAudioClassificationPipelineValidationORT(t *testing.T) {
	runORTPipelineValidation(t, testutil.ZeroShotAudioClassificationPipelineValidation)
}

// visual question answering

func TestVisualQuestionAnsweringPipelineORT(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.Skip("Phi-3.5 vision test fixture is not downloaded in CI")
	}
	runORTPipeline(t, testutil.VisualQuestionAnsweringPipeline)
}

func TestVisualQuestionAnsweringPipelineORTGoMLX(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.Skip("Phi-3.5 vision test fixture is not downloaded in CI")
	}
	runORTPipelineGoMLX(t, testutil.VisualQuestionAnsweringPipeline)
}

func TestVisualQuestionAnsweringPipelineORTCuda(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.Skip("Phi-3.5 vision test fixture is not downloaded in CI")
	}
	runORTPipelineCuda(t, testutil.VisualQuestionAnsweringPipeline)
}

func TestVisualQuestionAnsweringPipelineValidationORT(t *testing.T) {
	runORTPipelineValidation(t, testutil.VisualQuestionAnsweringPipelineValidation)
}

// document question answering

func TestDocumentQuestionAnsweringPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.DocumentQuestionAnsweringPipeline)
}

func TestDocumentQuestionAnsweringPipelineORTGoMLX(t *testing.T) {
	runORTPipelineGoMLX(t, testutil.DocumentQuestionAnsweringPipeline)
}

func TestDocumentQuestionAnsweringPipelineORTCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.DocumentQuestionAnsweringPipeline)
}

func TestDocumentQuestionAnsweringPipelineValidationORT(t *testing.T) {
	runORTPipelineValidation(t, testutil.DocumentQuestionAnsweringPipelineValidation)
}

// table question answering

func TestTableQuestionAnsweringPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.TableQuestionAnsweringPipeline)
}

func TestTableQuestionAnsweringPipelineORTGoMLX(t *testing.T) {
	runORTPipelineGoMLX(t, testutil.TableQuestionAnsweringPipeline)
}

func TestTableQuestionAnsweringPipelineORTCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.TableQuestionAnsweringPipeline)
}

func TestTableQuestionAnsweringPipelineValidationORT(t *testing.T) {
	runORTPipelineValidation(t, testutil.TableQuestionAnsweringPipelineValidation)
}

// text-to-speech

func TestTextToSpeechPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.TextToSpeechPipeline)
}

func TestTextToSpeechPipelineORTGoMLX(t *testing.T) {
	runORTPipelineGoMLX(t, testutil.TextToSpeechPipeline)
}

func TestTextToSpeechPipelineORTCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.TextToSpeechPipeline)
}

func TestTextToSpeechPipelineValidationORT(t *testing.T) {
	runORTPipelineValidation(t, testutil.TextToSpeechPipelineValidation)
}

// text-to-audio

func TestTextToAudioPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.TextToAudioPipeline)
}

func TestTextToAudioPipelineORTGoMLX(t *testing.T) {
	runORTPipelineGoMLX(t, testutil.TextToAudioPipeline)
}

func TestTextToAudioPipelineORTCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.TextToAudioPipeline)
}

func TestTextToAudioPipelineValidationORT(t *testing.T) {
	runORTPipelineValidation(t, testutil.TextToAudioPipelineValidation)
}

// audio classification

func TestAudioClassificationPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.AudioClassificationPipeline)
}

func TestAudioClassificationPipelineORTGoMLX(t *testing.T) {
	runORTPipelineGoMLX(t, testutil.AudioClassificationPipeline)
}

func TestAudioClassificationPipelineORTCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.AudioClassificationPipeline)
}

func TestAudioClassificationPipelineValidationORT(t *testing.T) {
	runORTPipelineValidation(t, testutil.AudioClassificationPipelineValidation)
}

// background removal

func TestBackgroundRemovalPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.BackgroundRemovalPipeline)
}

func TestBackgroundRemovalPipelineORTGoMLX(t *testing.T) {
	runORTPipelineGoMLX(t, testutil.BackgroundRemovalPipeline)
}

func TestBackgroundRemovalPipelineORTCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.BackgroundRemovalPipeline)
}

func TestBackgroundRemovalPipelineValidationORT(t *testing.T) {
	runORTPipelineValidation(t, testutil.BackgroundRemovalPipelineValidation)
}

// zero-shot image classification

func TestZeroShotImageClassificationPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.ZeroShotImageClassificationPipeline)
}

func TestZeroShotImageClassificationPipelineORTGoMLX(t *testing.T) {
	runORTPipelineGoMLX(t, testutil.ZeroShotImageClassificationPipeline)
}

func TestZeroShotImageClassificationPipelineORTCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.ZeroShotImageClassificationPipeline)
}

func TestZeroShotImageClassificationPipelineValidationORT(t *testing.T) {
	runORTPipelineValidation(t, testutil.ZeroShotImageClassificationPipelineValidation)
}

// automatic speech recognition

func TestAutomaticSpeechRecognitionPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.AutomaticSpeechRecognitionPipeline)
}

func TestAutomaticSpeechRecognitionPipelineORTGoMLX(t *testing.T) {
	runORTPipelineGoMLX(t, testutil.AutomaticSpeechRecognitionPipeline)
}

func TestAutomaticSpeechRecognitionPipelineORTCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.AutomaticSpeechRecognitionPipeline)
}

func TestAutomaticSpeechRecognitionPipelineValidationORT(t *testing.T) {
	runORTPipelineValidation(t, testutil.AutomaticSpeechRecognitionPipelineValidation)
}

// image-to-text

func TestImageToTextPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.ImageToTextPipeline)
}

func TestImageToTextPipelineORTGoMLX(t *testing.T) {
	runORTPipelineGoMLX(t, testutil.ImageToTextPipeline)
}

func TestImageToTextPipelineORTCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.ImageToTextPipeline)
}

func TestImageToTextPipelineValidationORT(t *testing.T) {
	runORTPipelineValidation(t, testutil.ImageToTextPipelineValidation)
}

// image-text-to-text

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

// Text generation
// These currently only run locally due to resource constraints in CI/CD

func TestTextGenerationPipelineORT(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	runORTPipeline(t, testutil.TextGenerationPipeline)
}

func TestTextGenerationPipelineORTGoMLX(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	runORTPipelineGoMLX(t, testutil.TextGenerationPipeline, options.WithGenerativeEngine())
}

func TestTextGenerationPipelineORTCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.TextGenerationPipeline, options.WithGenerativeEngine())
}

func TestTextGenerationPipelineValidationORT(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
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
	testutil.QuestionAnsweringPipelineValidation(t, nil)
}

// Tabular pipeline

func TestTabularPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.TabularPipeline)
}

func TestTabularPipelineORTGoMLX(t *testing.T) {
	t.Skip("Currently missing TreeEnsembleClassifier ONNX operator")
	runORTPipelineGoMLX(t, testutil.TabularPipeline)
}

func TestTabularPipelineORTCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.TabularPipeline)
}

func TestTabularPipelineValidationORT(t *testing.T) {
	testutil.TabularPipelineValidation(t, nil)
}

// No same name

func TestNoSameNamePipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.NoSameNamePipeline)
}

func TestNoSameNameAcrossTypesPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.NoSameNameAcrossTypesPipeline)
}

func TestClosePipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.DestroyPipelines)
}

// Thread safety

func TestThreadSafetyORT(t *testing.T) {
	runORTPipeline(t, func(t *testing.T, session *hugot.Session) {
		testutil.ThreadSafety(t, session, 250)
	})
}

func TestThreadSafetyGoMLX(t *testing.T) {
	runORTPipelineGoMLX(t, func(t *testing.T, session *hugot.Session) {
		testutil.ThreadSafety(t, session, 250)
	})
}

func TestThreadSafetyORTCuda(t *testing.T) {
	runORTPipelineCuda(t, func(t *testing.T, session *hugot.Session) {
		testutil.ThreadSafety(t, session, 1000)
	})
}
