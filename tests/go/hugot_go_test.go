//go:build (GO || ALL) && !TRAINING

package go_test

import (
	"encoding/json"
	"fmt"
	"os"
	"testing"

	"github.com/knights-analytics/hugot"
	testutil "github.com/knights-analytics/hugot/tests"
)

func runGoPipeline(t *testing.T, run func(*testing.T, *hugot.Session)) {
	t.Helper()
	session, err := hugot.NewGoSession(t.Context())
	testutil.CheckT(t, err)
	defer func() { testutil.CheckT(t, session.Destroy()) }()
	run(t, session)
}

func runGoPipelineValidation(t *testing.T, run func(*testing.T, *hugot.Session)) {
	t.Helper()
	runGoPipeline(t, run)
}

// FEATURE EXTRACTION

func TestFeatureExtractionPipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.FeatureExtractionPipeline)
}

func TestFeatureExtractionPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.FeatureExtractionPipelineValidation)
}

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

// Zero shot

func TestZeroShotClassificationPipelineGo(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	runGoPipeline(t, testutil.ZeroShotClassificationPipeline)
}

func TestZeroShotClassificationPipelineValidationGo(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	runGoPipelineValidation(t, testutil.ZeroShotClassificationPipelineValidation)
}

// Cross Encoder

func TestCrossEncoderPipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.CrossEncoderPipeline)
}

func TestCrossEncoderPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.CrossEncoderPipelineValidation)
}

// Image classification

func TestImageClassificationPipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.ImageClassificationPipeline)
}

func TestImageClassificationPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.ImageClassificationPipelineValidation)
}

// Object detection

func TestObjectDetectionPipelineGo(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	runGoPipeline(t, testutil.ObjectDetectionPipeline)
}

func TestObjectDetectionPipelineValidationGo(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	runGoPipelineValidation(t, testutil.ObjectDetectionPipelineValidation)
}

// fill-mask

func TestFillMaskPipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.FillMaskPipeline)
}

func TestFillMaskPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.FillMaskPipelineValidation)
}

// image feature extraction

func TestImageFeatureExtractionPipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.ImageFeatureExtractionPipeline)
}

func TestImageFeatureExtractionPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.ImageFeatureExtractionPipelineValidation)
}

// image segmentation

func TestImageSegmentationPipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.ImageSegmentationPipeline)
}

func TestImageSegmentationPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.ImageSegmentationPipelineValidation)
}

// depth estimation

func TestDepthEstimationPipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.DepthEstimationPipeline)
}

func TestDepthEstimationPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.DepthEstimationPipelineValidation)
}

// summarization

func TestSummarizationPipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.SummarizationPipeline)
}

func TestSummarizationPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.SummarizationPipelineValidation)
}

// translation

func TestTranslationPipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.TranslationPipeline)
}

func TestTranslationPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.TranslationPipelineValidation)
}

// text-to-text generation

func TestText2TextGenerationPipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.Text2TextGenerationPipeline)
}

func TestText2TextGenerationPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.Text2TextGenerationPipelineValidation)
}

// zero-shot object detection

func TestZeroShotObjectDetectionPipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.ZeroShotObjectDetectionPipeline)
}

func TestZeroShotObjectDetectionPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.ZeroShotObjectDetectionPipelineValidation)
}

// mask generation

func TestMaskGenerationPipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.MaskGenerationPipeline)
}

func TestMaskGenerationPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.MaskGenerationPipelineValidation)
}

// zero-shot audio classification

func TestZeroShotAudioClassificationPipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.ZeroShotAudioClassificationPipeline)
}

func TestZeroShotAudioClassificationPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.ZeroShotAudioClassificationPipelineValidation)
}

// visual question answering

func TestVisualQuestionAnsweringPipelineGo(t *testing.T) {
	t.Skip("visual question answering requires the ORT GenAI multimodal runtime")
	runGoPipeline(t, testutil.VisualQuestionAnsweringPipeline)
}

func TestVisualQuestionAnsweringPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.VisualQuestionAnsweringPipelineValidation)
}

// document question answering

func TestDocumentQuestionAnsweringPipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.DocumentQuestionAnsweringPipeline)
}

func TestDocumentQuestionAnsweringPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.DocumentQuestionAnsweringPipelineValidation)
}

// table question answering

func TestTableQuestionAnsweringPipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.TableQuestionAnsweringPipeline)
}

func TestTableQuestionAnsweringPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.TableQuestionAnsweringPipelineValidation)
}

// text-to-speech

func TestTextToSpeechPipelineGo(t *testing.T) {
	t.Skip("Xenova_mms-tts-eng cannot be compiled by the GoMLX backend")
	runGoPipeline(t, testutil.TextToSpeechPipeline)
}

func TestTextToSpeechPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.TextToSpeechPipelineValidation)
}

// text-to-audio

func TestTextToAudioPipelineGo(t *testing.T) {
	t.Skip("Xenova_mms-tts-eng cannot be compiled by the GoMLX backend")
	runGoPipeline(t, testutil.TextToAudioPipeline)
}

func TestTextToAudioPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.TextToAudioPipelineValidation)
}

// audio classification

func TestAudioClassificationPipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.AudioClassificationPipeline)
}

func TestAudioClassificationPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.AudioClassificationPipelineValidation)
}

// background removal

func TestBackgroundRemovalPipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.BackgroundRemovalPipeline)
}

func TestBackgroundRemovalPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.BackgroundRemovalPipelineValidation)
}

// zero-shot image classification

func TestZeroShotImageClassificationPipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.ZeroShotImageClassificationPipeline)
}

func TestZeroShotImageClassificationPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.ZeroShotImageClassificationPipelineValidation)
}

// automatic speech recognition

func TestAutomaticSpeechRecognitionPipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.AutomaticSpeechRecognitionPipeline)
}

func TestAutomaticSpeechRecognitionPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.AutomaticSpeechRecognitionPipelineValidation)
}

// image-to-text

func TestImageToTextPipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.ImageToTextPipeline)
}

func TestImageToTextPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.ImageToTextPipelineValidation)
}

// image-text-to-text

func TestImageTextToTextPipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.ImageTextToTextPipeline)
}

func TestImageTextToTextPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.ImageTextToTextPipelineValidation)
}

// Text generation
// These currently only run locally due to resource constraints in CI/CD

func TestTextGenerationPipelineGo(t *testing.T) {
	t.Skip("Generative models are not supported yet for Go")
	runGoPipeline(t, testutil.TextGenerationPipeline)
}

func TestTextGenerationPipelineValidationGo(t *testing.T) {
	t.Skip("Generative models are not supported yet for Go")
	runGoPipelineValidation(t, testutil.TextGenerationPipelineValidation)
}

// Question answering

func TestQAPipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.QuestionAnsweringPipeline)
}

func TestQAPipelineValidationGo(t *testing.T) {
	testutil.QuestionAnsweringPipelineValidation(t, nil)
}

// Tabular pipeline

func TestTabularPipelineGo(t *testing.T) {
	t.Skip("Currently missing TreeEnsembleClassifier ONNX operator")
	runGoPipeline(t, testutil.TabularPipeline)
}

func TestTabularPipelineValidationGo(t *testing.T) {
	testutil.TabularPipelineValidation(t, nil)
}

// No same name

func TestNoSameNamePipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.NoSameNamePipeline)
}

func TestNoSameNameAcrossTypesPipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.NoSameNameAcrossTypesPipeline)
}

func TestDestroyPipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.DestroyPipelines)
}

// Thread safety

func TestThreadSafetyGo(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	runGoPipeline(t, func(t *testing.T, session *hugot.Session) {
		testutil.ThreadSafety(t, session, 20)
	})
}

// README: test the readme examples

func TestReadmeExample(t *testing.T) {
	t.Helper()
	check := func(err error) {
		if err != nil {
			panic(err.Error())
		}
	}

	session, err := hugot.NewGoSession(t.Context())
	check(err)

	defer func(session *hugot.Session) {
		err := session.Destroy()
		check(err)
	}(session)

	modelPath := testutil.ModelsFolder + "KnightsAnalytics_distilbert-base-uncased-finetuned-sst-2-english"

	config := hugot.TextClassificationConfig{
		ModelPath: modelPath,
		Name:      "testPipeline",
	}

	sentimentPipeline, err := hugot.NewPipeline(session, config)
	check(err)

	batch := []string{"This movie is disgustingly good !", "The director tried too much"}
	batchResult, err := sentimentPipeline.RunPipeline(t.Context(), batch)
	check(err)

	s, err := json.Marshal(batchResult)
	check(err)
	fmt.Println(string(s))
	// OUTPUT: {"ClassificationOutputs":[[{"Label":"POSITIVE","Score":0.9998536}],[{"Label":"NEGATIVE","Score":0.99752176}]]}
}
