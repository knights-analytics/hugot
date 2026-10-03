//go:build (GO || ALL) && !TRAINING

package go_test

import (
	"testing"

	testutil "github.com/knights-analytics/hugot/tests"
)

// Zero-shot audio classification

func TestZeroShotAudioClassificationPipelineGo(t *testing.T) {
	t.Skip("Skipping test due to known nonConstantDepedencies given an unknown node output name \"\" issue with Go pipeline")
	runGoPipeline(t, testutil.ZeroShotAudioClassificationPipeline)
}

func TestZeroShotAudioClassificationPipelineValidationGo(t *testing.T) {
	t.Skip("Skipping test due to known nonConstantDepedencies given an unknown node output name \"\" issue with Go pipeline")
	runGoPipelineValidation(t, testutil.ZeroShotAudioClassificationPipelineValidation)
}

// Automatic speech recognition

func TestAutomaticSpeechRecognitionPipelineGo(t *testing.T) {
	t.Skip("Skipping test due to known attrs[epsilon (FLOAT)] issue with Go pipeline")
	runGoPipeline(t, testutil.AutomaticSpeechRecognitionPipeline)
}

func TestAutomaticSpeechRecognitionPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.AutomaticSpeechRecognitionPipelineValidation)
}

// Text-to-speech

func TestTextToSpeechPipelineGo(t *testing.T) {
	t.Skip("Skipping test due to known nonConstantDepedencies given an unknown node output name issue with Go pipeline")
	runGoPipeline(t, testutil.TextToSpeechPipeline)
}

func TestTextToSpeechPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.TextToSpeechPipelineValidation)
}

// Text-to-audio

func TestTextToAudioPipelineGo(t *testing.T) {
	t.Skip("Skipping test due to known nonConstantDepedencies given an unknown node output name issue with Go pipeline")
	runGoPipeline(t, testutil.TextToAudioPipeline)
}

func TestTextToAudioPipelineValidationGo(t *testing.T) {
	t.Skip("Skipping test due to known nonConstantDepedencies given an unknown node output name issue with Go pipeline")
	runGoPipelineValidation(t, testutil.TextToAudioPipelineValidation)
}

// Audio classification

func TestAudioClassificationPipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.AudioClassificationPipeline)
}

func TestAudioClassificationPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.AudioClassificationPipelineValidation)
}
