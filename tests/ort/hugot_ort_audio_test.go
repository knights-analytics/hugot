//go:build cgo && (ORT || ALL) && !TRAINING

package ort_test

import (
	"testing"

	testutil "github.com/knights-analytics/hugot/tests"
)

// Zero-shot audio classification

func TestZeroShotAudioClassificationPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.ZeroShotAudioClassificationPipeline)
}

func TestZeroShotAudioClassificationPipelineORTGoMLX(t *testing.T) {
	t.Skip("Skipping test due to known nonConstantDepedencies given an unknown node output name \"\" issue with GoMLX pipeline")
	runORTPipelineGoMLX(t, testutil.ZeroShotAudioClassificationPipeline)
}

func TestZeroShotAudioClassificationPipelineORTCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.ZeroShotAudioClassificationPipeline)
}

func TestZeroShotAudioClassificationPipelineValidationORT(t *testing.T) {
	runORTPipelineValidation(t, testutil.ZeroShotAudioClassificationPipelineValidation)
}

// Automatic speech recognition

func TestAutomaticSpeechRecognitionPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.AutomaticSpeechRecognitionPipeline)
}

func TestAutomaticSpeechRecognitionPipelineORTGoMLX(t *testing.T) {
	t.Skip("Skipping test due to known attrs[epsilon (FLOAT)] issue with GoMLX")
	runORTPipelineGoMLX(t, testutil.AutomaticSpeechRecognitionPipeline)
}

func TestAutomaticSpeechRecognitionPipelineORTCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.AutomaticSpeechRecognitionPipeline)
}

func TestAutomaticSpeechRecognitionPipelineValidationORT(t *testing.T) {
	runORTPipelineValidation(t, testutil.AutomaticSpeechRecognitionPipelineValidation)
}

// Text-to-speech

func TestTextToSpeechPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.TextToSpeechPipeline)
}

func TestTextToSpeechPipelineORTGoMLX(t *testing.T) {
	t.Skip("Skipping test due to known nonConstantDepedencies given an unknown node output name issue with GoMLX")
	runORTPipelineGoMLX(t, testutil.TextToSpeechPipeline)
}

func TestTextToSpeechPipelineORTCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.TextToSpeechPipeline)
}

func TestTextToSpeechPipelineValidationORT(t *testing.T) {
	runORTPipelineValidation(t, testutil.TextToSpeechPipelineValidation)
}

// Text-to-audio

func TestTextToAudioPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.TextToAudioPipeline)
}

func TestTextToAudioPipelineORTGoMLX(t *testing.T) {
	t.Skip("Skipping test due to known nonConstantDepedencies given an unknown node output name issue with GoMLX pipeline")
	runORTPipelineGoMLX(t, testutil.TextToAudioPipeline)
}

func TestTextToAudioPipelineORTCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.TextToAudioPipeline)
}

func TestTextToAudioPipelineValidationORT(t *testing.T) {
	runORTPipelineValidation(t, testutil.TextToAudioPipelineValidation)
}

// Audio classification

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
