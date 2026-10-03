//go:build cgo && (XLA || ALL) && !TRAINING

package xla_test

import (
	"testing"

	testutil "github.com/knights-analytics/hugot/tests"
)

// Zero-shot audio classification

func TestZeroShotAudioClassificationPipelineXLA(t *testing.T) {
	t.Skip("Skipping test due to known nonConstantDepedencies given an unknown node output name \"\" issue with Go pipeline")
	runXLAPipeline(t, testutil.ZeroShotAudioClassificationPipeline)
}

func TestZeroShotAudioClassificationPipelineXLACuda(t *testing.T) {
	t.Skip("Skipping test due to known nonConstantDepedencies given an unknown node output name \"\" issue with Go pipeline")
	runXLAPipelineCuda(t, testutil.ZeroShotAudioClassificationPipeline)
}

func TestZeroShotAudioClassificationPipelineValidationXLA(t *testing.T) {
	t.Skip("Skipping test due to known nonConstantDepedencies given an unknown node output name \"\" issue with Go pipeline")
	runXLAPipelineValidation(t, testutil.ZeroShotAudioClassificationPipelineValidation)
}

// Automatic speech recognition

func TestAutomaticSpeechRecognitionPipelineXLA(t *testing.T) {
	t.Skip("Skipping test due to known attrs[epsilon (FLOAT)] issue with XLA pipeline")
	runXLAPipeline(t, testutil.AutomaticSpeechRecognitionPipeline)
}

func TestAutomaticSpeechRecognitionPipelineXLACuda(t *testing.T) {
	t.Skip("Skipping test due to known attrs[epsilon (FLOAT)] issue with XLA pipeline")
	runXLAPipelineCuda(t, testutil.AutomaticSpeechRecognitionPipeline)
}

func TestAutomaticSpeechRecognitionPipelineValidationXLA(t *testing.T) {
	runXLAPipelineValidation(t, testutil.AutomaticSpeechRecognitionPipelineValidation)
}

// Text-to-speech

func TestTextToSpeechPipelineXLA(t *testing.T) {
	t.Skip("Skipping test due to known nonConstantDepedencies given an unknown node output name issue with XLA pipeline")
	runXLAPipeline(t, testutil.TextToSpeechPipeline)
}

func TestTextToSpeechPipelineXLACuda(t *testing.T) {
	t.Skip("Skipping test due to known nonConstantDepedencies given an unknown node output name issue with Go pipeline")
	runXLAPipelineCuda(t, testutil.TextToSpeechPipeline)
}

func TestTextToSpeechPipelineValidationXLA(t *testing.T) {
	runXLAPipelineValidation(t, testutil.TextToSpeechPipelineValidation)
}

// Text-to-audio

func TestTextToAudioPipelineXLA(t *testing.T) {
	t.Skip("Skipping test due to known nonConstantDepedencies given an unknown node output name issue with Go pipeline")
	runXLAPipeline(t, testutil.TextToAudioPipeline)
}

func TestTextToAudioPipelineXLACuda(t *testing.T) {
	t.Skip("Skipping test due to known nonConstantDepedencies given an unknown node output name issue with Go pipeline")
	runXLAPipelineCuda(t, testutil.TextToAudioPipeline)
}

func TestTextToAudioPipelineValidationXLA(t *testing.T) {
	t.Skip("Skipping test due to known nonConstantDepedencies given an unknown node output name issue with Go pipeline")
	runXLAPipelineValidation(t, testutil.TextToAudioPipelineValidation)
}

// Audio classification

func TestAudioClassificationPipelineXLA(t *testing.T) {
	runXLAPipeline(t, testutil.AudioClassificationPipeline)
}

func TestAudioClassificationPipelineXLACuda(t *testing.T) {
	runXLAPipelineCuda(t, testutil.AudioClassificationPipeline)
}

func TestAudioClassificationPipelineValidationXLA(t *testing.T) {
	runXLAPipelineValidation(t, testutil.AudioClassificationPipelineValidation)
}
