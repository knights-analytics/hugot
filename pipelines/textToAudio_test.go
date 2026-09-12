package pipelines

import (
	"strings"
	"testing"

	"github.com/knights-analytics/hugot/backends"
)

func TestTextToAudioPipelineValidate(t *testing.T) {
	model := &backends.Model{
		InputsMeta:  []backends.InputOutputInfo{{Name: "input_ids", Dimensions: backends.Shape{-1, -1}}},
		OutputsMeta: []backends.InputOutputInfo{{Name: "audio_values", Dimensions: backends.Shape{-1, -1}}}}
	pipeline := &TextToAudioPipeline{BasePipeline: &backends.BasePipeline{Model: model}, SampleRate: 16000}
	if err := pipeline.Validate(); err == nil || !strings.Contains(err.Error(), "requires a tokenizer") {
		t.Fatalf("expected tokenizer validation error, got %v", err)
	}
}

func TestTextToAudioPipelinePostprocess(t *testing.T) {
	pipeline := &TextToAudioPipeline{SampleRate: 22050}
	output, err := pipeline.postprocess(&backends.PipelineBatch{
		Size:         2,
		OutputValues: []any{[][]float32{{0.1, -0.2}, {0.3, 0.4}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(output.Audio) != 2 || output.Audio[0].SampleRate != 22050 || len(output.Audio[1].Samples) != 2 {
		t.Fatalf("unexpected audio output: %#v", output.Audio)
	}
}

func TestTextToAudioPipelinePostprocessRejectsStereo(t *testing.T) {
	pipeline := &TextToAudioPipeline{SampleRate: 16000}
	if _, err := pipeline.postprocess(&backends.PipelineBatch{
		Size:         1,
		OutputValues: []any{[][][]float32{{{0.1}, {0.2}}}},
	}); err == nil || !strings.Contains(err.Error(), "only mono audio") {
		t.Fatalf("expected mono validation error, got %v", err)
	}
}

func TestWithTextToAudioSampleRateRejectsInvalidValue(t *testing.T) {
	pipeline := &TextToAudioPipeline{}
	if err := WithTextToAudioSampleRate(0)(pipeline); err == nil {
		t.Fatal("expected invalid sample rate to fail")
	}
}
