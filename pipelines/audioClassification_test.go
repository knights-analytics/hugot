package pipelines

import (
	"strings"
	"testing"

	"github.com/knights-analytics/hugot/backends"
)

func TestAudioClassificationPipelineValidate(t *testing.T) {
	model := &backends.Model{
		InputsMeta:  []backends.InputOutputInfo{{Name: "input_values", Dimensions: backends.Shape{-1, -1}}},
		OutputsMeta: []backends.InputOutputInfo{{Name: "logits", Dimensions: backends.Shape{-1, 3}}},
		IDLabelMap:  map[int]string{0: "zero", 1: "one", 2: "two"}}
	pipeline := &AudioClassificationPipeline{BasePipeline: &backends.BasePipeline{Model: model}, TopK: 2, IDLabelMap: model.IDLabelMap}
	if err := pipeline.Validate(); err != nil {
		t.Fatalf("expected valid audio configuration: %v", err)
	}
	if pipeline.Input.Name != "input_values" {
		t.Fatalf("unexpected input metadata: %#v", pipeline.Input)
	}
}

func TestAudioClassificationPipelineRejectsUnsupportedInputLayout(t *testing.T) {
	model := &backends.Model{
		InputsMeta:  []backends.InputOutputInfo{{Name: "input_values", Dimensions: backends.Shape{-1, 1, 1, -1}}},
		OutputsMeta: []backends.InputOutputInfo{{Name: "logits", Dimensions: backends.Shape{-1, 2}}}}
	pipeline := &AudioClassificationPipeline{BasePipeline: &backends.BasePipeline{Model: model}, TopK: 1}
	if err := pipeline.Validate(); err == nil || !strings.Contains(err.Error(), "unsupported audio input layout") {
		t.Fatalf("expected explicit unsupported layout error, got %v", err)
	}
}

func TestAudioClassificationPipelineAcceptsRankThreeInput(t *testing.T) {
	model := &backends.Model{
		InputsMeta:  []backends.InputOutputInfo{{Name: "audio", Dimensions: backends.Shape{-1, 1, -1}}},
		OutputsMeta: []backends.InputOutputInfo{{Name: "logits", Dimensions: backends.Shape{-1, 2}}}}
	pipeline := &AudioClassificationPipeline{BasePipeline: &backends.BasePipeline{Model: model}, TopK: 1}
	if err := pipeline.Validate(); err != nil {
		t.Fatalf("expected rank-three audio input to be valid: %v", err)
	}
}

func TestAudioClassificationPipelinePostprocessTopK(t *testing.T) {
	pipeline := &AudioClassificationPipeline{
		BasePipeline: &backends.BasePipeline{Model: &backends.Model{}},
		IDLabelMap:   map[int]string{0: "quiet", 1: "speech", 2: "music"},
		TopK:         2,
	}
	output, err := pipeline.postprocess(&backends.PipelineBatch{OutputValues: []any{[][]float32{{0.1, 0.9, 0.4}}}})
	if err != nil {
		t.Fatal(err)
	}
	if len(output.Predictions) != 1 || len(output.Predictions[0]) != 2 {
		t.Fatalf("unexpected predictions: %#v", output.Predictions)
	}
	if output.Predictions[0][0].Label != "speech" || output.Predictions[0][1].Label != "music" {
		t.Fatalf("unexpected prediction order: %#v", output.Predictions)
	}
	if output.Predictions[0][0].Score <= output.Predictions[0][1].Score || output.Predictions[0][0].Score >= 1 {
		t.Fatalf("expected softmax probabilities, got %#v", output.Predictions)
	}
}

func TestAudioClassificationPipelinePostprocessRejectsUnsupportedOutput(t *testing.T) {
	pipeline := &AudioClassificationPipeline{TopK: 1}
	if _, err := pipeline.postprocess(&backends.PipelineBatch{OutputValues: []any{"invalid"}}); err == nil {
		t.Fatal("expected unsupported output type to fail")
	}
}

func TestAudioClassificationPipelineRejectsEmptyAudio(t *testing.T) {
	pipeline := &AudioClassificationPipeline{
		BasePipeline: &backends.BasePipeline{Model: &backends.Model{}},
	}
	if err := pipeline.preprocess(backends.NewBatch(1), [][]float32{{}}); err == nil || !strings.Contains(err.Error(), "waveform 0 is empty") {
		t.Fatalf("expected empty waveform error, got %v", err)
	}
}

func TestAudioClassificationPipelineRequiresAudioTensorBackend(t *testing.T) {
	pipeline := &AudioClassificationPipeline{
		BasePipeline: &backends.BasePipeline{Model: &backends.Model{}},
	}
	if err := pipeline.preprocess(backends.NewBatch(1), [][]float32{{0.1}}); err == nil || !strings.Contains(err.Error(), "audio tensor creation is unavailable") {
		t.Fatalf("expected explicit backend capability error, got %v", err)
	}
}
