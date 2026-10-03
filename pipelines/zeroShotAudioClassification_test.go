package pipelines

import (
	"math"
	"testing"

	"github.com/knights-analytics/hugot/backends"
)

func TestZeroShotAudioClassificationValidateAndPostprocess(t *testing.T) {
	model := &backends.Model{
		InputsMeta: []backends.InputOutputInfo{
			{Name: "input_features", Dimensions: backends.Shape{-1, 1, 1001, 64}},
			{Name: "is_longer", Dimensions: backends.Shape{-1, 1}},
			{Name: "input_ids", Dimensions: backends.Shape{-1, -1}},
			{Name: "attention_mask", Dimensions: backends.Shape{-1, -1}},
		},
		OutputsMeta: []backends.InputOutputInfo{{Name: "logits_per_audio", Dimensions: backends.Shape{-1, -1}}}}
	p := &ZeroShotAudioClassificationPipeline{BasePipeline: &backends.BasePipeline{Model: model}, Labels: []string{"speech", "music", "noise"}, TopK: 2, SampleRate: clapSampleRate, HypothesisTemplate: "This is a sound of {}."}
	if err := p.Validate(); err != nil {
		t.Fatalf("expected valid configuration: %v", err)
	}
	out := rankZeroShotAudio(p.Labels, []float32{0.1, 0.9, 0.4}, p.TopK)
	if len(out) != 1 || len(out[0]) != 2 || out[0][0].Label != "music" || out[0][1].Label != "noise" {
		t.Fatalf("unexpected predictions: %#v", out)
	}
}

func TestZeroShotAudioClassificationAcceptsArbitraryCandidateCount(t *testing.T) {
	p := &ZeroShotAudioClassificationPipeline{BasePipeline: &backends.BasePipeline{Model: &backends.Model{
		InputsMeta: []backends.InputOutputInfo{
			{Name: "input_features", Dimensions: backends.Shape{-1, 1, 1001, 64}},
			{Name: "input_ids", Dimensions: backends.Shape{-1, -1}},
		},
		OutputsMeta: []backends.InputOutputInfo{{Dimensions: backends.Shape{-1, -1}}}}}, Labels: []string{"one", "two", "three", "four"}, TopK: 1, SampleRate: clapSampleRate, HypothesisTemplate: "This is {}."}
	if err := p.Validate(); err != nil {
		t.Fatalf("expected arbitrary candidate count to be valid: %v", err)
	}
}

func TestZeroShotAudioClassificationRejectsUnsupportedOutput(t *testing.T) {
	if _, err := zeroShotAudioSimilarity(&backends.PipelineBatch{OutputValues: []any{"invalid"}}); err == nil {
		t.Fatal("expected unsupported output type to fail")
	}
}

func TestZeroShotAudioClassificationRequiresSinglePairScore(t *testing.T) {
	if _, err := zeroShotAudioSimilarity(&backends.PipelineBatch{OutputValues: []any{[][]float32{{0.1, 0.2}}}}); err == nil {
		t.Fatal("expected non-pairwise output shape to fail")
	}
}

func TestCLAPAudioFeaturesMatchModelShapeAndAreFinite(t *testing.T) {
	samples := make([]float32, clapSampleRate/2)
	for i := range samples {
		samples[i] = float32(0.1 * math.Sin(2*math.Pi*440*float64(i)/clapSampleRate))
	}
	features, isLonger, err := clapAudioFeatures(samples, clapSampleRate)
	if err != nil {
		t.Fatal(err)
	}
	if isLonger || len(features) != 1 || len(features[0]) != clapMaxSamples/clapHopLength+1 || len(features[0][0]) != clapMelBins {
		t.Fatalf("unexpected CLAP feature shape or duration flag: %dx%dx%d, %v", len(features), len(features[0]), len(features[0][0]), isLonger)
	}
	for _, frame := range features[0] {
		for _, value := range frame {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatalf("CLAP features contain a non-finite value: %v", value)
			}
		}
	}
}
