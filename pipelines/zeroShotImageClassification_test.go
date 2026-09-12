package pipelines

import (
	"reflect"
	"testing"

	"github.com/knights-analytics/hugot/backends"
)

func TestZeroShotImageClassificationValidationFindsImageInputAfterTextInputs(t *testing.T) {
	model := &backends.Model{
		InputsMeta: []backends.InputOutputInfo{
			{Name: "input_ids", Dimensions: backends.NewShape(-1, -1)},
			{Name: "pixel_values", Dimensions: backends.NewShape(-1, 3, -1, -1)},
		},
		Tokenizer: &backends.Tokenizer{},
	}
	pipeline := &ZeroShotImageClassificationPipeline{
		BasePipeline: backends.NewBasePipeline(t.Context(), backends.PipelineConfig[*ZeroShotImageClassificationPipeline]{}, model),
		Labels:       []string{"cat"},
	}

	if err := pipeline.Validate(); err != nil {
		t.Fatalf("expected validation to find the image input after text inputs, got: %v", err)
	}
}

func TestZeroShotImageScores(t *testing.T) {
	tests := []struct {
		name     string
		output   any
		expected int
		want     []float32
		wantErr  bool
	}{
		{name: "pairwise diagonal", output: [][]float32{{0.1, 0.9, 0.2}, {0.8, 0.3, 0.1}, {0.2, 0.4, 0.7}}, expected: 3, want: []float32{0.1, 0.3, 0.7}},
		{name: "single score column", output: [][]float32{{0.2}, {0.4}}, expected: 2, want: []float32{0.2, 0.4}},
		{name: "score vector", output: []float32{0.6, 0.3}, expected: 2, want: []float32{0.6, 0.3}},
		{name: "ambiguous shape", output: [][]float32{{0.1, 0.9}, {0.8, 0.2}}, expected: 3, wantErr: true},
		{name: "inconsistent rows", output: [][]float32{{0.1, 0.9}, {0.8}}, expected: 2, wantErr: true},
		{name: "wrong batch size", output: []float32{0.1}, expected: 2, wantErr: true},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, err := zeroShotImageScores(test.output, test.expected)
			if (err != nil) != test.wantErr {
				t.Fatalf("zeroShotImageScores() error = %v, wantErr %v", err, test.wantErr)
			}
			if !test.wantErr && !reflect.DeepEqual(got, test.want) {
				t.Fatalf("zeroShotImageScores() = %v, want %v", got, test.want)
			}
		})
	}
}

func TestZeroShotImageClassificationUsesConfiguredImageFormat(t *testing.T) {
	pipeline := &ZeroShotImageClassificationPipeline{imageFormat: "NHWC"}
	format, err := pipeline.resolveImageFormat()
	if err != nil {
		t.Fatal(err)
	}
	if format != "NHWC" {
		t.Fatalf("got image format %q, want NHWC", format)
	}
}
