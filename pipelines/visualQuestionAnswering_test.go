package pipelines

import (
	"context"
	"encoding/json"
	"errors"
	"image"
	"image/color"
	"math"
	"reflect"
	"testing"

	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/testcases/embedded"
)

func TestNativeVQAPythonProcessorReference(t *testing.T) {
	var fixture struct {
		Tolerance float64 `json:"pixel_absolute_tolerance"`
		Cases     []struct {
			Width, Height int
			Shape         []int64 `json:"pixel_values_shape"`
			MaskShape     []int64 `json:"pixel_mask_shape"`
			Probes        []struct {
				Index [4]int
				Value float64
			} `json:"pixel_probes"`
		}
	}
	if err := json.Unmarshal(embedded.ViltReferenceByte, &fixture); err != nil {
		t.Fatal(err)
	}
	native := &nativeVisualQuestionAnswering{shortestEdge: 384, sizeDivisor: 32, resample: 3, mean: [3]float32{0.5, 0.5, 0.5}, std: [3]float32{0.5, 0.5, 0.5}}
	for _, tc := range fixture.Cases {
		img := image.NewRGBA(image.Rect(0, 0, tc.Width, tc.Height))
		for y := range tc.Height {
			for x := range tc.Width {
				img.SetRGBA(x, y, color.RGBA{R: uint8((3*x + y) % 256), G: uint8((5*y + x) % 256), B: uint8((x + 7*y) % 256), A: 255})
			}
		}
		pixels, mask, err := native.imageTensors(img)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(pixels.Shape, tc.Shape) || !reflect.DeepEqual(mask.Shape, tc.MaskShape) {
			t.Fatalf("unexpected image/mask dimensions: %v / %v", pixels.Shape, mask.Shape)
		}
		for _, value := range mask.Data.([]int64) {
			if value != 1 {
				t.Fatal("individual unpadded image has an invalid pixel mask")
			}
		}
		data := pixels.Data.([]float32)
		for _, probe := range tc.Probes {
			index := (probe.Index[1]*int(pixels.Shape[2])+probe.Index[2])*int(pixels.Shape[3]) + probe.Index[3]
			if math.Abs(float64(data[index])-probe.Value) > fixture.Tolerance {
				t.Errorf("pixel %v got %g, want %g", probe.Index, data[index], probe.Value)
			}
		}
	}
}

func TestNativeVQARanking(t *testing.T) {
	labels := map[int]string{0: "cat", 1: "dog", 2: "bird"}
	logits := backends.Tensor{Shape: []int64{1, 3}, Data: []float32{2, -1, 0}}
	answers, err := rankVisualAnswers(logits, labels, 2)
	if err != nil || len(answers) != 2 || answers[0].Answer != "cat" || answers[1].Answer != "bird" || answers[1].Score != 0.5 {
		t.Fatalf("unexpected ranked answers: %v, %v", answers, err)
	}
	all, err := rankVisualAnswers(logits, labels, 10)
	if err != nil || len(all) != 3 {
		t.Fatalf("top_k is not capped to answer count: %v, %v", all, err)
	}
	for _, bad := range []backends.Tensor{
		{Shape: []int64{1, 3}, Data: []float32{1, 2}},
		{Shape: []int64{3}, Data: []float32{1, 2, 3}},
		{Shape: []int64{1, 3}, Data: []float32{1, float32(math.NaN()), 3}},
		{Shape: []int64{1, 3}, Data: []float32{1, 2, float32(math.Inf(1))}},
	} {
		if _, err := rankVisualAnswers(bad, labels, 2); err == nil {
			t.Fatal("invalid classification tensor was accepted")
		}
	}
	if _, err := rankVisualAnswers(logits, map[int]string{0: "cat", 1: "dog"}, 2); err == nil {
		t.Fatal("incomplete label map was accepted")
	}
}

func TestNativeVQARejectsChatAndInvalidInputs(t *testing.T) {
	model := &backends.Model{
		IDLabelMap: map[int]string{0: "cat"}, MaxPositionEmbeddings: 40,
		InputsMeta:  []backends.InputOutputInfo{{Name: "input_ids"}, {Name: "attention_mask"}, {Name: "token_type_ids"}, {Name: "pixel_values"}, {Name: "pixel_mask"}},
		OutputsMeta: []backends.InputOutputInfo{{Name: "logits"}},
	}
	p := &VisualQuestionAnsweringPipeline{questionAnsweringFamily: questionAnsweringFamily{&multimodalGeneration{
		BasePipeline: &backends.BasePipeline{Model: model}, MaxLength: 40,
	}}, TopK: 5, native: &nativeVisualQuestionAnswering{}}
	if err := p.Validate(); err != nil {
		t.Fatal(err)
	}
	p.Streaming = true
	if err := p.Validate(); err == nil {
		t.Fatal("native classification accepted streaming")
	}
	p.Streaming = false
	if _, err := p.RunPipeline(t.Context(), nil); err == nil {
		t.Fatal("empty input accepted")
	}
	if _, err := p.RunPipeline(t.Context(), []VisualQuestionAnsweringInput{{ImagePath: "unused", Question: "q", Role: "user"}}); err == nil {
		t.Fatal("native classification accepted conversational roles")
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := p.RunPipeline(ctx, []VisualQuestionAnsweringInput{{ImagePath: "unused", Question: "q"}}); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation not propagated: %v", err)
	}
	output := &QuestionAnsweringTextOutput{Responses: []string{"cat"}, Answers: [][]VisualQuestionAnsweringAnswer{{{Answer: "cat", Score: 0.9}}}}
	if output.GetOutput()[0] != "cat" {
		t.Fatal("legacy GetOutput behavior changed")
	}
}
