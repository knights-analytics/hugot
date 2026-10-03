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

func testCaptionProcessor(t *testing.T) captionProcessor {
	t.Helper()
	var p captionProcessor
	err := json.Unmarshal([]byte(`{"size":{"height":224,"width":224},"do_resize":true,"do_rescale":true,"do_normalize":true,"resample":2,"rescale_factor":0.0039215686,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5]}`), &p)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func TestCaptionProcessorRGB(t *testing.T) {
	p := testCaptionProcessor(t)
	img := image.NewNRGBA(image.Rect(3, 5, 5, 7))
	for y := 5; y < 7; y++ {
		for x := 3; x < 5; x++ {
			img.SetNRGBA(x, y, color.NRGBA{R: 255, G: 128, B: 0, A: 255})
		}
	}
	tensor, err := p.pixels(img)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(tensor.Shape, []int64{1, 3, 224, 224}) {
		t.Fatalf("shape: %v", tensor.Shape)
	}
	data := tensor.Data.([]float32)
	for c, want := range []float32{1, 128.0/255*2 - 1, -1} {
		for _, got := range data[c*224*224 : (c+1)*224*224] {
			if math.Abs(float64(got-want)) > 1e-6 {
				t.Fatalf("channel %d: got %v, want %v", c, got, want)
			}
		}
	}
	for _, img := range []image.Image{nil, image.NewRGBA(image.Rectangle{})} {
		if _, err := p.pixels(img); err == nil {
			t.Fatal("accepted empty image")
		}
	}
}

func TestCaptionProcessorRejectsUnsupported(t *testing.T) {
	for _, mutate := range []func(*captionProcessor){
		func(p *captionProcessor) { p.Resize = false },
		func(p *captionProcessor) { p.Rescale = false },
		func(p *captionProcessor) { p.Normalize = false },
		func(p *captionProcessor) { p.Resample = 3 },
		func(p *captionProcessor) { p.Size.Width = 384 },
		func(p *captionProcessor) { p.CenterCrop = true },
		func(p *captionProcessor) { p.Factor = math.NaN() },
		func(p *captionProcessor) { p.Mean[0] = 0 },
		func(p *captionProcessor) { p.Std = nil },
	} {
		p := testCaptionProcessor(t)
		mutate(&p)
		if err := p.validate(); err == nil {
			t.Fatal("accepted unsupported processor")
		}
	}
}

func TestCaptionProcessorPythonReference(t *testing.T) {
	var reference struct {
		SchemaVersion int `json:"schema_version"`
		Model         struct {
			ID       string `json:"id"`
			Revision string `json:"revision"`
		} `json:"model"`
		Processor captionProcessor `json:"processor"`
		Cases     []struct {
			Name  string `json:"name"`
			Input struct {
				Kind   string  `json:"kind"`
				Width  int     `json:"width"`
				Height int     `json:"height"`
				RGB    [3]byte `json:"rgb"`
			} `json:"input"`
			Shape     []int64 `json:"output_shape"`
			Tolerance float64 `json:"absolute_tolerance"`
			Samples   []struct {
				X      int        `json:"x"`
				Y      int        `json:"y"`
				Values [3]float32 `json:"values"`
			} `json:"samples"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(embedded.CaptionReferenceByte, &reference); err != nil {
		t.Fatal(err)
	}
	if reference.SchemaVersion != 1 || reference.Model.ID != "Xenova/vit-gpt2-image-captioning" || reference.Model.Revision != "215b4edcb7ec1fad5905a18a03f7b2007f6fabd0" || len(reference.Cases) != 4 {
		t.Fatal("unexpected caption reference contract")
	}
	for _, tc := range reference.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			if tc.Input.Width <= 0 || tc.Input.Height <= 0 || len(tc.Samples) == 0 || math.IsNaN(tc.Tolerance) || tc.Tolerance <= 0 || tc.Tolerance > 2.0/255+1e-6 {
				t.Fatal("invalid caption reference case")
			}
			img := image.NewNRGBA(image.Rect(3, 5, 3+tc.Input.Width, 5+tc.Input.Height))
			for y := range tc.Input.Height {
				for x := range tc.Input.Width {
					var rgb [3]byte
					for c := range 3 {
						switch tc.Input.Kind {
						case "constant":
							rgb[c] = tc.Input.RGB[c]
						case "spatial":
							rgb[c] = byte((17*x + 29*y + 43*c + 7*((x*y)%19)) % 256)
						default:
							t.Fatalf("unknown input kind %q", tc.Input.Kind)
						}
					}
					img.SetNRGBA(3+x, 5+y, color.NRGBA{R: rgb[0], G: rgb[1], B: rgb[2], A: 255})
				}
			}
			tensor, err := reference.Processor.pixels(img)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(tensor.Shape, tc.Shape) {
				t.Fatalf("shape: got %v, want %v", tensor.Shape, tc.Shape)
			}
			data := tensor.Data.([]float32)
			for _, sample := range tc.Samples {
				if sample.X < 0 || sample.X >= 224 || sample.Y < 0 || sample.Y >= 224 {
					t.Fatal("reference sample is out of bounds")
				}
				for c, want := range sample.Values {
					got := data[c*224*224+sample.Y*224+sample.X]
					if math.IsNaN(float64(got)) || math.IsInf(float64(got), 0) || math.IsNaN(float64(want)) || math.IsInf(float64(want), 0) || math.Abs(float64(got)-float64(want)) > tc.Tolerance {
						t.Errorf("pixel (%d,%d) channel %d: got %v, Python %v, tolerance %v", sample.X, sample.Y, c, got, want, tc.Tolerance)
					}
				}
			}
		})
	}
}

func TestCaptionInputValidation(t *testing.T) {
	for _, input := range []ImageTextPrompt{
		{}, {ImagePath: " "}, {ImagePath: "image.png", Prompt: "describe"},
		{ImagePath: "image.png", Role: "user"},
		{ImagePath: "image.png", History: []backends.Message{{Role: "user", Content: "hi"}}},
	} {
		if err := validateCaptionInput(input); err == nil {
			t.Fatalf("accepted %+v", input)
		}
	}
	if err := validateCaptionInput(ImageTextPrompt{ImagePath: "image.png"}); err != nil {
		t.Fatal(err)
	}
}

func TestCaptionGraphRejectsPastInputs(t *testing.T) {
	model := &backends.Model{
		InputsMeta:  []backends.InputOutputInfo{{Name: "input_ids"}, {Name: "encoder_hidden_states"}},
		OutputsMeta: []backends.InputOutputInfo{{Name: "logits"}, {Name: "present.0.key"}},
	}
	if err := captionGraphContract(model, []string{"input_ids", "encoder_hidden_states"}, []string{"logits"}); err != nil {
		t.Fatal(err)
	}
	model.InputsMeta = append(model.InputsMeta, backends.InputOutputInfo{Name: "past_key_values.0.key"})
	if err := captionGraphContract(model, []string{"input_ids", "encoder_hidden_states"}, []string{"logits"}); err == nil {
		t.Fatal("accepted past inputs")
	}
	model.InputsMeta = []backends.InputOutputInfo{{Name: "input_ids"}, {Name: "past_key_values.0.key"}}
	if err := captionGraphContract(model, []string{"input_ids", "encoder_hidden_states"}, []string{"logits"}); err == nil {
		t.Fatal("accepted wrong input names")
	}
}

func TestGenerateCaptionTokens(t *testing.T) {
	for _, tc := range []struct {
		name     string
		sequence []int
		limit    int
		want     []uint32
		calls    int
	}{
		{"immediate EOS", []int{3}, 4, []uint32{}, 1},
		{"EOS after tokens", []int{1, 2, 3}, 4, []uint32{1, 2}, 3},
		{"token limit", []int{1, 2, 1}, 2, []uint32{1, 2}, 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			calls := 0
			embeddings := backends.Tensor{Shape: []int64{1, 1, 1}, Data: []float32{0.25}}
			infer := func(_ context.Context, inputs map[string]backends.Tensor) (map[string]backends.Tensor, error) {
				ids := inputs["input_ids"].Data.([]int64)
				want := []int64{0}
				for _, token := range tc.sequence[:calls] {
					want = append(want, int64(token))
				}
				if !reflect.DeepEqual(ids, want) || !reflect.DeepEqual(inputs["input_ids"].Shape, []int64{1, int64(len(want))}) {
					t.Fatalf("decoder prefix: %v", inputs["input_ids"])
				}
				if !reflect.DeepEqual(inputs["encoder_hidden_states"], embeddings) {
					t.Fatal("encoder state changed")
				}
				data := make([]float32, len(ids)*4)
				data[len(data)-4+tc.sequence[calls]] = 1
				calls++
				return map[string]backends.Tensor{"logits": {Shape: []int64{1, int64(len(ids)), 4}, Data: data}}, nil
			}
			got, err := generateCaptionTokens(context.Background(), embeddings, 0, 3, tc.limit, infer)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, tc.want) || calls != tc.calls {
				t.Fatalf("tokens %v, calls %d", got, calls)
			}
		})
	}
}

func TestGenerateCaptionErrors(t *testing.T) {
	for _, data := range [][]float32{{float32(math.NaN()), 1}, {0, float32(math.Inf(1))}} {
		_, err := generateCaptionTokens(context.Background(), backends.Tensor{}, 0, 1, 2, func(context.Context, map[string]backends.Tensor) (map[string]backends.Tensor, error) {
			return map[string]backends.Tensor{"logits": {Shape: []int64{1, 1, 2}, Data: data}}, nil
		})
		if err == nil {
			t.Fatal("accepted nonfinite logits")
		}
	}
	sentinel := errors.New("inference failed")
	_, err := generateCaptionTokens(context.Background(), backends.Tensor{}, 0, 1, 2, func(context.Context, map[string]backends.Tensor) (map[string]backends.Tensor, error) {
		return nil, sentinel
	})
	if !errors.Is(err, sentinel) {
		t.Fatalf("inference error: %v", err)
	}
	_, err = generateCaptionTokens(context.Background(), backends.Tensor{}, 0, 1, 2, func(context.Context, map[string]backends.Tensor) (map[string]backends.Tensor, error) { return nil, nil })
	if err == nil {
		t.Fatal("accepted missing logits")
	}
}

func TestGenerateCaptionCancellation(t *testing.T) {
	for _, before := range []bool{true, false} {
		ctx, cancel := context.WithCancel(context.Background())
		if before {
			cancel()
		}
		calls := 0
		_, err := generateCaptionTokens(ctx, backends.Tensor{}, 0, 1, 2, func(context.Context, map[string]backends.Tensor) (map[string]backends.Tensor, error) {
			calls++
			cancel()
			return map[string]backends.Tensor{"logits": {Shape: []int64{1, 1, 2}, Data: []float32{0, 1}}}, nil
		})
		cancel()
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("cancellation: %v", err)
		}
		want := 1
		if before {
			want = 0
		}
		if calls != want {
			t.Fatalf("calls %d, want %d", calls, want)
		}
	}
}
