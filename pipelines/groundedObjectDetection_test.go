package pipelines

import (
	"context"
	"math"
	"testing"

	"github.com/knights-analytics/hugot/backends"
)

func TestGroundedCanvas(t *testing.T) {
	cases := []struct {
		name          string
		width, height int
		wantW, wantH  int
	}{
		{name: "four by three", width: 4000, height: 3000, wantW: 1066, wantH: 800},
		{name: "portrait", width: 3000, height: 4000, wantW: 800, wantH: 1066},
		{name: "longest edge capped", width: 1000, height: 100, wantW: 1333, wantH: 133},
		{name: "capped short edge rounds up", width: 1500, height: 200, wantW: 1333, wantH: 178},
		{name: "half rounds to even", width: 2, height: 1, wantW: 1333, wantH: 666},
		{name: "already the capped short edge", width: 178, height: 1330, wantW: 178, wantH: 1330},
		{name: "square", width: 100, height: 100, wantW: 800, wantH: 800},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			gotW, gotH, err := groundedCanvas(tc.width, tc.height)
			if err != nil {
				t.Fatal(err)
			}
			if gotW != tc.wantW || gotH != tc.wantH {
				t.Fatalf("canvas %d x %d, want %d x %d", gotW, gotH, tc.wantW, tc.wantH)
			}
		})
	}
	t.Run("empty", func(t *testing.T) {
		if _, _, err := groundedCanvas(0, 10); err == nil {
			t.Fatal("expected an error")
		}
	})
}

func TestGroundedDetections(t *testing.T) {
	logits := [][]float32{
		{20, -20, 20, -20},
		{-20, -20, -20, -20},
	}
	boxes := [][]float32{
		{0.5, 0.5, 0.2, 0.2},
		{0.1, 0.1, 0.1, 0.1},
	}
	got, err := groundedDetections(logits, boxes, []int64{101, 10, 11, 102}, 100, 50, 0.25, 0.25, func(ids []int64) (string, error) {
		if len(ids) != 1 || ids[0] != 11 {
			t.Fatalf("phrase tokens %v", ids)
		}
		return "tank", nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 {
		t.Fatalf("detections %d", len(got))
	}
	det := got[0]
	if det.Label != "tank" {
		t.Fatalf("label %q", det.Label)
	}
	if math.Abs(float64(det.Score)-float64(groundedSigmoid(20))) > 1e-5 {
		t.Fatalf("score %g", det.Score)
	}
	want := [4]float32{40, 20, 60, 30}
	for i := range want {
		if math.Abs(float64(det.Box[i]-want[i])) > 1e-4 {
			t.Fatalf("box %v", det.Box)
		}
	}

	t.Run("threshold is exclusive", func(t *testing.T) {
		got, err := groundedDetections([][]float32{{0, -20, -20, -20}}, boxes[:1], []int64{101, 10, 11, 102}, 100, 50, 0.5, 0.25, func([]int64) (string, error) {
			return "", nil
		})
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != 0 {
			t.Fatalf("detections %d", len(got))
		}
	})
	t.Run("class axis start is not a label", func(t *testing.T) {
		got, err := groundedDetections([][]float32{{20, -20, -20, -20}}, boxes[:1], []int64{101, 10, 11, 102}, 100, 50, 0.25, 0.25, func(ids []int64) (string, error) {
			if len(ids) != 0 {
				t.Fatalf("phrase tokens %v", ids)
			}
			return "", nil
		})
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != 1 || got[0].Label != "" {
			t.Fatalf("detections %v", got)
		}
	})
}

func TestGroundedSpatial(t *testing.T) {
	model := &backends.Model{InputsMeta: []backends.InputOutputInfo{
		{Name: "pixel_values", Dimensions: backends.Shape{1, 3, 800, 1066}},
		{Name: "pixel_mask", Dimensions: backends.Shape{1, 800, 1066}},
	}}
	if err := groundedSpatial(model, 800, 1066); err != nil {
		t.Fatal(err)
	}
	if err := groundedSpatial(model, 800, 800); err == nil {
		t.Fatal("expected a width mismatch")
	}
	maskOnly := &backends.Model{InputsMeta: []backends.InputOutputInfo{
		{Name: "pixel_values", Dimensions: backends.Shape{1, 3, -1, -1}},
		{Name: "pixel_mask", Dimensions: backends.Shape{1, 800, 1066}},
	}}
	if err := groundedSpatial(maskOnly, 800, 800); err == nil {
		t.Fatal("expected a pixel mask mismatch")
	}
}

func TestGroundedObjectDetectionValidation(t *testing.T) {
	model := &backends.Model{
		Tokenizer: &backends.Tokenizer{},
		InputsMeta: []backends.InputOutputInfo{
			{Name: "pixel_values", Dimensions: backends.Shape{1, 3, 800, 1066}},
			{Name: "pixel_mask", Dimensions: backends.Shape{1, 800, 1066}},
			{Name: "input_ids", Dimensions: backends.Shape{1, -1}},
			{Name: "attention_mask", Dimensions: backends.Shape{1, -1}},
			{Name: "token_type_ids", Dimensions: backends.Shape{1, -1}},
		},
		OutputsMeta: []backends.InputOutputInfo{{Name: "logits"}, {Name: "pred_boxes"}},
	}
	pipeline, err := NewGroundedObjectDetectionPipeline(context.Background(), backends.PipelineConfig[*GroundedObjectDetectionPipeline]{}, model)
	if err != nil {
		t.Fatal(err)
	}
	if pipeline.IsGenerative() {
		t.Fatal("grounded object detection must not be generative")
	}
	if _, err := pipeline.Run(context.Background(), []string{"image.jpg"}); err == nil {
		t.Fatal("expected caption to be required")
	}
	model.Tokenizer = nil
	if err := pipeline.Validate(); err == nil {
		t.Fatal("expected a tokenizer")
	}
	if err := WithGroundedThresholds(-0.1, 0.25)(pipeline); err == nil {
		t.Fatal("expected a threshold rejection")
	}
}
