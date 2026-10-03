package pipelines

import (
	"context"
	"image"
	"math"
	"reflect"
	"testing"

	"github.com/knights-analytics/hugot/backends"
)

func TestRawFeaturesPreserveHiddenStates(t *testing.T) {
	features := [][][]float32{{{1, 2}, {3, 4}, {5, 6}}, {{6, 7}, {8, 9}, {10, 11}}}
	output := backends.InputOutputInfo{Name: "last_hidden_state", Dimensions: backends.Shape{-1, -1, 2}}
	model := &backends.Model{OutputsMeta: []backends.InputOutputInfo{{Name: "unused"}, output}}
	batch := &backends.PipelineBatch{Size: 2, OutputValues: []any{"unused", features}}
	text := &FeatureExtractionPipeline{Output: output, OutputIndex: 1, Normalization: true}
	vision := &ImageFeatureExtractionPipeline{BasePipeline: &backends.BasePipeline{Model: model}, OutputIndex: 1, Normalization: true}
	for name, postprocess := range map[string]func(*backends.PipelineBatch) (*RawFeatureOutput, error){
		"text": text.postprocessRaw, "image": vision.postprocessRaw,
	} {
		t.Run(name, func(t *testing.T) {
			result, err := postprocess(batch)
			if err != nil {
				t.Fatal(err)
			}
			if result.OutputName != output.Name || !reflect.DeepEqual(result.Dimensions, backends.Shape{2, 3, 2}) {
				t.Fatalf("wrong output identity or dimensions: %+v", result)
			}
			if result.Embeddings != nil || !reflect.DeepEqual(result.HiddenStates, features) {
				t.Fatal("raw features must retain all tokens without pooling or normalization")
			}
			if !reflect.DeepEqual(result.GetOutput(), []any{features[0], features[1]}) {
				t.Fatal("GetOutput must retain per-input sequence dimensions")
			}
			result.HiddenStates[0][0][0] = 99
			if features[0][0][0] != 1 {
				t.Fatal("returned raw features must not alias backend output storage")
			}
		})
	}
}

func TestRawFeaturesPreserveRankTwo(t *testing.T) {
	features := [][]float32{{3, 4}, {5, 12}}
	result, err := rawFeatures(&backends.PipelineBatch{Size: 2, OutputValues: []any{features}},
		backends.InputOutputInfo{Name: "pooler_output", Dimensions: backends.Shape{-1, 2}}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if result.HiddenStates != nil || !reflect.DeepEqual(result.Embeddings, features) || !reflect.DeepEqual(result.Dimensions, backends.Shape{2, 2}) {
		t.Fatal("rank-two features must not acquire a fabricated sequence dimension")
	}
	if !reflect.DeepEqual(result.GetOutput(), []any{features[0], features[1]}) {
		t.Fatal("rank-two GetOutput must return one vector per input")
	}
	result.Embeddings[0][0] = 99
	if features[0][0] != 3 {
		t.Fatal("rank-two features must not alias backend storage")
	}
}

func TestRawFeaturesRejectMalformedOutputs(t *testing.T) {
	for name, test := range map[string]struct {
		value any
		shape backends.Shape
		size  int
	}{
		"rank one":           {[]float32{1}, backends.Shape{1}, 1},
		"rank four":          {[][][][]float32{{{{1}}}}, backends.Shape{1, 1, 1, 1}, 1},
		"integer":            {[][]int{{1}}, backends.Shape{1, 1}, 1},
		"nil tensor":         {nil, backends.Shape{1, 1}, 1},
		"empty batch":        {[][]float32{}, backends.Shape{-1, 1}, 0},
		"wrong batch":        {[][]float32{{1}, {2}}, backends.Shape{-1, 1}, 1},
		"empty vector":       {[][]float32{{}}, backends.Shape{1, -1}, 1},
		"empty sequence":     {[][][]float32{{}}, backends.Shape{1, -1, 1}, 1},
		"empty hidden":       {[][][]float32{{{}}}, backends.Shape{1, 1, -1}, 1},
		"ragged vectors":     {[][]float32{{1}, {2, 3}}, backends.Shape{-1, -1}, 2},
		"ragged tokens":      {[][][]float32{{{1}, {2, 3}}}, backends.Shape{1, -1, -1}, 1},
		"ragged sequences":   {[][][]float32{{{1}}, {{2}, {3}}}, backends.Shape{-1, -1, 1}, 2},
		"nan":                {[][]float32{{float32(math.NaN())}}, backends.Shape{1, 1}, 1},
		"positive infinity":  {[][][]float32{{{float32(math.Inf(1))}}}, backends.Shape{1, 1, 1}, 1},
		"negative infinity":  {[][]float32{{float32(math.Inf(-1))}}, backends.Shape{1, 1}, 1},
		"metadata rank":      {[][]float32{{1}}, backends.Shape{1, 1, 1}, 1},
		"metadata dimension": {[][]float32{{1}}, backends.Shape{1, 2}, 1},
		"metadata invalid":   {[][]float32{{1}}, backends.Shape{1, -2}, 1},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := rawFeatures(&backends.PipelineBatch{Size: test.size, OutputValues: []any{test.value}},
				backends.InputOutputInfo{Dimensions: test.shape}, 0)
			if err == nil {
				t.Fatal("expected malformed raw features to be rejected")
			}
		})
	}
	for _, index := range []int{-1, 0, 1} {
		if _, err := rawFeatures(&backends.PipelineBatch{Size: 1}, backends.InputOutputInfo{}, index); err == nil {
			t.Fatal("missing output must fail")
		}
	}
	if _, err := rawFeatures(nil, backends.InputOutputInfo{}, 0); err == nil {
		t.Fatal("nil batch must fail")
	}
}

func TestRawFeaturesKeepLegacyEmbeddings(t *testing.T) {
	features := [][][]float32{{{1, 2}, {3, 4}, {5, 6}}, {{6, 7}, {8, 9}, {10, 11}}}
	output := backends.InputOutputInfo{Name: "last_hidden_state", Dimensions: backends.Shape{-1, -1, 2}}
	batch := &backends.PipelineBatch{
		Size: 2, MaxSequenceLength: 3, OutputValues: []any{features},
		Input: []backends.TokenizedInput{
			{AttentionMask: []uint32{1, 1, 0}, MaxAttentionIndex: 1},
			{AttentionMask: []uint32{1, 1, 1}, MaxAttentionIndex: 2},
		},
	}
	text := &FeatureExtractionPipeline{Output: output}
	vision := &ImageFeatureExtractionPipeline{}
	for _, normalized := range []bool{false, true} {
		text.Normalization, vision.Normalization = normalized, normalized
		textResult, err := text.postprocess(batch)
		if err != nil {
			t.Fatal(err)
		}
		imageResult, err := vision.postprocess(batch)
		if err != nil {
			t.Fatal(err)
		}
		for name, test := range map[string]struct{ got, want [][]float32 }{
			"text":  {textResult.Embeddings, [][]float32{{2, 3}, {8, 9}}},
			"image": {imageResult.Embeddings, [][]float32{{3, 4}, {8, 9}}},
		} {
			for i, row := range test.want {
				denominator := float32(1)
				if normalized {
					denominator = float32(math.Sqrt(float64(row[0]*row[0] + row[1]*row[1])))
				}
				for j, value := range row {
					if math.Abs(float64(test.got[i][j]-value/denominator)) > 1e-6 {
						t.Fatalf("%s legacy embedding changed: %v", name, test.got)
					}
				}
			}
		}
		raw, err := text.postprocessRaw(batch)
		if err != nil || !reflect.DeepEqual(raw.HiddenStates, features) {
			t.Fatal("raw output changed after legacy pooling/normalization")
		}
	}
}

func TestRawFeaturesModelPoolerSelection(t *testing.T) {
	model := &backends.Model{
		InputsMeta: []backends.InputOutputInfo{{Name: "pixel_values", Dimensions: backends.Shape{-1, 3, 16, 16}}},
		OutputsMeta: []backends.InputOutputInfo{
			{Name: "last_hidden_state", Dimensions: backends.Shape{-1, -1, 2}},
			{Name: "pooler_output", Dimensions: backends.Shape{-1, 2}},
		},
	}
	ctx := context.Background()
	vision, err := NewImageFeatureExtractionPipeline(ctx, backends.PipelineConfig[*ImageFeatureExtractionPipeline]{}, model)
	if err != nil {
		t.Fatal(err)
	}
	pooler, err := NewImageFeatureExtractionPipeline(ctx, backends.PipelineConfig[*ImageFeatureExtractionPipeline]{
		Options: []backends.PipelineOption[*ImageFeatureExtractionPipeline]{WithImageModelPooler(), WithImageEmbeddingNormalization()},
	}, model)
	if err != nil {
		t.Fatal(err)
	}
	text, err := NewFeatureExtractionPipeline(ctx, backends.PipelineConfig[*FeatureExtractionPipeline]{
		Options: []backends.PipelineOption[*FeatureExtractionPipeline]{WithImageMode(), WithModelPooler()},
	}, model)
	if err != nil {
		t.Fatal(err)
	}
	if vision.OutputIndex != 0 || pooler.OutputIndex != 1 || text.OutputIndex != 1 {
		t.Fatal("explicit pooler selection must differ from default hidden-state mean pooling")
	}
	batch := &backends.PipelineBatch{Size: 1, OutputValues: []any{[][][]float32{{{1, 2}, {3, 4}}}, [][]float32{{3, 4}}}}
	mean, err := vision.postprocess(batch)
	if err != nil {
		t.Fatal(err)
	}
	pooled, err := pooler.postprocess(batch)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(mean.Embeddings, [][]float32{{2, 3}}) || !reflect.DeepEqual(pooled.Embeddings, [][]float32{{0.6, 0.8}}) {
		t.Fatalf("mean and normalized model pooler results are distinct: %v, %v", mean, pooled)
	}
	raw, err := pooler.postprocessRaw(batch)
	if err != nil || !reflect.DeepEqual(raw.Embeddings, [][]float32{{3, 4}}) {
		t.Fatal("raw model pooler must remain unnormalized")
	}
	text.Normalization = true
	textPooled, err := text.postprocess(batch)
	if err != nil || !reflect.DeepEqual(textPooled.Embeddings, [][]float32{{0.6, 0.8}}) {
		t.Fatal("text model pooler must preserve legacy normalization")
	}
	textRaw, err := text.postprocessRaw(batch)
	if err != nil || !reflect.DeepEqual(textRaw.Embeddings, [][]float32{{3, 4}}) {
		t.Fatal("text model pooler normalization must not modify the raw tensor")
	}
	for _, outputs := range [][]backends.InputOutputInfo{model.OutputsMeta[:1], {{Name: "pooler_output", Dimensions: backends.Shape{-1, -1, 2}}}} {
		model.OutputsMeta = outputs
		if _, err := NewImageFeatureExtractionPipeline(ctx, backends.PipelineConfig[*ImageFeatureExtractionPipeline]{
			Options: []backends.PipelineOption[*ImageFeatureExtractionPipeline]{WithImageModelPooler()},
		}, model); err == nil {
			t.Fatal("missing or rank-three model pooler must fail")
		}
		if _, err := NewFeatureExtractionPipeline(ctx, backends.PipelineConfig[*FeatureExtractionPipeline]{
			Options: []backends.PipelineOption[*FeatureExtractionPipeline]{WithImageMode(), WithModelPooler()},
		}, model); err == nil {
			t.Fatal("missing or rank-three text model pooler must fail")
		}
	}
}

func TestRawFeaturesRejectEmptyInputsAndWrongMode(t *testing.T) {
	ctx := context.Background()
	text := &FeatureExtractionPipeline{}
	vision := &ImageFeatureExtractionPipeline{}
	for _, run := range []func() (*RawFeatureOutput, error){
		func() (*RawFeatureOutput, error) { return text.RunRaw(ctx, nil) },
		func() (*RawFeatureOutput, error) {
			return text.RunRawWithImages(ctx, []image.Image{image.NewRGBA(image.Rect(0, 0, 1, 1))})
		},
		func() (*RawFeatureOutput, error) { return text.RunRawWithImagePaths(ctx, []string{"missing"}) },
		func() (*RawFeatureOutput, error) { return vision.RunRaw(ctx, nil) },
		func() (*RawFeatureOutput, error) { return vision.RunRawWithImages(ctx, nil) },
	} {
		if _, err := run(); err == nil {
			t.Fatal("empty or wrong-mode raw inference must fail before preprocessing")
		}
	}
	text.imageMode = true
	if _, err := text.RunRaw(ctx, []string{"text"}); err == nil {
		t.Fatal("image mode must reject text raw inference")
	}
	if _, err := text.RunRawWithImages(ctx, nil); err == nil {
		t.Fatal("empty image batch must fail")
	}
	if _, err := text.RunRawWithImagePaths(ctx, nil); err == nil {
		t.Fatal("empty image paths must fail")
	}
}
