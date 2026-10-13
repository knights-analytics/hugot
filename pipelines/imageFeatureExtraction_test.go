package pipelines

import (
	"testing"

	"github.com/knights-analytics/hugot/backends"
)

func TestImageFeatureExtractionPipelineValidate(t *testing.T) {
	model := &backends.Model{
		InputsMeta: []backends.InputOutputInfo{{Name: "pixel_values", Dimensions: backends.Shape{-1, 3, 224, 224}}}}
	pipeline := &ImageFeatureExtractionPipeline{BasePipeline: &backends.BasePipeline{Model: model}}
	if err := pipeline.Validate(); err == nil {
		t.Fatal("expected missing image output metadata to fail")
	}
}

func TestImageFeatureCLSToken(t *testing.T) {
	model := &backends.Model{
		InputsMeta:  []backends.InputOutputInfo{{Name: "pixel_values", Dimensions: backends.Shape{-1, 3, 16, 16}}},
		OutputsMeta: []backends.InputOutputInfo{{Name: "last_hidden_state", Dimensions: backends.Shape{-1, -1, 2}}},
	}
	cls, err := NewImageFeatureExtractionPipeline(t.Context(), backends.PipelineConfig[*ImageFeatureExtractionPipeline]{
		Options: []backends.PipelineOption[*ImageFeatureExtractionPipeline]{WithImageCLSToken()},
	}, model)
	if err != nil {
		t.Fatal(err)
	}
	batch := &backends.PipelineBatch{OutputValues: []any{[][][]float32{{{1, 2}, {3, 4}}}}}
	got, err := cls.postprocess(batch)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Embeddings) != 1 || len(got.Embeddings[0]) != 2 || got.Embeddings[0][0] != 1 || got.Embeddings[0][1] != 2 {
		t.Fatalf("cls token %v", got.Embeddings)
	}
	mean, err := NewImageFeatureExtractionPipeline(t.Context(), backends.PipelineConfig[*ImageFeatureExtractionPipeline]{}, model)
	if err != nil {
		t.Fatal(err)
	}
	pooled, err := mean.postprocess(batch)
	if err != nil {
		t.Fatal(err)
	}
	if len(pooled.Embeddings) != 1 || pooled.Embeddings[0][0] != 2 || pooled.Embeddings[0][1] != 3 {
		t.Fatalf("mean pool %v", pooled.Embeddings)
	}
	if _, err := cls.postprocess(&backends.PipelineBatch{OutputValues: []any{[][]float32{{1, 2}}}}); err == nil {
		t.Fatal("cls token requires a token sequence")
	}
	if _, err := NewImageFeatureExtractionPipeline(t.Context(), backends.PipelineConfig[*ImageFeatureExtractionPipeline]{
		Options: []backends.PipelineOption[*ImageFeatureExtractionPipeline]{WithImageCLSToken(), WithImageModelPooler()},
	}, &backends.Model{
		InputsMeta: model.InputsMeta,
		OutputsMeta: []backends.InputOutputInfo{
			{Name: "last_hidden_state", Dimensions: backends.Shape{-1, -1, 2}},
			{Name: "pooler_output", Dimensions: backends.Shape{-1, 2}},
		},
	}); err == nil {
		t.Fatal("cls token and model pooler must not combine")
	}
}

func TestImageFeatureExtractionPipelinePostprocessRejectsUnsupportedOutput(t *testing.T) {
	pipeline := &ImageFeatureExtractionPipeline{OutputIndex: 0}
	_, err := pipeline.postprocess(&backends.PipelineBatch{OutputValues: []any{"invalid"}})
	if err == nil {
		t.Fatal("expected unsupported output type to fail")
	}
}
