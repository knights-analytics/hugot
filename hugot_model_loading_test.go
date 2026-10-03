package hugot

import (
	"testing"

	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/pipelines"
)

func TestResolvePipelineModelLoading(t *testing.T) {
	tests := []struct {
		name     string
		pipeline backends.Pipeline
		loading  backends.ModelLoading
		filename string
		want     backends.ModelLoading
		wantErr  bool
	}{
		{name: "ordinary default", pipeline: (*pipelines.FeatureExtractionPipeline)(nil), want: backends.ModelLoadingONNX},
		{name: "VLM default", pipeline: (*pipelines.VisualQuestionAnsweringPipeline)(nil), want: backends.ModelLoadingGenAI},
		{name: "explicit default", pipeline: (*pipelines.ImageToTextPipeline)(nil), loading: "default", want: backends.ModelLoadingGenAI},
		{name: "native VQA", pipeline: (*pipelines.VisualQuestionAnsweringPipeline)(nil), loading: backends.ModelLoadingONNX, filename: "model.onnx", want: backends.ModelLoadingONNX},
		{name: "native caption", pipeline: (*pipelines.ImageToTextPipeline)(nil), loading: backends.ModelLoadingONNX, filename: "encoder_model.onnx", want: backends.ModelLoadingONNX},
		{name: "classification GenAI", pipeline: (*pipelines.FeatureExtractionPipeline)(nil), loading: backends.ModelLoadingGenAI, wantErr: true},
		{name: "text generation ONNX", pipeline: (*pipelines.TextGenerationPipeline)(nil), loading: backends.ModelLoadingONNX, wantErr: true},
		{name: "GenAI filename", pipeline: (*pipelines.VisualQuestionAnsweringPipeline)(nil), filename: "model.onnx", wantErr: true},
		{name: "unknown mode", pipeline: (*pipelines.VisualQuestionAnsweringPipeline)(nil), loading: "unknown", wantErr: true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := resolvePipelineModelLoading(tc.pipeline, tc.loading, tc.filename)
			if (err != nil) != tc.wantErr || !tc.wantErr && got != tc.want {
				t.Fatalf("got (%q, %v), want (%q, error=%v)", got, err, tc.want, tc.wantErr)
			}
		})
	}
}

func TestPipelineModelCacheIdentity(t *testing.T) {
	onnx := pipelineModelCacheID("model", "", backends.ModelLoadingONNX)
	genai := pipelineModelCacheID("model", "", backends.ModelLoadingGenAI)
	if onnx == genai {
		t.Fatal("ordinary graph and GenAI models share an identity")
	}
	if pipelineModelCacheID("a:b", "c", backends.ModelLoadingONNX) == pipelineModelCacheID("a", "b:c", backends.ModelLoadingONNX) {
		t.Fatal("path and filename delimiters collide")
	}
	for _, requested := range []backends.ModelLoading{backends.ModelLoadingDefault, "default", backends.ModelLoadingONNX} {
		resolved, err := resolvePipelineModelLoading((*pipelines.FeatureExtractionPipeline)(nil), requested, "")
		if err != nil || pipelineModelCacheID("model", "", resolved) != onnx {
			t.Fatal("equivalent loading modes do not share an identity")
		}
	}
}
