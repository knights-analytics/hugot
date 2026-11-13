package backends

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/util/fileutil"
)

func TestMultimodalSessionRejectsTextEngine(t *testing.T) {
	model := &Model{IsGenerative: true, loadOptions: options.Defaults()}
	if err := model.ValidateMultimodalSession(); err != nil {
		t.Fatal(err)
	}
	model.loadOptions.ORTOptions.UseEngine = true
	if err := model.ValidateMultimodalSession(); err == nil {
		t.Fatal("text engine accepted multimodal generation")
	}
	model.IsGenerative = false
	if err := model.ValidateMultimodalSession(); err == nil {
		t.Fatal("ordinary ONNX graph accepted multimodal generation")
	}
}

func TestLoadModelConfigReadsImageSize(t *testing.T) {
	modelPath := t.TempDir()
	configPath := filepath.Join(modelPath, "config.json")
	if err := os.WriteFile(configPath, []byte(`{"image_size":384}`), 0o600); err != nil {
		t.Fatal(err)
	}

	model := &Model{Path: modelPath}
	ctx := fileutil.WithFileSystem(context.Background(), nil)
	if err := loadModelConfig(ctx, model); err != nil {
		t.Fatal(err)
	}
	if model.ImageSize != 384 {
		t.Fatalf("unexpected model image size: got %d, want 384", model.ImageSize)
	}
}

func TestLoadModelConfigReadsVisionImageSize(t *testing.T) {
	modelPath := t.TempDir()
	configPath := filepath.Join(modelPath, "config.json")
	if err := os.WriteFile(configPath, []byte(`{"vision_config":{"image_size":960}}`), 0o600); err != nil {
		t.Fatal(err)
	}

	model := &Model{Path: modelPath}
	ctx := fileutil.WithFileSystem(context.Background(), nil)
	if err := loadModelConfig(ctx, model); err != nil {
		t.Fatal(err)
	}
	if model.ImageSize != 960 {
		t.Fatalf("unexpected model image size: got %d, want 960", model.ImageSize)
	}
}

func TestTextTensorBatchCounts(t *testing.T) {
	tests := []struct {
		name              string
		textRank          int
		batchSize         int
		queryCount        int
		hasQueryDimension bool
		wantBatchSize     int
		wantQueryCount    int
	}{
		{name: "rank 2 batch", textRank: 2, batchSize: 4, queryCount: 1, wantBatchSize: 4, wantQueryCount: 1},
		{name: "rank 2 labels", textRank: 2, batchSize: 1, queryCount: 2, hasQueryDimension: true, wantBatchSize: 2, wantQueryCount: 1},
		{name: "rank 3 labels", textRank: 3, batchSize: 1, queryCount: 2, hasQueryDimension: true, wantBatchSize: 1, wantQueryCount: 2},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			batchSize, queryCount := textTensorBatchCounts(test.textRank, test.batchSize, test.queryCount, test.hasQueryDimension)
			if batchSize != test.wantBatchSize || queryCount != test.wantQueryCount {
				t.Fatalf("unexpected text tensor counts: got (%d, %d), want (%d, %d)", batchSize, queryCount, test.wantBatchSize, test.wantQueryCount)
			}
		})
	}
}

func TestGoTokenizerOptionsSkipMediaInputs(t *testing.T) {
	model := &Model{InputsMeta: []InputOutputInfo{
		{Name: "input_ids"},
		{Name: "attention_mask"},
		{Name: "image_features"},
		{Name: "video_features"},
		{Name: "audio_features"},
	}}
	_, typeIDs, attentionMask, err := getGoTokenizerOptions(model)
	if err != nil {
		t.Fatal(err)
	}
	if typeIDs || !attentionMask {
		t.Fatalf("unexpected tokenizer inputs: got type IDs %v, attention mask %v; want the attention mask only", typeIDs, attentionMask)
	}
}

func TestEmptyMediaFeaturesShape(t *testing.T) {
	text := NewBatch(1)
	images := NewBatch(1)
	images.ImageValues = [][][][]float32{{{{0}}}}
	tests := []struct {
		name      string
		batch     *PipelineBatch
		input     InputOutputInfo
		wantShape Shape
		wantEmpty bool
	}{
		{name: "image features", batch: text, input: InputOutputInfo{Name: "image_features", Dimensions: NewShape(-1, 512)}, wantShape: NewShape(0, 512), wantEmpty: true},
		{name: "video features", batch: text, input: InputOutputInfo{Name: "video_features", Dimensions: NewShape(-1, 512)}, wantShape: NewShape(0, 512), wantEmpty: true},
		{name: "audio features", batch: text, input: InputOutputInfo{Name: "audio_features", Dimensions: NewShape(-1, 512)}, wantShape: NewShape(0, 512), wantEmpty: true},
		{name: "text input", batch: text, input: InputOutputInfo{Name: "input_ids", Dimensions: NewShape(-1, -1)}},
		{name: "batch with images", batch: images, input: InputOutputInfo{Name: "image_features", Dimensions: NewShape(-1, 512)}},
		{name: "pixel values", batch: text, input: InputOutputInfo{Name: "pixel_values", Dimensions: NewShape(-1, 3, 224, 224)}},
		{name: "static features", batch: text, input: InputOutputInfo{Name: "image_features", Dimensions: NewShape(1, 512)}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			shape, empty := emptyMediaFeaturesShape(test.batch, test.input)
			if empty != test.wantEmpty || (empty && !slices.Equal(shape, test.wantShape)) {
				t.Fatalf("unexpected empty media features: got (%v, %v), want (%v, %v)", shape, empty, test.wantShape, test.wantEmpty)
			}
		})
	}
}
