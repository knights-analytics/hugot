package backends

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/knights-analytics/hugot/util/fileutil"
)

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
