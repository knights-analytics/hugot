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

	model := &Model{ModelMetadata: ModelMetadata{Path: modelPath}}
	ctx := fileutil.WithFileSystem(context.Background(), nil)
	if err := loadModelConfig(ctx, model); err != nil {
		t.Fatal(err)
	}
	if model.ImageSize != 384 {
		t.Fatalf("unexpected model image size: got %d, want 384", model.ImageSize)
	}
}