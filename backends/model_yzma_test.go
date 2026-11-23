//go:build YZMA || ALL

package backends

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/knights-analytics/hugot/util/fileutil"
)

func TestFindGGUFModel(t *testing.T) {
	root := t.TempDir()
	model := filepath.Join(root, "model-q4.gguf")
	if err := os.WriteFile(model, []byte("fixture"), 0o600); err != nil {
		t.Fatal(err)
	}
	ctx := fileutil.WithFileSystem(context.Background(), nil)
	if got, err := findGGUFModel(ctx, model); err != nil || got != model {
		t.Fatalf("findGGUFModel(file) = %q, %v; want %q", got, err, model)
	}
	if got, err := findGGUFModel(ctx, root); err != nil || got != model {
		t.Fatalf("findGGUFModel(directory) = %q, %v; want %q", got, err, model)
	}
	second := filepath.Join(root, "model-q8.gguf")
	if err := os.WriteFile(second, []byte("fixture"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := findGGUFModel(ctx, root); err == nil || !strings.Contains(err.Error(), "multiple model GGUF") {
		t.Fatalf("expected ambiguous GGUF error, got %v", err)
	}
}

func TestFindYZMAModelFilesPairsProjector(t *testing.T) {
	root := t.TempDir()
	model := filepath.Join(root, "vision-model-q4.gguf")
	projector := filepath.Join(root, "mmproj-model-f16.gguf")
	for _, file := range []string{model, projector} {
		if err := os.WriteFile(file, []byte("fixture"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	ctx := fileutil.WithFileSystem(context.Background(), nil)
	gotModel, gotProjector, err := findYZMAModelFiles(ctx, root, nil)
	if err != nil || gotModel != model || gotProjector != projector {
		t.Fatalf("findYZMAModelFiles() = %q, %q, %v; want %q, %q", gotModel, gotProjector, err, model, projector)
	}
}
