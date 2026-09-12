//go:build !NODOWNLOAD

package hugot

import (
	"context"
	"path/filepath"
	"strings"
	"testing"
)

func TestDownloadModelRejectsUnsafeAdditionalPathsBeforeHubAccess(t *testing.T) {
	for _, name := range []string{"../outside", "sub/../../outside", "/absolute", `C:\outside`, `sub\..\outside`, "sub/../inside"} {
		t.Run(name, func(t *testing.T) {
			options := NewDownloadOptions()
			options.PreservePaths = true
			options.AdditionalFilePaths = []string{name}
			options.MaxRetries = 0 // An invalid path must fail before any hub operation.
			_, err := DownloadModel(context.Background(), "example/model", t.TempDir(), options)
			if err == nil || !strings.Contains(err.Error(), "path") || strings.Contains(err.Error(), "after 0 attempts") {
				t.Fatalf("expected unsafe path error for %q, got %v", name, err)
			}
		})
	}
}

func TestPreservedDownloadPath(t *testing.T) {
	root := t.TempDir()
	for _, name := range []string{"model.onnx", "onnx/model.onnx", "cpu_and_mobile/config.json"} {
		t.Run(name, func(t *testing.T) {
			got, err := preservedDownloadPath(root, name)
			if err != nil {
				t.Fatal(err)
			}
			want := filepath.Join(root, filepath.FromSlash(name))
			if got != want {
				t.Fatalf("got %q, want %q", got, want)
			}
		})
	}
	for _, name := range []string{"", ".", "..", "../outside", "a/../../outside", "a/../inside", "/absolute", `C:\outside`, `a\..\outside`, `\\server\share\file`} {
		t.Run(name, func(t *testing.T) {
			if got, err := preservedDownloadPath(root, name); err == nil {
				t.Fatalf("accepted unsafe path %q as %q", name, got)
			}
		})
	}
}