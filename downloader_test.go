//go:build !NODOWNLOAD

package hugot

import (
	"context"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func TestProcessorAndGraphBundleArtifacts(t *testing.T) {
	names := []string{"config.json", "preprocessor_config.json", "processor_config.json", "video_processor_config.json", "generation_config.json", "variants/cpu/model.onnx", "variants/cpu/model.onnx_data", "variants/cpu/model.onnx_data_0", "variants/cpu/decoder.onnx", "variants/cpu/external/weights.bin", "variants/gpu/model.onnx"}
	opts := NewDownloadOptions()
	opts.OnnxFilePath = "variants/cpu/model.onnx"
	files, err := selectModelArtifacts(slices.Clone(names), opts)
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range names[:5] {
		if !slices.Contains(files, name) {
			t.Fatalf("missing processor artifact %s", name)
		}
	}
	if !slices.Contains(files, opts.OnnxFilePath+"_data") {
		t.Fatal("missing external data")
	}
	if !slices.Contains(files, opts.OnnxFilePath+"_data_0") {
		t.Fatal("missing external data shard")
	}
	if slices.Contains(files, "variants/cpu/decoder.onnx") {
		t.Fatal("bundle was selected without opt-in")
	}
	opts.DownloadGraphBundle = true
	files, err = selectModelArtifacts(slices.Clone(names), opts)
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range names[:len(names)-1] {
		if !slices.Contains(files, name) {
			t.Fatalf("missing bundle artifact %s", name)
		}
	}
	if slices.Contains(files, "variants/gpu/model.onnx") {
		t.Fatal("unselected graph variant was collected")
	}
	opts.AdditionalFilePaths = []string{"missing.json"}
	if _, err := selectModelArtifacts(slices.Clone(names), opts); err == nil {
		t.Fatal("missing explicit artifact was accepted")
	}
	opts.AdditionalFilePaths = nil
	opts.ExternalDataPath = "missing.weights"
	if _, err := selectModelArtifacts(slices.Clone(names), opts); err == nil {
		t.Fatal("missing explicit external data was accepted")
	}
}

func TestGGUFModelArtifacts(t *testing.T) {
	names := []string{"model-q4.gguf", "model-q8.gguf", "mmproj-model-f16.gguf", "README.md"}
	opts := NewDownloadOptions()
	opts.GGUFFilePath = "model-q4.gguf"
	opts.GGUFProjectorFilePath = "mmproj-model-f16.gguf"
	files, err := selectModelArtifacts(names, opts)
	if err != nil {
		t.Fatal(err)
	}
	if len(files) != 2 || !slices.Contains(files, opts.GGUFFilePath) || !slices.Contains(files, opts.GGUFProjectorFilePath) {
		t.Fatalf("got artifacts %v, want model %q and projector %q", files, opts.GGUFFilePath, opts.GGUFProjectorFilePath)
	}
	opts.GGUFFilePath = "missing.gguf"
	if _, err := selectModelArtifacts(names, opts); err == nil {
		t.Fatal("missing GGUF file was accepted")
	}
}

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
	options := NewDownloadOptions()
	options.GGUFFilePath = "../outside/model.gguf"
	options.MaxRetries = 0
	if _, err := DownloadModel(context.Background(), "example/model", t.TempDir(), options); err == nil || !strings.Contains(err.Error(), "path") {
		t.Fatalf("expected unsafe GGUF path error before hub access, got %v", err)
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
