//go:build cgo && (ORT || ALL)

package backends

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"runtime"
	"strings"
	"testing"

	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/util/fileutil"
	"github.com/knights-analytics/ortgenai"
	"google.golang.org/protobuf/proto"
)

type inertGenAIFS struct{ fileutil.FileSystem }

type memoryGenAITestFS struct {
	fileutil.FileSystem
	files map[string][]byte
}

func (fs memoryGenAITestFS) OpenFile(ctx context.Context, filename string) (io.ReadCloser, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	data, ok := fs.files[filename]
	if !ok {
		return nil, os.ErrNotExist
	}
	return io.NopCloser(bytes.NewReader(data)), nil
}

func (fs memoryGenAITestFS) FileExists(ctx context.Context, filename string) (bool, error) {
	if err := ctx.Err(); err != nil {
		return false, err
	}
	_, ok := fs.files[filename]
	return ok, nil
}

func TestMissingExternalWeightUsesAdapterAndKeepsLocalModelIsolated(t *testing.T) {
	localModelPath := t.TempDir()
	config := []byte(`{"model":{"decoder":{"filename":"model.onnx"}}}`)
	graph, err := proto.Marshal(&protos.ModelProto{Graph: &protos.GraphProto{Initializer: []*protos.TensorProto{{
		DataLocation: protos.TensorProto_EXTERNAL,
		ExternalData: []*protos.StringStringEntryProto{{Key: "location", Value: "weights.bin"}},
	}}}})
	if err != nil {
		t.Fatal(err)
	}
	localFiles := map[string][]byte{
		"genai_config.json":     config,
		"tokenizer_config.json": []byte(`{}`),
		"tokenizer.json":        []byte(`{}`),
		"model.onnx":            graph,
		"weights.bin":           []byte("local-only weights"),
	}
	adapterFiles := map[string][]byte{}
	for name, data := range localFiles {
		if err := os.WriteFile(localModelPath+string(os.PathSeparator)+name, data, 0o600); err != nil {
			t.Fatal(err)
		}
		if name != "weights.bin" {
			adapterFiles[fileutil.PathJoinSafe(localModelPath, name)] = data
		}
	}
	ctx := fileutil.WithFileSystem(context.Background(), memoryGenAITestFS{files: adapterFiles})
	err = createORTGenerativeSession(ctx, &Model{Path: localModelPath}, &options.Options{ORTOptions: &options.OrtOptions{}})
	if err == nil || !strings.Contains(err.Error(), `reading model bundle file "weights.bin"`) || !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("createORTGenerativeSession() error = %v, want adapter-specific missing weight error", err)
	}
}

func TestCustomAdapterNeverUsesSameNamedLocalModelDirectory(t *testing.T) {
	localModelPath := t.TempDir()
	if err := os.WriteFile(localModelPath+string(os.PathSeparator)+"local-only", []byte("must not be read"), 0o600); err != nil {
		t.Fatal(err)
	}
	ctx := fileutil.WithFileSystem(context.Background(), inertGenAIFS{})
	err := createORTGenerativeSession(ctx, &Model{Path: localModelPath}, &options.Options{
		ORTOptions: &options.OrtOptions{UseEngine: true},
	})
	if err == nil || !strings.Contains(err.Error(), "in-memory GenAI loading does not support WithGenerativeEngine") {
		t.Fatalf("custom adapter loading error = %v; expected memory-path rejection before native directory loading", err)
	}
}

func TestMemoryLoadingPropagatesCancellationBeforeInitialization(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	err := createORTGenerativeMemorySession(ctx, &Model{}, &options.Options{ORTOptions: &options.OrtOptions{}})
	if err == nil || err != context.Canceled {
		t.Fatalf("createORTGenerativeMemorySession() error = %v, want context.Canceled", err)
	}
}

func TestORTGenAILibraryLookupUsesLocalOSFilesystem(t *testing.T) {
	if ortgenai.IsInitialized() {
		t.Skip("ORT GenAI environment is already initialized")
	}
	name := "libonnxruntime-genai.so"
	switch runtime.GOOS {
	case "windows":
		name = "onnxruntime-genai.dll"
	case "darwin":
		name = "libonnxruntime-genai.dylib"
	}
	libraryDir := t.TempDir()
	libraryPath := fileutil.PathJoinSafe(libraryDir, name)
	ctx := fileutil.WithFileSystem(context.Background(), memoryGenAITestFS{files: map[string][]byte{libraryPath: {}}})
	err := initialiseORTGenAI(ctx, &options.Options{ORTOptions: &options.OrtOptions{LibraryDir: &libraryDir}})
	if err == nil || !strings.Contains(err.Error(), "cannot find the ortgenai library at") {
		t.Fatalf("initialiseORTGenAI() error = %v, want local OS lookup to report missing library", err)
	}
}

func TestDefaultOSGenerativeCUDAKeepsDirectoryLoading(t *testing.T) {
	if ortgenai.IsInitialized() {
		t.Skip("ORT GenAI environment is already initialized")
	}
	libraryDir := t.TempDir()
	ctx := fileutil.WithFileSystem(context.Background(), nil)
	err := createORTGenerativeSession(ctx, &Model{Path: "missing-model"}, &options.Options{
		ORTOptions: &options.OrtOptions{LibraryDir: &libraryDir, CudaOptions: map[string]string{}},
	})
	if err == nil || !strings.Contains(err.Error(), "cannot find the ortgenai library at") {
		t.Fatalf("default OS CUDA loading error = %v, want directory path to reach native library check", err)
	}
}

func TestDefaultOSGenerativeCPUKeepsDirectoryLoading(t *testing.T) {
	if ortgenai.IsInitialized() {
		t.Skip("ORT GenAI environment is already initialized")
	}
	libraryDir := t.TempDir()
	ctx := fileutil.WithFileSystem(context.Background(), nil)
	err := createORTGenerativeSession(ctx, &Model{Path: "missing-model"}, &options.Options{
		ORTOptions: &options.OrtOptions{LibraryDir: &libraryDir},
	})
	if err == nil || !strings.Contains(err.Error(), "cannot find the ortgenai library at") {
		t.Fatalf("default OS CPU loading error = %v, want directory path to reach native library check", err)
	}
}

func TestMemoryConfigGraphFilesPreserveGenAINames(t *testing.T) {
	config := []byte(`{"model":{"decoder":{"filename":"./decoder.onnx"},"encoder":{"filename":"encoder.onnx"}}}`)
	graphs, err := memoryConfigGraphFiles(config)
	if err != nil {
		t.Fatal(err)
	}
	if len(graphs) != 2 {
		t.Fatalf("got %d graph files, want 2", len(graphs))
	}
	if graphs[0].name != "./decoder.onnx" || graphs[0].bundlePath != "decoder.onnx" {
		t.Fatalf("first graph = %#v, want original GenAI name and normalized bundle path", graphs[0])
	}
	if graphs[1].name != "encoder.onnx" || graphs[1].bundlePath != "encoder.onnx" {
		t.Fatalf("second graph = %#v, want encoder.onnx", graphs[1])
	}
}

func TestMemoryConfigRejectsUnsupportedBundlesAndReferences(t *testing.T) {
	tests := []struct {
		name   string
		config string
		want   string
	}{
		{name: "parent traversal", config: `{"model":{"decoder":{"filename":"../model.onnx"}}}`, want: "escapes"},
		{name: "absolute path", config: `{"model":{"decoder":{"filename":"/tmp/model.onnx"}}}`, want: "relative"},
		{name: "multimodal", config: `{"model":{"decoder":{"filename":"model.onnx"},"vision":{"filename":"vision.onnx"}}}`, want: "multimodal"},
		{name: "GPU provider", config: `{"model":{"decoder":{"filename":"model.onnx","provider":"CUDAExecutionProvider"}}}`, want: "CPU provider"},
		{name: "GPU device", config: `{"model":{"decoder":{"filename":"model.onnx","device_id":0}}}`, want: "GPU/device"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := memoryConfigGraphFiles([]byte(test.config))
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("memoryConfigGraphFiles() error = %v, want containing %q", err, test.want)
			}
		})
	}
}

func TestMemoryExternalDataReferences(t *testing.T) {
	model := &protos.ModelProto{Graph: &protos.GraphProto{Initializer: []*protos.TensorProto{{
		DataLocation: protos.TensorProto_EXTERNAL,
		ExternalData: []*protos.StringStringEntryProto{{Key: "location", Value: "weights/model.bin"}},
	}}}}
	graph, err := proto.Marshal(model)
	if err != nil {
		t.Fatal(err)
	}
	locations, err := onnxExternalDataLocations(graph)
	if err != nil {
		t.Fatal(err)
	}
	if len(locations) != 1 || locations[0] != "weights/model.bin" {
		t.Fatalf("external locations = %v, want [weights/model.bin]", locations)
	}
	resolved, err := resolveGraphRelativeBundlePath("graphs/model.onnx", locations[0])
	if err != nil || resolved != "graphs/weights/model.bin" {
		t.Fatalf("resolved external path = %q, %v; want graphs/weights/model.bin", resolved, err)
	}
}

func TestMemoryExternalDataRejectsMissingAndEscapingReferences(t *testing.T) {
	model := &protos.ModelProto{Graph: &protos.GraphProto{Initializer: []*protos.TensorProto{{DataLocation: protos.TensorProto_EXTERNAL}}}}
	graph, err := proto.Marshal(model)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := onnxExternalDataLocations(graph); err == nil || !strings.Contains(err.Error(), "missing its location") {
		t.Fatalf("external tensor without location error = %v", err)
	}
	for _, reference := range []string{"../../secret.bin", "/secret.bin", `..\\secret.bin`} {
		if _, err := resolveGraphRelativeBundlePath("graphs/model.onnx", reference); err == nil {
			t.Errorf("resolveGraphRelativeBundlePath(%q) unexpectedly succeeded", reference)
		}
	}
}

func TestValidateMemoryGenAIOptions(t *testing.T) {
	tests := []struct {
		name string
		opts options.OrtOptions
		want string
	}{
		{name: "defaults", opts: options.OrtOptions{}},
		{name: "engine", opts: options.OrtOptions{UseEngine: true}, want: "WithGenerativeEngine"},
		{name: "mtp", opts: options.OrtOptions{UseMTP: new(true)}, want: "MTP"},
		{name: "cuda", opts: options.OrtOptions{CudaOptions: map[string]string{}}, want: "CPU only"},
		{name: "GenAI GPU", opts: options.OrtOptions{GenAIGPUDeviceID: new(0)}, want: "CPU only"},
		{name: "adapter", opts: options.OrtOptions{GenAIAdapters: []options.GenAIAdapterConfig{{Path: "adapter.onnx"}}}, want: "path-based"},
		{name: "threads", opts: options.OrtOptions{IntraOpNumThreads: new(1)}, want: "session-specific"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := validateMemoryGenAIOptions(&test.opts)
			if test.want == "" {
				if err != nil {
					t.Fatalf("validateMemoryGenAIOptions() error = %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("validateMemoryGenAIOptions() error = %v, want containing %q", err, test.want)
			}
		})
	}
}

func TestStandaloneChatTemplateIsMergedOnlyWhenMissing(t *testing.T) {
	template := []byte(`{{ "hello" }}`)
	merged, err := mergeStandaloneChatTemplate([]byte(`{"model_max_length":77}`), template)
	if err != nil {
		t.Fatal(err)
	}
	var mergedObject map[string]any
	if err := json.Unmarshal(merged, &mergedObject); err != nil {
		t.Fatal(err)
	}
	if mergedObject["chat_template"] != string(template) {
		t.Fatalf("merged chat_template = %#v, want %q", mergedObject["chat_template"], template)
	}

	embedded := []byte(`{"chat_template":"existing"}`)
	unchanged, err := mergeStandaloneChatTemplate(embedded, template)
	if err != nil || string(unchanged) != string(embedded) {
		t.Fatalf("embedded chat_template result = %s, %v; want existing config unchanged", unchanged, err)
	}
	merged, err = mergeStandaloneChatTemplate([]byte(`{"chat_template":""}`), template)
	if err != nil || json.Unmarshal(merged, &mergedObject) != nil || mergedObject["chat_template"] != string(template) {
		t.Fatalf("empty embedded chat_template was not filled from standalone file: %s (%v)", merged, err)
	}
}
