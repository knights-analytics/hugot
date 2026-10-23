//go:build !NODOWNLOAD

package hugot

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/gomlx/go-huggingface/hub"

	"github.com/knights-analytics/hugot/util/fileutil"
)

// DownloadOptions is a struct of options that can be passed to DownloadModel.
type DownloadOptions struct {
	AuthToken             string
	OnnxFilePath          string
	ExternalDataPath      string
	Branch                string
	MaxRetries            int
	RetryInterval         int
	ConcurrentConnections int
	Verbose               bool
	AdditionalFilePaths   []string
	// DownloadGraphBundle collects every artifact beside the selected graph and
	// in its subdirectories, including child graphs and external weights. It
	// implies PreservePaths. Select a specific variant with OnnxFilePath.
	DownloadGraphBundle bool
	PreservePaths       bool
	CacheDir            string
}

// NewDownloadOptions creates new DownloadOptions struct with default values.
// Override the values to specify different download options.
func NewDownloadOptions() DownloadOptions {
	d := DownloadOptions{
		Branch:                "main",
		MaxRetries:            5,
		RetryInterval:         5,
		ConcurrentConnections: 5,
	}
	return d
}

// DownloadModel can be used to download a model directly from huggingface. Before the model is downloaded,
// validation occurs to ensure there is an .onnx and tokenizers.json file. Hugot only works with onnx models.
func DownloadModel(ctx context.Context, modelName string, destination string, options DownloadOptions) (string, error) {
	if options.DownloadGraphBundle {
		options.PreservePaths = true
	}
	// replicates code in hf downloader
	// DownloadModel is usable without a session, so bind the default filesystem
	// before using the context-based file helpers below.
	ctx = fileutil.WithFileSystem(ctx, nil)
	modelP := modelName
	if strings.Contains(modelP, ":") {
		modelP = strings.Split(modelName, ":")[0]
	}
	modelPath := path.Join(destination, strings.ReplaceAll(modelP, "/", "_"))

	// The files are only copied under destination after the whole model has
	// been fetched, so reject a destination that cannot hold them before
	// contacting the hub.
	info, statErr := fileutil.FileStats(ctx, destination)
	if statErr != nil {
		return "", fmt.Errorf("could not inspect destination %s: %w", destination, statErr)
	}
	if !info.IsDir() {
		return "", fmt.Errorf("destination %s is not a directory", destination)
	}
	if options.PreservePaths {
		for _, name := range options.AdditionalFilePaths {
			if _, err := preservedDownloadPath(modelPath, name); err != nil {
				return "", err
			}
		}
	}

	repo := hub.New(modelName)
	if options.AuthToken != "" {
		repo = repo.WithAuth(options.AuthToken)
	}
	if options.ConcurrentConnections > 0 {
		repo.MaxParallelDownload = options.ConcurrentConnections
	}
	if options.Verbose {
		repo.Verbosity = 1
		repo.WithProgressBar(true)
	} else {
		repo.Verbosity = 0
		repo.WithProgressBar(false)
	}
	if options.Branch != "" {
		repo.WithRevision(options.Branch)
	}
	if options.CacheDir != "" {
		repo.WithCacheDir(options.CacheDir)
	}

	// make sure it's an onnx model with tokenizer
	downloadFiles, err := ValidateDownloadedHFModel(repo, options)
	if err != nil {
		return "", err
	}
	if options.PreservePaths {
		for _, name := range downloadFiles {
			if _, err := preservedDownloadPath(modelPath, name); err != nil {
				return "", err
			}
		}
	}

	for i := 0; i < options.MaxRetries; i++ {
		downloadPaths, downloadErr := repo.DownloadFiles(downloadFiles...)
		if downloadErr != nil {
			if options.Verbose {
				fmt.Printf("Warning: attempt %d / %d failed, error: %s\n", i+1, options.MaxRetries, downloadErr)
			}
			time.Sleep(time.Duration(options.RetryInterval) * time.Second)
			continue
		}

		if mkdirErr := os.MkdirAll(modelPath, 0o755); mkdirErr != nil {
			return "", mkdirErr
		}

		for j, downloadPath := range downloadPaths {
			truePath, symErr := filepath.EvalSymlinks(downloadPath)
			if symErr != nil {
				return "", symErr
			}
			destinationPath := filepath.Join(modelPath, path.Base(downloadFiles[j]))
			if options.PreservePaths {
				destinationPath, err = preservedDownloadPath(modelPath, downloadFiles[j])
				if err != nil {
					return "", err
				}
				if mkdirErr := os.MkdirAll(filepath.Dir(destinationPath), 0o755); mkdirErr != nil {
					return "", mkdirErr
				}
			}
			moveErr := fileutil.CopyFile(ctx, truePath, destinationPath)
			if moveErr != nil {
				return "", moveErr
			}
		}

		if options.Verbose {
			fmt.Printf("\nDownload of %s completed successfully\n", modelName)
		}
		return modelPath, nil
	}

	return "", fmt.Errorf("failed to download %s after %d attempts", modelName, options.MaxRetries)
}

func preservedDownloadPath(modelPath, name string) (string, error) {
	if name == "" || path.IsAbs(name) || filepath.IsAbs(name) || strings.ContainsAny(name, `\:`) {
		return "", fmt.Errorf("invalid hub-relative path %q", name)
	}
	for segment := range strings.SplitSeq(name, "/") {
		if segment == ".." || segment == "." || segment == "" {
			return "", fmt.Errorf("invalid hub-relative path %q", name)
		}
	}
	destinationPath := filepath.Join(modelPath, filepath.FromSlash(name))
	rel, err := filepath.Rel(modelPath, destinationPath)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("invalid hub-relative path %q", name)
	}
	return destinationPath, nil
}

func ValidateDownloadedHFModel(repo *hub.Repo, options DownloadOptions) ([]string, error) {
	for i := 0; i < options.MaxRetries; i++ {
		err := repo.DownloadInfo(false)
		if err != nil {
			if options.Verbose {
				fmt.Printf("Warning: list repo attempt %d / %d failed, error: %s\n", i+1, options.MaxRetries, err)
			}
			if i+1 == options.MaxRetries {
				return nil, err
			}
			time.Sleep(time.Duration(options.RetryInterval) * time.Second)
		}
	}

	var names []string
	for fileName, err := range repo.IterFileNames() {
		if err != nil {
			return nil, err
		}
		names = append(names, fileName)
	}
	return selectModelArtifacts(names, options)
}

func selectModelArtifacts(names []string, options DownloadOptions) ([]string, error) {
	slices.Sort(names)
	files := map[string]bool{}
	var graphs []string
	for _, name := range names {
		switch path.Base(name) {
		case "tokenizer.json", "special_tokens_map.json", "tokenizer_config.json", "config.json", "genai_config.json", "vocab.txt", "chat_template.jinja",
			"processor_config.json", "preprocessor_config.json", "video_processor_config.json", "generation_config.json", "vision_processor.json", "speech_processor.json", "audio_processor_config.json":
			files[name] = true
		}
		if path.Ext(name) == ".onnx" {
			graphs = append(graphs, name)
		}
	}
	graph := options.OnnxFilePath
	if graph == "" {
		if len(graphs) == 0 {
			return nil, errors.New("model does not have a .onnx file, Hugot only works with onnx models")
		}
		if len(graphs) > 1 {
			return nil, fmt.Errorf("model has multiple .onnx files, please specify one of the following onnxFilePaths: %s", strings.Join(graphs, " "))
		}
		graph = graphs[0]
	} else if !slices.Contains(graphs, graph) {
		return nil, fmt.Errorf("model .onnx file not found at %s", graph)
	}
	files[graph] = true
	if options.ExternalDataPath != "" && !slices.Contains(names, options.ExternalDataPath) {
		return nil, fmt.Errorf("external model data %q not found", options.ExternalDataPath)
	}
	directory := path.Dir(graph)
	for _, name := range names {
		if name == graph+".data" || name == graph+"_data" || strings.HasPrefix(name, graph+"_data_") || name == options.ExternalDataPath {
			files[name] = true
		}
		if options.DownloadGraphBundle && (directory == "." || strings.HasPrefix(name, directory+"/")) {
			files[name] = true
		}
	}
	for _, name := range options.AdditionalFilePaths {
		if !slices.Contains(names, name) {
			return nil, fmt.Errorf("additional model artifact %q not found", name)
		}
		files[name] = true
	}
	result := make([]string, 0, len(files))
	for name := range files {
		result = append(result, name)
	}
	slices.Sort(result)
	return result, nil
}
