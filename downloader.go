//go:build !NODOWNLOAD

package hugot

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path"
	"path/filepath"
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
	PreservePaths         bool
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

	// make sure it's an onnx model with tokenizer
	downloadFiles, err := ValidateDownloadedHFModel(repo, options)
	if err != nil {
		return "", err
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
				destinationPath = filepath.Join(modelPath, downloadFiles[j])
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

	tokenizerPath := ""
	onnxPath := ""
	var toDownload []string
	var allOnnx []string
	for fileName, err := range repo.IterFileNames() {
		if err != nil {
			return nil, err
		}

		baseFileName := filepath.Base(fileName)
		if baseFileName == "tokenizer.json" {
			tokenizerPath = fileName
		} else if baseFileName == "special_tokens_map.json" ||
			baseFileName == "tokenizer_config.json" ||
			baseFileName == "config.json" ||
			baseFileName == "genai_config.json" ||
			baseFileName == "vocab.txt" ||
			baseFileName == "chat_template.jinja" {
			toDownload = append(toDownload, fileName)
		} else if filepath.Ext(baseFileName) == ".onnx" {
			if options.OnnxFilePath != "" {
				if fileName == options.OnnxFilePath {
					onnxPath = fileName
				}
			} else {
				onnxPath = fileName
			}
			allOnnx = append(allOnnx, fileName)
		} else if options.ExternalDataPath != "" && fileName == options.ExternalDataPath {
			toDownload = append(toDownload, fileName)
		}
	}

	var errs []error

	if options.OnnxFilePath != "" {
		if onnxPath == "" {
			errs = append(errs, fmt.Errorf("model .onnx file not found at %s", options.OnnxFilePath))
		}
	} else {
		numModels := len(allOnnx)
		if numModels == 0 {
			errs = append(errs, fmt.Errorf("model does not have a .onnx file, Hugot only works with onnx models"))
		} else if numModels > 1 {
			errs = append(errs, fmt.Errorf("model has multiple .onnx files, please specify one of the following onnxFilePaths: %s", strings.Join(allOnnx, " ")))
		}
	}

	files := append(toDownload, onnxPath)
	if tokenizerPath != "" {
		files = append(files, tokenizerPath)
	}
	files = append(files, options.AdditionalFilePaths...)
	return files, errors.Join(errs...)
}
