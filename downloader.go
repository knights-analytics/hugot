//go:build !NODOWNLOAD

package hugot

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"path"
	"path/filepath"
	"strings"
	"time"

	hfd "github.com/bodaay/HuggingFaceModelDownloader/hfdownloader"
)

// DownloadOptions is a struct of options that can be passed to DownloadModel.
type DownloadOptions struct {
	AuthToken             string
	SkipSha               bool
	Branch                string
	MaxRetries            int
	RetryInterval         int
	ConcurrentConnections int
	Verbose               bool
}

// NewDownloadOptions creates new DownloadOptions struct with default values.
// Override the values to specify different download options.
func NewDownloadOptions() DownloadOptions {
	d := DownloadOptions{}
	d.Branch = "main"
	d.MaxRetries = 5
	d.RetryInterval = 5
	d.ConcurrentConnections = 5
	return d
}

// DownloadModel can be used to download a model directly from huggingface. Before the model is downloaded,
// validation occurs to ensure there is an .onnx and tokenizers.json file. Hugot only works with onnx models.
func DownloadModel(modelName string, destination string, options DownloadOptions) (string, error) {
	// make sure it's an onnx model with tokenizer
	err := validateDownloadHfModel(modelName, options.Branch, options.AuthToken)
	if err != nil {
		return "", err
	}

	// replicates code in hf downloader
	modelP := modelName
	if strings.Contains(modelP, ":") {
		modelP = strings.Split(modelName, ":")[0]
	}
	modelPath := path.Join(destination, strings.Replace(modelP, "/", "_", -1))

	for i := 0; i < options.MaxRetries; i++ {
		if err := hfd.DownloadModel(modelName, false, options.SkipSha, false, destination, options.Branch, options.ConcurrentConnections, options.AuthToken, !options.Verbose); err != nil {
			if options.Verbose {
				fmt.Printf("Warning: attempt %d / %d failed, error: %s\n", i+1, options.MaxRetries, err)
			}
			time.Sleep(time.Duration(options.RetryInterval) * time.Second)
			continue
		}
		if options.Verbose {
			fmt.Printf("\nDownload of %s completed successfully\n", modelName)
		}
		return modelPath, nil
	}
	return "", fmt.Errorf("failed to download %s after %d attempts", modelName, options.MaxRetries)
}

type hfFile struct {
	Type        string `json:"type"`
	Path        string `json:"path"`
	IsDirectory bool
}

func validateDownloadHfModel(modelPath string, branch string, authToken string) error {
	if strings.Contains(modelPath, ":") {
		return errors.New("model filters are not supported")
	}

	client := &http.Client{}

	hasTokenizer, hasOnxx, err := checkURL(client, fmt.Sprintf("https://huggingface.co/api/models/%s/tree/%s", modelPath, branch), authToken)
	if err != nil {
		return err
	}

	var errs []error
	if !hasOnxx {
		errs = append(errs, fmt.Errorf("model does not have a model.onnx file, Hugot only works with onnx models"))
	}
	if !hasTokenizer {
		errs = append(errs, fmt.Errorf("model does not have a tokenizer.json file"))
	}
	return errors.Join(errs...)
}

func checkURL(client *http.Client, url string, authToken string) (bool, bool, error) {
	var tokenizerFound bool
	var onnxFound bool
	req, err := http.NewRequestWithContext(context.Background(), http.MethodGet, url, nil)
	if err != nil {
		return false, false, err
	}
	if authToken != "" {
		req.Header.Add("Authorization", "Bearer "+authToken)
	}
	resp, err := client.Do(req)
	if err != nil {
		return false, false, err
	}
	defer func(resp *http.Response) {
		err = errors.Join(err, resp.Body.Close())
	}(resp)

	var filesList []hfFile
	e := json.NewDecoder(resp.Body).Decode(&filesList)
	if e != nil {
		return false, false, e
	}

	var dirs []hfFile
	for _, f := range filesList {
		if filepath.Base(f.Path) == "tokenizer.json" {
			tokenizerFound = true
		}
		if filepath.Ext(f.Path) == ".onnx" {
			onnxFound = true
		}
		if f.Type == "directory" {
			// Do dirs later if files not found at this level
			dirs = append(dirs, f)
		}
		if onnxFound && tokenizerFound {
			break
		}
	}

	if !(onnxFound && tokenizerFound) {
		for _, dir := range dirs {
			tokenizerFoundRec, onnxFoundRec, dirErr := checkURL(client, url+"/"+dir.Path, authToken)
			if dirErr != nil {
				return false, false, dirErr
			}
			tokenizerFound = tokenizerFound || tokenizerFoundRec
			onnxFound = onnxFound || onnxFoundRec
			if onnxFound && tokenizerFound {
				break
			}
		}
	}

	return tokenizerFound, onnxFound, nil
}
