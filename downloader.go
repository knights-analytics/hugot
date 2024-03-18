//go:build !NODOWNLOAD

package hugot

import (
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

// DownloadOptions is a struct of options that can be passed to DownloadModel
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
func (s *Session) DownloadModel(modelName string, destination string, options DownloadOptions) (string, error) {
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
			fmt.Printf("Warning: attempt %d / %d failed, error: %s\n", i+1, options.MaxRetries, err)
			time.Sleep(time.Duration(options.RetryInterval) * time.Second)
			continue
		}
		fmt.Printf("\nDownload of %s completed successfully\n", modelName)
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
	client := &http.Client{}
	if strings.Contains(modelPath, ":") {
		return errors.New("model filters are not supported")
	}
	req, err := http.NewRequest("GET", fmt.Sprintf("https://huggingface.co/api/models/%s/tree/%s", modelPath, branch), nil)
	if err != nil {
		return err
	}
	if authToken != "" {
		req.Header.Add("Authorization", "Bearer "+authToken)
	}

	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer func(resp *http.Response) {
		err = errors.Join(err, resp.Body.Close())
	}(resp)

	var filesList []hfFile
	e := json.NewDecoder(resp.Body).Decode(&filesList)
	if e != nil {
		return e
	}
	hasTokenizer := false
	hasOnxx := false

	for _, f := range filesList {
		if f.Path == "tokenizer.json" {
			hasTokenizer = true
		}
		if filepath.Ext(f.Path) == ".onnx" {
			hasOnxx = true
		}
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
