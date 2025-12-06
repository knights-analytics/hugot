package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/util/fileutil"
)

// download the test models.

type downloadModel struct {
	name             string
	onnxFilePath     string
	externalDataPath string
}

var models = []downloadModel{
	{name: "KnightsAnalytics/all-MiniLM-L6-v2"},
	{name: "KnightsAnalytics/deberta-v3-base-zeroshot-v1"},
	{name: "KnightsAnalytics/distilbert-base-uncased-finetuned-sst-2-english"},
	{name: "KnightsAnalytics/distilbert-NER"},
	{name: "KnightsAnalytics/roberta-base-go_emotions"},
	{name: "KnightsAnalytics/jina-reranker-v1-tiny-en", onnxFilePath: "model.onnx"},
	{name: "KnightsAnalytics/resnet50"},
	{name: "KnightsAnalytics/detr-resnet-50", onnxFilePath: "onnx/model.onnx"},
	{name: "KnightsAnalytics/Phi-3.5-mini-instruct-onnx", onnxFilePath: "phi-3.5-mini-instruct-cpu-int4-awq-block-128-acc-level-4.onnx", externalDataPath: "phi-3.5-mini-instruct-cpu-int4-awq-block-128-acc-level-4.onnx.data"},
}

// Additional files to download (direct URLs).
var extraFiles = []struct {
	url, dest string
}{
	// Cat image from HuggingFace cats-image dataset
	{"https://huggingface.co/datasets/huggingface/cats-image/resolve/main/cats_image.jpeg", "./models/imageData/cat.jpg"},
}

func main() {
	if ok, err := fileutil.FileExists("./models"); err == nil {
		if !ok {
			err = os.MkdirAll("./models", os.ModePerm)
			if err != nil {
				panic(err)
			}
		}
		for _, model := range models {
			if os.Getenv("CI") != "" && model.name == "KnightsAnalytics/Phi-3.5-mini-instruct-onnx" {
				continue // skipping this model for cicd
			}

			if ok, err = fileutil.FileExists("./models/" + strings.ReplaceAll(model.name, "/", "_")); err == nil {
				if !ok {
					options := hugot.NewDownloadOptions()
					options.OnnxFilePath = model.onnxFilePath
					options.ExternalDataPath = model.externalDataPath
					fmt.Printf("Downloading %s\n", model.name)
					outPath, dlErr := hugot.DownloadModel(model.name, "./models", options)
					if dlErr != nil {
						panic(dlErr)
					}
					fmt.Printf("Downloaded %s to %s\n", model.name, outPath)
				}
			} else {
				panic(err)
			}
		}
	} else {
		panic(err)
	}

	// Download extra files (images, labels)
	if ok, err := fileutil.FileExists("./models/imageData"); err == nil {
		if !ok {
			err = os.MkdirAll("./models/imageData", os.ModePerm)
			if err != nil {
				panic(err)
			}
		}
		for _, f := range extraFiles {
			if exists, _ := fileutil.FileExists(f.dest); !exists {
				if err = downloadFile(context.Background(), f.url, f.dest); err != nil {
					panic(err)
				}
			}
		}
	} else {
		panic(err)
	}
}

// downloadFile downloads a file from a URL to a destination path.
func downloadFile(ctx context.Context, url string, dest string) error {
	out, err := os.Create(dest)
	if err != nil {
		return err
	}
	defer func() {
		err = errors.Join(out.Close())
	}()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil) // #nosec G107 Users may choose to download models from internal sources
	if err != nil {
		return err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer func() {
		err = errors.Join(resp.Body.Close())
	}()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("failed to download %s: status %s", url, resp.Status)
	}

	_, err = io.Copy(out, resp.Body)
	return err
}
