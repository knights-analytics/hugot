package main

import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/util"
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
	{name: "KnightsAnalytics/Phi-3.5-mini-instruct-onnx", onnxFilePath: "phi-3.5-mini-instruct-cpu-int4-awq-block-128-acc-level-4.onnx", externalDataPath: "phi-3.5-mini-instruct-cpu-int4-awq-block-128-acc-level-4.onnx.data"},
}

// Additional files to download (direct URLs)
var extraFiles = []struct {
	url, dest string
}{
	// Cat image from HuggingFace cats-image dataset
	{"https://huggingface.co/datasets/huggingface/cats-image/resolve/main/cats_image.jpeg", "./models/imageData/cat.jpg"},
}

func main() {
	if ok, err := util.FileExists("./models"); err == nil {
		if !ok {
			err = os.MkdirAll("./models", os.ModePerm)
			if err != nil {
				panic(err)
			}
		}
		for _, model := range models {
			if ok, err = util.FileExists("./models/" + strings.Replace(model.name, "/", "_", -1)); err == nil {
				if !ok {
					options := hugot.NewDownloadOptions()
					options.OnnxFilePath = model.onnxFilePath
					options.ExternalDataPath = model.externalDataPath
					fmt.Println(fmt.Sprintf("Downloading %s", model.name))
					outPath, dlErr := hugot.DownloadModel(model.name, "./models", options)
					if dlErr != nil {
						panic(dlErr)
					}
					fmt.Println(fmt.Sprintf("Downloaded %s to %s", model.name, outPath))
				}
			} else {
				panic(err)
			}
		}
	} else {
		panic(err)
	}

	// Download extra files (images, labels)
	if ok, err := util.FileExists("./models/imageData"); err == nil {
		if !ok {
			err = os.MkdirAll("./models/imageData", os.ModePerm)
			if err != nil {
				panic(err)
			}
		}
		for _, f := range extraFiles {
			if exists, _ := util.FileExists(f.dest); !exists {
				if err := downloadFile(f.url, f.dest); err != nil {
					panic(err)
				}
			}
		}
	} else {
		panic(err)
	}
}

// downloadFile downloads a file from a URL to a destination path.
func downloadFile(url, dest string) error {
	out, err := os.Create(dest)
	if err != nil {
		return err
	}
	defer func() {
		err = errors.Join(out.Close())
	}()

	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer func() {
		err = errors.Join(resp.Body.Close())
	}()

	if resp.StatusCode != 200 {
		return fmt.Errorf("failed to download %s: status %s", url, resp.Status)
	}

	_, err = io.Copy(out, resp.Body)
	return err
}
