package main

import (
	"fmt"
	"io"
	"net/http"
	"os"

	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/util"
)

// download the test models.

type downloadModel struct {
	name         string
	onnxFilePath string
}

var models = []downloadModel{
	{"KnightsAnalytics/all-MiniLM-L6-v2", ""},
	{"KnightsAnalytics/deberta-v3-base-zeroshot-v1", ""},
	{"KnightsAnalytics/distilbert-base-uncased-finetuned-sst-2-english", ""},
	{"KnightsAnalytics/distilbert-NER", ""},
	{"KnightsAnalytics/roberta-base-go_emotions", ""},
	{"KnightsAnalytics/jina-reranker-v1-tiny-en", "model.onnx"},
	{"KnightsAnalytics/resnet50", ""},
	{"KnightsAnalytics/Phi-3-mini-4k-instruct-onnx", "model.onnx"},
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
		for _, downloadModel := range models {
			options := hugot.NewDownloadOptions()
			if downloadModel.onnxFilePath != "" {
				options.OnnxFilePath = downloadModel.onnxFilePath
			}
			_, dlErr := hugot.DownloadModel(downloadModel.name, "./models", options)
			if dlErr != nil {
				panic(dlErr)
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
	defer out.Close()

	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return fmt.Errorf("failed to download %s: status %s", url, resp.Status)
	}

	_, err = io.Copy(out, resp.Body)
	return err
}
