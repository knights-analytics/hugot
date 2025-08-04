package main

import (
	"os"

	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/util"
)

// download the test models.

type downloadModel struct {
	name         string
	onnxFilePath string
}

var models []downloadModel = []downloadModel{
	{"KnightsAnalytics/all-MiniLM-L6-v2", ""},
	{"KnightsAnalytics/deberta-v3-base-zeroshot-v1", ""},
	{"KnightsAnalytics/distilbert-base-uncased-finetuned-sst-2-english", ""},
	{"KnightsAnalytics/distilbert-NER", ""},
	{"KnightsAnalytics/roberta-base-go_emotions", ""},
	{"KnightsAnalytics/jina-reranker-v1-tiny-en", "model.onnx"},
}

func main() {
	if ok, err := util.FileExists("./models"); err == nil {
		if !ok {

			err = os.MkdirAll("./models", os.ModePerm)
			if err != nil {
				panic(err)
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
		}
	} else {
		panic(err)
	}
}
