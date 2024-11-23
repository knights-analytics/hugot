package main

import (
	"context"
	"os"

	"github.com/knights-analytics/hugot"
	util "github.com/knights-analytics/hugot/utils"
)

var onnxruntimeSharedLibrary = "/usr/lib64/onnxruntime.so"

// download the test models.
func main() {
	if ok, err := util.FileSystem.Exists(context.Background(), "./models"); err == nil {
		if !ok {
			session, err := hugot.NewSession("ORT", hugot.WithOnnxLibraryPath(onnxruntimeSharedLibrary))
			if err != nil {
				panic(err)
			}
			defer func(s hugot.Session) {
				err := s.Destroy()
				if err != nil {
					panic(err)
				}
			}(*session)

			err = os.MkdirAll("./models", os.ModePerm)
			if err != nil {
				panic(err)
			}
			downloadOptions := hugot.NewDownloadOptions()
			for _, modelName := range []string{
				"sentence-transformers/all-MiniLM-L6-v2",
				"protectai/deberta-v3-base-zeroshot-v1-onnx",
				"KnightsAnalytics/distilbert-base-uncased-finetuned-sst-2-english",
				"KnightsAnalytics/distilbert-NER",
				"SamLowe/roberta-base-go_emotions-onnx"} {
				_, err := session.DownloadModel(modelName, "./models", downloadOptions)
				if err != nil {
					panic(err)
				}
			}
		}
	} else {
		panic(err)
	}
}
