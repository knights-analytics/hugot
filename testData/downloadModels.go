package main

import (
	"context"
	"os"

	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/util"
)

var onnxruntimeSharedLibrary = "/usr/lib64/onnxruntime.so"

// download the test models.
func main() {
	if ok, err := util.FileSystem.Exists(context.Background(), "./models"); err == nil {
		if !ok {
			session, err := hugot.NewORTSession(options.WithOnnxLibraryPath(onnxruntimeSharedLibrary))
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
				"KnightsAnalytics/all-MiniLM-L6-v2",
				"KnightsAnalytics/deberta-v3-base-zeroshot-v1",
				"KnightsAnalytics/distilbert-base-uncased-finetuned-sst-2-english",
				"KnightsAnalytics/distilbert-NER",
				"KnightsAnalytics/roberta-base-go_emotions"} {
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
