package main

import (
	"os"

	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/util"
)

// download the test models.
func main() {
	if ok, err := util.FileExists("./models"); err == nil {
		if !ok {

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
				_, dlErr := hugot.DownloadModel(modelName, "./models", downloadOptions)
				if dlErr != nil {
					panic(dlErr)
				}
			}
		}
	} else {
		panic(err)
	}
}
