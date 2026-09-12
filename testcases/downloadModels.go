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
	additionalFiles  []string
	preservePaths    bool
}

var models = []downloadModel{
	{name: "KnightsAnalytics/all-MiniLM-L6-v2"},
	{name: "KnightsAnalytics/deberta-v3-base-zeroshot-v1"},
	{name: "KnightsAnalytics/distilbert-base-uncased-finetuned-sst-2-english"},
	{name: "KnightsAnalytics/distilbert-NER"},
	{name: "KnightsAnalytics/distilbert-onnx"},
	{name: "KnightsAnalytics/roberta-base-go_emotions"},
	{name: "KnightsAnalytics/jina-reranker-v1-tiny-en", onnxFilePath: "model.onnx"},
	{name: "KnightsAnalytics/resnet50"},
	{name: "KnightsAnalytics/detr-resnet-50", onnxFilePath: "model.onnx"},
	{name: "Xenova/clip-vit-base-patch32", onnxFilePath: "onnx/model.onnx"},
	{name: "Xenova/owlv2-base-patch16", onnxFilePath: "onnx/model.onnx"},
	{name: "Xenova/segformer-b0-finetuned-ade-512-512", onnxFilePath: "onnx/model.onnx"},
	{name: "Xenova/dpt-large", onnxFilePath: "onnx/model.onnx"},
	{name: "KnightsAnalytics/iris-decision-tree", onnxFilePath: "model.onnx"},
	{name: "KnightsAnalytics/qwen3-4B-int4", onnxFilePath: "model.onnx", externalDataPath: "model.onnx.data"},
	{name: "Xenova/wav2vec2-large-xlsr-53-gender-recognition-librispeech", onnxFilePath: "onnx/model.onnx"},
	{name: "Xenova/wav2vec2-base-960h", onnxFilePath: "onnx/model.onnx"},
	{name: "Xenova/mms-tts-eng", onnxFilePath: "onnx/model.onnx"},
	{name: "Xenova/mms-tts-spa", onnxFilePath: "onnx/model.onnx"},
	{
		name:             "microsoft/Phi-3.5-vision-instruct-onnx",
		onnxFilePath:     "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/phi-3.5-v-instruct-text.onnx",
		externalDataPath: "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/phi-3.5-v-instruct-text.onnx.data",
		additionalFiles: []string{
			"cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/genai_config.json",
			"cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/phi-3.5-v-instruct-embedding.onnx",
			"cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/phi-3.5-v-instruct-embedding.onnx.data",
			"cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/phi-3.5-v-instruct-vision.onnx",
			"cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/phi-3.5-v-instruct-vision.onnx.data",
			"cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/processor_config.json",
			"cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/special_tokens_map.json",
			"cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/tokenizer.json",
			"cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/tokenizer_config.json",
		},
		preservePaths: true,
	},
}

// Additional files to download (direct URLs).
var extraFiles = []struct {
	url, dest string
}{
	// Cat image from HuggingFace cats-image dataset
	{"https://huggingface.co/datasets/huggingface/cats-image/resolve/main/cats_image.jpeg", "./models/imageData/cat.jpg"},
}

func main() {
	ctx := fileutil.WithFileSystem(context.Background(), nil)
	if ok, err := fileutil.FileExists(ctx, "./models"); err == nil {
		if !ok {
			err = os.MkdirAll("./models", os.ModePerm)
			if err != nil {
				panic(err)
			}
		}
		for _, model := range models {
			if os.Getenv("CI") != "" && (model.name == "KnightsAnalytics/qwen3-4B-int4" || model.name == "microsoft/Phi-3.5-vision-instruct-onnx") {
				continue // skipping this model for cicd
			}

			if ok, err = fileutil.FileExists(ctx, "./models/"+strings.ReplaceAll(model.name, "/", "_")); err == nil {
				if !ok {
					options := hugot.NewDownloadOptions()
			options.OnnxFilePath = model.onnxFilePath
			options.ExternalDataPath = model.externalDataPath
			options.AdditionalFilePaths = model.additionalFiles
			options.PreservePaths = model.preservePaths
					fmt.Printf("Downloading %s\n", model.name)
					outPath, dlErr := hugot.DownloadModel(ctx, model.name, "./models", options)
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
	if ok, err := fileutil.FileExists(ctx, "./models/imageData"); err == nil {
		if !ok {
			err = os.MkdirAll("./models/imageData", os.ModePerm)
			if err != nil {
				panic(err)
			}
		}
		for _, f := range extraFiles {
			if exists, _ := fileutil.FileExists(ctx, f.dest); !exists {
				if err = downloadFile(ctx, f.url, f.dest); err != nil {
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
