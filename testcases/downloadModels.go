package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
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
	{name: "Xenova/distilbert-base-uncased-finetuned-sst-2-english", onnxFilePath: "onnx/model.onnx"},
	{name: "Xenova/bert-base-uncased", onnxFilePath: "onnx/model.onnx"},
	{name: "KnightsAnalytics/distilbert-NER"},
	{name: "KnightsAnalytics/distilbert-onnx"},
	{name: "SamLowe/roberta-base-go_emotions-onnx", onnxFilePath: "onnx/model.onnx"},
	{name: "jinaai/jina-reranker-v1-tiny-en", onnxFilePath: "onnx/model.onnx"},
	{name: "KnightsAnalytics/resnet50"},
	{name: "Xenova/detr-resnet-50", onnxFilePath: "onnx/model.onnx"},
	{name: "Xenova/clip-vit-base-patch32", onnxFilePath: "onnx/model.onnx"},
	{name: "Xenova/owlv2-base-patch16", onnxFilePath: "onnx/model_q4.onnx"},
	{name: "Xenova/segformer-b0-finetuned-ade-512-512", onnxFilePath: "onnx/model.onnx"},
	{name: "Xenova/dpt-large", onnxFilePath: "onnx/model_q4.onnx"},
	{name: "KnightsAnalytics/iris-decision-tree", onnxFilePath: "model.onnx"},
	{name: "KnightsAnalytics/qwen3-4B-int4", onnxFilePath: "model.onnx", externalDataPath: "model.onnx.data"},
	{name: "Xenova/wav2vec2-large-xlsr-53-gender-recognition-librispeech", onnxFilePath: "onnx/model.onnx"},
	{name: "Xenova/wav2vec2-base-960h", onnxFilePath: "onnx/model_q4.onnx"},
	{name: "Xenova/mms-tts-eng", onnxFilePath: "onnx/model.onnx"},
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
	// Speech sample from the HuggingFace LibriSpeech dataset
	{"https://huggingface.co/datasets/bezzam/audio_samples/resolve/main/librispeech_mr_quilter.wav", "./models/audioData/librispeech.wav"},
}

func main() {
	ctx := fileutil.WithFileSystem(context.Background(), nil)
	if err := os.MkdirAll("./models", os.ModePerm); err != nil {
		panic(err)
	}
	for _, model := range models {
		if os.Getenv("CI") != "" && (model.name == "KnightsAnalytics/qwen3-4B-int4" || model.name == "microsoft/Phi-3.5-vision-instruct-onnx") {
			continue // skipping this model for cicd
		}
		exists, err := fileutil.FileExists(ctx, "./models/"+strings.ReplaceAll(model.name, "/", "_"))
		if err != nil {
			panic(err)
		}
		if exists {
			continue
		}
		options := hugot.NewDownloadOptions()
		options.OnnxFilePath = model.onnxFilePath
		options.ExternalDataPath = model.externalDataPath
		options.AdditionalFilePaths = model.additionalFiles
		options.PreservePaths = model.preservePaths
		fmt.Printf("Downloading %s\n", model.name)
		outPath, err := hugot.DownloadModel(ctx, model.name, "./models", options)
		if err != nil {
			panic(err)
		}
		fmt.Printf("Downloaded %s to %s\n", model.name, outPath)
	}

	// Download extra files (images, audio, labels)
	if err := os.MkdirAll("./models/imageData", os.ModePerm); err != nil {
		panic(err)
	}
	if err := os.MkdirAll("./models/audioData", os.ModePerm); err != nil {
		panic(err)
	}
	for _, f := range extraFiles {
		exists, err := fileutil.FileExists(ctx, f.dest)
		if err != nil {
			panic(err)
		}
		if !exists {
			if err := downloadFile(ctx, f.url, f.dest); err != nil {
				panic(err)
			}
		}
	}
}

// downloadFile downloads a file from a URL to a destination path.
func downloadFile(ctx context.Context, url string, dest string) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil) // #nosec G107 Users may choose to download models from internal sources
	if err != nil {
		return err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer func() {
		err = errors.Join(err, resp.Body.Close())
	}()
	if resp.StatusCode != http.StatusOK {
		err = fmt.Errorf("failed to download %s: status %s", url, resp.Status)
		return err
	}
	out, err := os.CreateTemp(filepath.Dir(dest), ".download-*")
	if err != nil {
		return err
	}
	defer func() {
		err = errors.Join(err, os.Remove(out.Name()))
	}()
	_, copyErr := io.Copy(out, resp.Body)
	if copyErr != nil {
		err = errors.Join(copyErr, out.Close())
		return err
	}
	syncErr := out.Sync()
	closeFileErr := out.Close()
	if err = errors.Join(copyErr, syncErr, closeFileErr); err != nil {
		return err
	}
	err = os.Rename(out.Name(), dest)
	return err
}
