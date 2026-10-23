package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"image"
	"image/jpeg"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/util/fileutil"
	"golang.org/x/image/draw"
)

// download the test models.

type downloadModel struct {
	name             string
	onnxFilePath     string
	externalDataPath string
	additionalFiles  []string
	preservePaths    bool
	revision         string
}

var models = []downloadModel{
	{name: "KnightsAnalytics/all-MiniLM-L6-v2"},
	{name: "KnightsAnalytics/deberta-v3-base-zeroshot-v1"},
	{name: "Xenova/distilbert-base-uncased-finetuned-sst-2-english", onnxFilePath: "onnx/model.onnx"},
	{name: "Xenova/bert-base-uncased", onnxFilePath: "onnx/model.onnx"},
	{name: "KnightsAnalytics/distilbert-NER"},
	{name: "KnightsAnalytics/distilbert-onnx"},
	{
		name:         "KnightsAnalytics/deepparse-bpemb-onnx",
		onnxFilePath: "model.onnx",
		additionalFiles: []string{
			"encoder.onnx",
			"decoder_step.onnx",
			"manifest.json",
			"tokenizer.json",
			"tokenizer.model",
			"bpemb_sentencepiece.model",
		},
		preservePaths: true,
	},
	{
		name:         "KnightsAnalytics/deepparse-bpemb-attn-onnx",
		onnxFilePath: "model.onnx",
		additionalFiles: []string{
			"encoder.onnx",
			"decoder_step.onnx",
			"manifest.json",
			"tokenizer.json",
			"tokenizer.model",
			"bpemb_sentencepiece.model",
		},
		preservePaths: true,
	},
	{name: "SamLowe/roberta-base-go_emotions-onnx", onnxFilePath: "onnx/model.onnx"},
	{name: "jinaai/jina-reranker-v1-tiny-en", onnxFilePath: "onnx/model.onnx"},
	{name: "KnightsAnalytics/resnet50", onnxFilePath: "resnet50-v1-12.onnx"},
	{name: "KnightsAnalytics/tapas-base-finetuned-wtq", onnxFilePath: "model.onnx", additionalFiles: []string{"config.json", "tokenizer_config.json", "vocab.txt"}},
	{name: "Xenova/detr-resnet-50", onnxFilePath: "onnx/model.onnx"},
	{name: "Xenova/clip-vit-base-patch32", onnxFilePath: "onnx/model.onnx"},
	{name: "Xenova/owlv2-base-patch16", onnxFilePath: "onnx/model_q4.onnx"},
	{name: "KnightsAnalytics/grounding-dino-base", onnxFilePath: "model.onnx", additionalFiles: []string{"processor_config.json"}},
	{name: "Xenova/segformer-b0-finetuned-ade-512-512", onnxFilePath: "onnx/model.onnx"},
	{name: "Xenova/slimsam-77-uniform", onnxFilePath: "onnx/vision_encoder.onnx", additionalFiles: []string{"onnx/prompt_encoder_mask_decoder.onnx", "preprocessor_config.json"}},
	{name: "Xenova/donut-base-finetuned-docvqa", onnxFilePath: "onnx/encoder_model_quantized.onnx", additionalFiles: []string{"onnx/decoder_model_quantized.onnx", "preprocessor_config.json"}},
	{name: "KnightsAnalytics/tapas-base-finetuned-sqa", onnxFilePath: "model.onnx"},
	{name: "Xenova/dino-vits16", onnxFilePath: "onnx/model.onnx", additionalFiles: []string{"preprocessor_config.json"}},
	{name: "Xenova/modnet", onnxFilePath: "onnx/model.onnx", additionalFiles: []string{"preprocessor_config.json"}},
	{name: "Xenova/dpt-large", onnxFilePath: "onnx/model_q4.onnx"},
	{name: "KnightsAnalytics/iris-decision-tree", onnxFilePath: "model.onnx"},
	{name: "KnightsAnalytics/qwen3-4B-int4", onnxFilePath: "model.onnx", externalDataPath: "model.onnx.data"},
	{name: "Xenova/wav2vec2-large-xlsr-53-gender-recognition-librispeech", onnxFilePath: "onnx/model.onnx"},
	{name: "Xenova/larger_clap_music_and_speech", onnxFilePath: "onnx/model_quantized.onnx"},
	{name: "Xenova/wav2vec2-base-960h", onnxFilePath: "onnx/model_q4.onnx"},
	{name: "Xenova/mms-tts-eng", onnxFilePath: "onnx/model.onnx"},
	{
		name:         "onnx-community/Janus-Pro-1B-ONNX",
		onnxFilePath: "onnx/embed_tokens.onnx",
		additionalFiles: []string{
			"onnx/gen_head.onnx",
			"onnx/gen_img_embeds.onnx",
			"onnx/image_decode.onnx",
			"onnx/language_model_q4.onnx",
			"onnx/lm_head.onnx",
			"onnx/prepare_inputs_embeds.onnx",
		},
		preservePaths: true,
		revision:      "04efdf2e36cb07a034b0d94f7322356b292f0418",
	},
	{
		name:         "LiquidAI/LFM2.5-Audio-1.5B-ONNX",
		onnxFilePath: "onnx/audio_embedding_q4.onnx",
		additionalFiles: []string{
			"onnx/audio_detokenizer_q4.onnx",
			"onnx/audio_detokenizer_q4.onnx_data",
			"onnx/decoder_q4.onnx",
			"onnx/decoder_q4.onnx_data",
			"onnx/embed_tokens.bin",
			"onnx/embed_tokens.json",
			"onnx/mel_config.json",
			"onnx/vocoder_depthformer_q4.onnx",
			"onnx/vocoder_depthformer_q4.onnx_data",
		},
		preservePaths: true,
		revision:      "62318d95ddf42a65e742cdd6fd33df91874a801d",
	},
	{
		name:         "microsoft/Phi-4-multimodal-instruct-onnx",
		onnxFilePath: "gpu/gpu-int4-rtn-block-32/phi-4-mm-text.onnx",
		additionalFiles: []string{
			"gpu/gpu-int4-rtn-block-32/added_tokens.json",
			"gpu/gpu-int4-rtn-block-32/genai_config.json",
			"gpu/gpu-int4-rtn-block-32/merges.txt",
			"gpu/gpu-int4-rtn-block-32/phi-4-mm-embedding.onnx",
			"gpu/gpu-int4-rtn-block-32/phi-4-mm-embedding.onnx.data",
			"gpu/gpu-int4-rtn-block-32/phi-4-mm-speech.onnx",
			"gpu/gpu-int4-rtn-block-32/phi-4-mm-speech.onnx.data",
			"gpu/gpu-int4-rtn-block-32/phi-4-mm-speech.onnx_adapter",
			"gpu/gpu-int4-rtn-block-32/phi-4-mm-vision.onnx",
			"gpu/gpu-int4-rtn-block-32/phi-4-mm-vision.onnx.data",
			"gpu/gpu-int4-rtn-block-32/phi-4-mm-vision.onnx_adapter",
			"gpu/gpu-int4-rtn-block-32/vocab.json",
		},
		preservePaths: true,
		revision:      "295df4cddf8c3740df7cd66e5d05f06a872fcaf5",
	},
	{
		name:         "KnightsAnalytics/videomae-base-finetuned-kinetics",
		onnxFilePath: "model.onnx",
		additionalFiles: []string{
			"config.json",
			"preprocessor_config.json",
			"export_verification.json",
		},
		preservePaths: true,
	},
	{
		name:         "KnightsAnalytics/superglue_outdoor",
		onnxFilePath: "model.onnx",
		additionalFiles: []string{
			"config.json",
			"preprocessor_config.json",
			"export_verification.json",
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
	{"https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/invoice.png", "./models/imageData/invoice.png"},
	{"https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/portrait-of-woman.jpg", "./models/imageData/portrait.jpg"},
	// Speech sample from the HuggingFace LibriSpeech dataset
	{"https://huggingface.co/datasets/bezzam/audio_samples/resolve/main/librispeech_mr_quilter.wav", "./models/audioData/librispeech.wav"},
}

func main() {
	selected := flag.String("model", "", "download only this Hugging Face model (default: all test models)")
	timeout := flag.Duration("timeout", 30*time.Minute, "deadline for downloading test models and data")
	connections := flag.Int("connections", 5, "parallel file downloads per model")
	verbose := flag.Bool("verbose", false, "show download progress and retry errors")
	flag.Parse()
	ctx, cancel := context.WithTimeout(context.Background(), *timeout)
	defer cancel()
	ctx = fileutil.WithFileSystem(ctx, nil)
	if err := os.MkdirAll("./models", os.ModePerm); err != nil {
		panic(err)
	}
	for _, model := range models {
		if *selected != "" && model.name != *selected {
			continue
		}
		if os.Getenv("CI") != "" && model.name == "KnightsAnalytics/qwen3-4B-int4" {
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
		options.ConcurrentConnections = *connections
		options.Verbose = *verbose
		options.OnnxFilePath = model.onnxFilePath
		options.ExternalDataPath = model.externalDataPath
		options.AdditionalFilePaths = model.additionalFiles
		options.PreservePaths = model.preservePaths
		if model.revision != "" {
			options.Branch = model.revision
		}
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
	if err := createResizedImage("./models/imageData/portrait.jpg", "./models/imageData/portrait-small.jpg", 448); err != nil {
		panic(err)
	}
}

// createResizedImage writes a bounded-size JPEG copy for the CPU multimodal smoke test.
func createResizedImage(source, dest string, maxDimension int) error {
	input, err := os.Open(source)
	if err != nil {
		return err
	}
	sourceImage, _, decodeErr := image.Decode(input)
	closeErr := input.Close()
	if err := errors.Join(decodeErr, closeErr); err != nil {
		return err
	}
	if maxDimension <= 0 {
		return fmt.Errorf("max image dimension must be greater than zero")
	}

	sourceBounds := sourceImage.Bounds()
	width, height := sourceBounds.Dx(), sourceBounds.Dy()
	resizedWidth, resizedHeight := width, height
	if width >= height && width > maxDimension {
		resizedWidth = maxDimension
		resizedHeight = max(1, (height*maxDimension+width/2)/width)
	} else if height > maxDimension {
		resizedHeight = maxDimension
		resizedWidth = max(1, (width*maxDimension+height/2)/height)
	}

	resized := image.NewRGBA(image.Rect(0, 0, resizedWidth, resizedHeight))
	draw.CatmullRom.Scale(resized, resized.Bounds(), sourceImage, sourceBounds, draw.Over, nil)

	output, err := os.CreateTemp(filepath.Dir(dest), ".portrait-small-*")
	if err != nil {
		return err
	}
	tempPath := output.Name()
	defer os.Remove(tempPath)
	encodeErr := jpeg.Encode(output, resized, &jpeg.Options{Quality: 88})
	closeErr = output.Close()
	if err := errors.Join(encodeErr, closeErr); err != nil {
		return err
	}
	return os.Rename(tempPath, dest)
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
