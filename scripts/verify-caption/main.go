package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"time"

	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/util/fileutil"
)

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	ctx = fileutil.WithFileSystem(ctx, nil)
	opts := options.Defaults()
	opts.Backend = options.BackendORT
	check(backends.InitializeORT(opts))
	defer func() { check(opts.Destroy()) }()
	model, err := backends.LoadModel(ctx, "models/Xenova_vit-gpt2-image-captioning", "encoder_model_quantized.onnx", opts, false)
	check(err)
	defer func() { check(model.Close()) }()
	decoder, err := model.LoadGraph(ctx, "decoder_model_quantized.onnx")
	check(err)
	report := map[string]any{
		"model_id": "Xenova/vit-gpt2-image-captioning",
		"revision": "215b4edcb7ec1fad5905a18a03f7b2007f6fabd0",
		"encoder_inputs": model.InputsMeta, "encoder_outputs": model.OutputsMeta,
		"decoder_inputs": decoder.InputsMeta, "decoder_outputs": decoder.OutputsMeta,
	}
	check(json.NewEncoder(os.Stdout).Encode(report))
}

func check(err error) {
	if err != nil {
		panic(fmt.Sprintf("caption graph verification: %v", err))
	}
}