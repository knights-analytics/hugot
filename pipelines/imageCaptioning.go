package pipelines

import (
	"context"
	"errors"
	"fmt"
	"image"
	"math"
	"strings"

	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/util/imageutil"
)

type nativeImageCaptioning struct {
	decoder    *backends.Model
	processor  captionProcessor
	startToken int64
	eosToken   int64
}

type captionProcessor struct {
	Size struct {
		Height int `json:"height"`
		Width  int `json:"width"`
	} `json:"size"`
	Resize     bool      `json:"do_resize"`
	Rescale    bool      `json:"do_rescale"`
	Normalize  bool      `json:"do_normalize"`
	Resample   int       `json:"resample"`
	Factor     float64   `json:"rescale_factor"`
	Mean       []float32 `json:"image_mean"`
	Std        []float32 `json:"image_std"`
	CenterCrop bool      `json:"do_center_crop"`
}

func newNativeImageCaptioning(ctx context.Context, model *backends.Model) (*nativeImageCaptioning, error) {
	if model == nil || model.IsGenerative {
		return nil, errors.New("native captioning requires a ViT-GPT2 encoder model")
	}
	if err := captionGraphContract(model, []string{"pixel_values"}, []string{"last_hidden_state"}); err != nil {
		return nil, err
	}
	var cfg struct {
		ModelType string `json:"model_type"`
		Encoder   struct {
			ModelType string `json:"model_type"`
		} `json:"encoder"`
		Decoder struct {
			ModelType string `json:"model_type"`
		} `json:"decoder"`
		Start *int64 `json:"decoder_start_token_id"`
		EOS   *int64 `json:"eos_token_id"`
	}
	if err := readNativeConfig(ctx, model.Path, "config.json", &cfg); err != nil {
		return nil, err
	}
	if cfg.ModelType != "vision-encoder-decoder" || cfg.Encoder.ModelType != "vit" || cfg.Decoder.ModelType != "gpt2" || cfg.Start == nil || cfg.EOS == nil || *cfg.Start < 0 || *cfg.Start >= 50257 || *cfg.EOS < 0 || *cfg.EOS >= 50257 {
		return nil, errors.New("captioning requires a ViT-GPT2 configuration with valid decoder start and EOS tokens")
	}
	n := &nativeImageCaptioning{startToken: *cfg.Start, eosToken: *cfg.EOS}
	if err := readNativeConfig(ctx, model.Path, "preprocessor_config.json", &n.processor); err != nil {
		return nil, err
	}
	if err := n.processor.validate(); err != nil {
		return nil, err
	}
	decoder, err := model.LoadGraph(ctx, "decoder_model_quantized.onnx")
	if err != nil {
		return nil, err
	}
	if err := captionGraphContract(decoder, []string{"input_ids", "encoder_hidden_states"}, []string{"logits"}); err != nil {
		return nil, err
	}
	n.decoder = decoder
	return n, nil
}

func captionGraphContract(model *backends.Model, inputs, outputs []string) error {
	if model == nil || len(model.InputsMeta) != len(inputs) {
		return errors.New("captioning graph has unsupported inputs (cached/past-only decoding is not supported)")
	}
	return requireTensorNames(model, inputs, outputs)
}

func (p captionProcessor) validate() error {
	if p.Size.Height != 224 || p.Size.Width != 224 || !p.Resize || !p.Rescale || !p.Normalize || p.Resample != 2 || p.CenterCrop || math.IsNaN(p.Factor) || math.IsInf(p.Factor, 0) || math.Abs(p.Factor-1.0/255) > 1e-10 || len(p.Mean) != 3 || len(p.Std) != 3 {
		return errors.New("unsupported ViT captioning processor")
	}
	for i := range 3 {
		if p.Mean[i] != 0.5 || p.Std[i] != 0.5 {
			return errors.New("unsupported ViT captioning normalization")
		}
	}
	return nil
}

func (p captionProcessor) pixels(img image.Image) (backends.Tensor, error) {
	if err := p.validate(); err != nil {
		return backends.Tensor{}, err
	}
	if img == nil || img.Bounds().Empty() {
		return backends.Tensor{}, errors.New("caption image is empty")
	}
	resized, err := imageutil.ResizeRGB(img, p.Size.Width, p.Size.Height, p.Resample)
	if err != nil {
		return backends.Tensor{}, err
	}
	w, h := p.Size.Width, p.Size.Height
	data := make([]float32, 3*w*h)
	for y := range h {
		for x := range w {
			r, g, b, _ := resized.At(x, y).RGBA()
			for c, value := range []uint32{r, g, b} {
				data[c*w*h+y*w+x] = (float32(float64(value>>8)*p.Factor) - p.Mean[c]) / p.Std[c]
			}
		}
	}
	return backends.Tensor{Shape: []int64{1, 3, int64(h), int64(w)}, Data: data}, nil
}

func (p *ImageToTextPipeline) validateNativeCaption() error {
	if p == nil || p.multimodalGeneration == nil || p.BasePipeline == nil || p.Model == nil || p.native == nil || p.native.decoder == nil || p.Model.Tokenizer == nil {
		return errors.New("native captioning requires an encoder, decoder, and tokenizer")
	}
	if p.MaxLength <= 0 || p.MaxLength > 1023 {
		return errors.New("native captioning max length must be between 1 and 1023 generated tokens")
	}
	if p.Streaming || p.SystemPrompt != "" || p.Temperature != nil || p.TopP != nil || p.Seed != nil {
		return errors.New("native captioning supports only non-streaming greedy generation without chat options")
	}
	if err := captionGraphContract(p.Model, []string{"pixel_values"}, []string{"last_hidden_state"}); err != nil {
		return err
	}
	if err := captionGraphContract(p.native.decoder, []string{"input_ids", "encoder_hidden_states"}, []string{"logits"}); err != nil {
		return err
	}
	return p.native.processor.validate()
}

func validateCaptionInput(input ImageTextPrompt) error {
	if strings.TrimSpace(input.ImagePath) == "" {
		return errors.New("captioning requires an image")
	}
	if input.Prompt != "" || input.Role != "" || len(input.History) != 0 {
		return errors.New("native captioning does not accept prompts, roles, or conversation history")
	}
	return nil
}

func (p *ImageToTextPipeline) runNativeCaption(ctx context.Context, inputs []ImageTextPrompt) (*MultimodalTextOutput, error) {
	if err := p.validateNativeCaption(); err != nil {
		return nil, err
	}
	if len(inputs) == 0 {
		return nil, errors.New("captioning requires at least one image")
	}
	for _, input := range inputs {
		if err := validateCaptionInput(input); err != nil {
			return nil, err
		}
	}
	out := &MultimodalTextOutput{Responses: make([]string, len(inputs))}
	for i, input := range inputs {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		images, err := imageutil.LoadImagesFromPaths(ctx, []string{input.ImagePath})
		if err != nil {
			return nil, err
		}
		pixels, err := p.native.processor.pixels(images[0])
		if err != nil {
			return nil, err
		}
		embeddings, err := runNativeTensors(ctx, p.BasePipeline, p.Model, map[string]backends.Tensor{"pixel_values": pixels})
		if err != nil {
			return nil, err
		}
		infer := func(ctx context.Context, tensors map[string]backends.Tensor) (map[string]backends.Tensor, error) {
			return runNativeTensors(ctx, p.BasePipeline, p.native.decoder, tensors)
		}
		tokens, err := generateCaptionTokens(ctx, embeddings["last_hidden_state"], p.native.startToken, p.native.eosToken, p.MaxLength, infer)
		if err != nil {
			return nil, err
		}
		text, err := backends.Decode(tokens, p.Model.Tokenizer)
		if err != nil {
			return nil, err
		}
		out.Responses[i] = strings.TrimSpace(text)
	}
	return out, nil
}

type captionInference func(context.Context, map[string]backends.Tensor) (map[string]backends.Tensor, error)

func generateCaptionTokens(ctx context.Context, embeddings backends.Tensor, start, eos int64, maxLength int, infer captionInference) ([]uint32, error) {
	if maxLength <= 0 || maxLength > 1023 || start < 0 || eos < 0 || infer == nil {
		return nil, errors.New("invalid caption generation parameters")
	}
	ids := []int64{start}
	generated := make([]uint32, 0, maxLength)
	for range maxLength {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		outputs, err := infer(ctx, map[string]backends.Tensor{
			"input_ids":             {Shape: []int64{1, int64(len(ids))}, Data: ids},
			"encoder_hidden_states": embeddings,
		})
		if err != nil {
			return nil, fmt.Errorf("caption decoder: %w", err)
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		token, err := lastTokenArgmax(outputs["logits"])
		if err != nil {
			return nil, err
		}
		if token == eos {
			break
		}
		ids = append(ids, token)
		generated = append(generated, uint32(token))
	}
	return generated, nil
}
