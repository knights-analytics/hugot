package pipelines

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"math"
	"sort"
	"strings"

	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/util/imageutil"
)

type nativeVisualQuestionAnswering struct {
	shortestEdge int
	sizeDivisor  int
	resample     int
	mean, std    [3]float32
}

func WithVisualQuestionAnsweringTopK(k int) VisualQuestionAnsweringOption {
	return func(p *VisualQuestionAnsweringPipeline) error {
		if k <= 0 {
			return errors.New("VQA top_k must be positive")
		}
		p.TopK = k
		return nil
	}
}

func newNativeVisualQuestionAnswering(ctx context.Context, model *backends.Model) (*nativeVisualQuestionAnswering, error) {
	if err := requireTensorNames(model, []string{"input_ids", "attention_mask", "token_type_ids", "pixel_values", "pixel_mask"}, []string{"logits"}); err != nil {
		return nil, fmt.Errorf("native VQA requires a ViLT classification graph: %w", err)
	}
	var cfg struct {
		ModelType string `json:"model_type"`
	}
	if err := readNativeConfig(ctx, model.Path, "config.json", &cfg); err != nil {
		return nil, err
	}
	if cfg.ModelType != "vilt" {
		return nil, errors.New("native VQA currently requires model_type vilt")
	}
	var processor struct {
		Size        json.RawMessage `json:"size"`
		SizeDivisor int             `json:"size_divisor"`
		Resample    int             `json:"resample"`
		Mean        []float32       `json:"image_mean"`
		Std         []float32       `json:"image_std"`
		DoResize    *bool           `json:"do_resize"`
		DoNormalize *bool           `json:"do_normalize"`
		DoRescale   *bool           `json:"do_rescale"`
		Scale       *float64        `json:"rescale_factor"`
	}
	if err := readNativeConfig(ctx, model.Path, "preprocessor_config.json", &processor); err != nil {
		return nil, err
	}
	for _, flag := range []*bool{processor.DoResize, processor.DoNormalize, processor.DoRescale} {
		if flag != nil && !*flag {
			return nil, errors.New("native ViLT requires resize, rescale and normalization")
		}
	}
	if processor.Scale != nil && math.Abs(*processor.Scale-1.0/255.0) > 1e-12 {
		return nil, errors.New("native ViLT requires a 1/255 rescale factor")
	}
	native := &nativeVisualQuestionAnswering{sizeDivisor: processor.SizeDivisor, resample: processor.Resample}
	if err := json.Unmarshal(processor.Size, &native.shortestEdge); err != nil {
		var size struct {
			ShortestEdge int `json:"shortest_edge"`
		}
		if err := json.Unmarshal(processor.Size, &size); err != nil {
			return nil, err
		}
		native.shortestEdge = size.ShortestEdge
	}
	if native.shortestEdge <= 0 || native.sizeDivisor <= 0 || len(processor.Mean) != 3 || len(processor.Std) != 3 ||
		(native.resample != 2 && native.resample != 3) {
		return nil, errors.New("unsupported native ViLT image processor")
	}
	copy(native.mean[:], processor.Mean)
	copy(native.std[:], processor.Std)
	for c := range 3 {
		if math.IsNaN(float64(native.mean[c])) || math.IsInf(float64(native.mean[c]), 0) ||
			math.IsNaN(float64(native.std[c])) || math.IsInf(float64(native.std[c]), 0) || native.std[c] <= 0 {
			return nil, errors.New("invalid ViLT normalization parameters")
		}
	}
	return native, nil
}

func (p *VisualQuestionAnsweringPipeline) validateNativeVQA() error {
	if p.BasePipeline == nil || p.Model == nil || p.Model.IsGenerative {
		return errors.New("native VQA requires an ordinary ONNX model")
	}
	if p.Streaming || p.SystemPrompt != "" || p.Temperature != nil || p.TopP != nil || p.Seed != nil {
		return errors.New("native VQA classification does not support chat, streaming or generation options")
	}
	if p.TopK <= 0 || p.MaxLength <= 0 || p.Model.MaxPositionEmbeddings > 0 && p.MaxLength > p.Model.MaxPositionEmbeddings {
		return errors.New("invalid native VQA top_k or question length")
	}
	if len(p.Model.IDLabelMap) == 0 {
		return errors.New("native VQA requires an answer label map")
	}
	for i := range len(p.Model.IDLabelMap) {
		if strings.TrimSpace(p.Model.IDLabelMap[i]) == "" {
			return errors.New("native VQA answer labels must be contiguous and nonempty")
		}
	}
	return requireTensorNames(p.Model, []string{"input_ids", "attention_mask", "token_type_ids", "pixel_values", "pixel_mask"}, []string{"logits"})
}

func (n *nativeVisualQuestionAnswering) imageTensors(img image.Image) (backends.Tensor, backends.Tensor, error) {
	if img == nil || img.Bounds().Empty() {
		return backends.Tensor{}, backends.Tensor{}, errors.New("VQA requires a nonempty image")
	}
	w, h := img.Bounds().Dx(), img.Bounds().Dy()
	scale := float64(n.shortestEdge) / float64(min(w, h))
	newW, newH := float64(w)*scale, float64(h)*scale
	longest := int(float64(n.shortestEdge) * 1333 / 800)
	if max(newW, newH) > float64(longest) {
		scale = float64(longest) / max(newW, newH)
		newW, newH = newW*scale, newH*scale
	}
	w = int(newW+0.5) / n.sizeDivisor * n.sizeDivisor
	h = int(newH+0.5) / n.sizeDivisor * n.sizeDivisor
	resized, err := imageutil.ResizeRGB(img, w, h, n.resample)
	if err != nil {
		return backends.Tensor{}, backends.Tensor{}, err
	}
	values, err := backends.PreprocessImages("NCHW", []image.Image{resized}, nil,
		[]imageutil.NormalizationStep{imageutil.RescaleStep(), imageutil.PixelNormalizationStep(n.mean, n.std)})
	if err != nil {
		return backends.Tensor{}, backends.Tensor{}, err
	}
	data := make([]float32, 0, 3*w*h)
	for _, channel := range values[0] {
		for _, row := range channel {
			data = append(data, row...)
		}
	}
	mask := make([]int64, w*h)
	for i := range mask {
		mask[i] = 1
	}
	// Inputs are inferred individually, so there is no inter-image padding.
	return backends.Tensor{Shape: []int64{1, 3, int64(h), int64(w)}, Data: data},
		backends.Tensor{Shape: []int64{1, int64(h), int64(w)}, Data: mask}, nil
}

func rankVisualAnswers(logits backends.Tensor, labels map[int]string, k int) ([]VisualQuestionAnsweringAnswer, error) {
	values, ok := logits.Data.([]float32)
	if !ok || len(logits.Shape) != 2 || logits.Shape[0] != 1 || logits.Shape[1] != int64(len(labels)) || len(values) != len(labels) || k <= 0 {
		return nil, errors.New("native VQA logits must have shape [1, number of answer labels]")
	}
	answers := make([]VisualQuestionAnsweringAnswer, len(values))
	for i, value := range values {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || labels[i] == "" {
			return nil, errors.New("native VQA requires finite logits and a complete label map")
		}
		answers[i] = VisualQuestionAnsweringAnswer{Answer: labels[i], Score: float32(1 / (1 + math.Exp(-float64(value))))}
	}
	sort.SliceStable(answers, func(i, j int) bool { return answers[i].Score > answers[j].Score })
	return answers[:min(k, len(answers))], nil
}

func (p *VisualQuestionAnsweringPipeline) runNativeVQA(ctx context.Context, inputs []VisualQuestionAnsweringInput) (*QuestionAnsweringTextOutput, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	if len(inputs) == 0 {
		return nil, errors.New("at least one VQA input is required")
	}
	out := &QuestionAnsweringTextOutput{Responses: make([]string, len(inputs)), Answers: make([][]VisualQuestionAnsweringAnswer, len(inputs))}
	for i, input := range inputs {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if input.Role != "" || len(input.History) != 0 {
			return nil, errors.New("native VQA does not support conversational roles or history")
		}
		if strings.TrimSpace(input.ImagePath) == "" || strings.TrimSpace(input.Question) == "" {
			return nil, fmt.Errorf("native VQA input %d requires an image and question", i)
		}
		images, err := imageutil.LoadImagesFromPaths(ctx, []string{input.ImagePath})
		if err != nil {
			return nil, err
		}
		pixels, mask, err := p.native.imageTensors(images[0])
		if err != nil {
			return nil, err
		}
		batch := backends.NewBatch(1)
		if p.Model.Tokenizer == nil {
			return nil, errors.New("native VQA requires a tokenizer")
		}
		backends.TokenizeInputs(batch, p.Model.Tokenizer, []string{input.Question})
		if len(batch.Input) != 1 || len(batch.Input[0].TokenIDs) == 0 || len(batch.Input[0].TokenIDs) > p.MaxLength {
			return nil, errors.New("VQA question exceeds configured token limit or has no tokens")
		}
		tokens := batch.Input[0]
		ids, attention, types := make([]int64, len(tokens.TokenIDs)), make([]int64, len(tokens.TokenIDs)), make([]int64, len(tokens.TokenIDs))
		for j, id := range tokens.TokenIDs {
			ids[j] = int64(id)
			attention[j] = 1
			if j < len(tokens.TypeIDs) {
				types[j] = int64(tokens.TypeIDs[j])
			}
		}
		shape := []int64{1, int64(len(ids))}
		outputs, err := p.Model.RunTensors(ctx, map[string]backends.Tensor{
			"input_ids": {Shape: shape, Data: ids}, "attention_mask": {Shape: shape, Data: attention},
			"token_type_ids": {Shape: shape, Data: types}, "pixel_values": pixels, "pixel_mask": mask,
		})
		if err != nil {
			return nil, err
		}
		out.Answers[i], err = rankVisualAnswers(outputs["logits"], p.Model.IDLabelMap, p.TopK)
		if err != nil {
			return nil, err
		}
		out.Responses[i] = out.Answers[i][0].Answer
	}
	return out, nil
}
