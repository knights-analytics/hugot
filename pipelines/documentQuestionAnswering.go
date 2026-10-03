package pipelines

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"math"
	"strings"

	"golang.org/x/image/draw"

	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/util/fileutil"
	"github.com/knights-analytics/hugot/util/imageutil"
)

// DocumentQuestionAnsweringPipeline runs a Donut DocVQA encoder/decoder.
// It is not an ORT GenAI conversational pipeline.
type DocumentQuestionAnsweringPipeline struct {
	*backends.BasePipeline
	Decoder         *backends.Model
	DecoderFilename string
	MaxLength       int
	processor       donutProcessor
	eosToken        int64
}

type (
	DocumentQuestionAnsweringResult struct{ Answer string }
	DocumentQuestionAnsweringOutput struct {
		Results []DocumentQuestionAnsweringResult
	}
)

func (o *DocumentQuestionAnsweringOutput) GetOutput() []any {
	out := make([]any, len(o.Results))
	for i, result := range o.Results {
		out[i] = result
	}
	return out
}

type donutProcessor struct {
	Size struct {
		Height int `json:"height"`
		Width  int `json:"width"`
	} `json:"size"`
	Mean          []float32 `json:"image_mean"`
	Std           []float32 `json:"image_std"`
	AlignLongAxis bool      `json:"do_align_long_axis"`
}

func NewDocumentQuestionAnsweringPipeline(ctx context.Context, config DocumentQuestionAnsweringConfig, model *backends.Model) (*DocumentQuestionAnsweringPipeline, error) {
	if model == nil {
		return nil, errors.New("document QA requires a Donut DocVQA model")
	}
	p := &DocumentQuestionAnsweringPipeline{BasePipeline: backends.NewBasePipeline(ctx, config, model), DecoderFilename: "decoder_model_quantized.onnx", MaxLength: 128, eosToken: 2}
	for _, option := range config.Options {
		if err := option(p); err != nil {
			return nil, err
		}
	}
	if err := p.Validate(); err != nil {
		return nil, err
	}
	if err := readNativeConfig(ctx, model.Path, "preprocessor_config.json", &p.processor); err != nil {
		return nil, err
	}
	if p.processor.Size.Height <= 0 || p.processor.Size.Width <= 0 || len(p.processor.Mean) != 3 || len(p.processor.Std) != 3 {
		return nil, errors.New("donut processor requires image size and three-channel normalization")
	}
	for _, std := range p.processor.Std {
		if std <= 0 {
			return nil, errors.New("invalid Donut normalization standard deviation")
		}
	}
	var cfg struct {
		Decoder struct {
			EOS int64 `json:"eos_token_id"`
		} `json:"decoder"`
	}
	if err := readNativeConfig(ctx, model.Path, "config.json", &cfg); err != nil {
		return nil, err
	}
	p.eosToken = cfg.Decoder.EOS
	decoder, err := model.LoadGraph(ctx, p.DecoderFilename)
	if err != nil {
		return nil, err
	}
	p.Decoder = decoder
	if err := requireTensorNames(decoder, []string{"input_ids", "encoder_hidden_states"}, []string{"logits"}); err != nil {
		return nil, err
	}
	return p, nil
}

func (*DocumentQuestionAnsweringPipeline) IsGenerative() bool          { return false }
func (p *DocumentQuestionAnsweringPipeline) GetModel() *backends.Model { return p.Model }
func (p *DocumentQuestionAnsweringPipeline) GetMetadata() backends.PipelineMetadata {
	return nativeMetadata(p.Model)
}

func (p *DocumentQuestionAnsweringPipeline) GetStatistics() backends.PipelineStatistics {
	return nativeStatistics(p.BasePipeline)
}

func (p *DocumentQuestionAnsweringPipeline) Validate() error {
	if p == nil || p.BasePipeline == nil || p.Model == nil {
		return errors.New("document QA requires a Donut DocVQA model")
	}
	if err := requireTensorNames(p.Model, []string{"pixel_values"}, []string{"last_hidden_state"}); err != nil {
		return err
	}
	if p.Model.Tokenizer == nil || p.MaxLength <= 0 {
		return errors.New("document QA requires a tokenizer and positive max length")
	}
	return nil
}

func readNativeConfig(ctx context.Context, path, name string, value any) error {
	data, err := fileutil.ReadFileBytes(ctx, fileutil.PathJoinSafe(path, name))
	if err != nil {
		return err
	}
	return json.Unmarshal(data, value)
}

func (p *DocumentQuestionAnsweringPipeline) RunPipeline(ctx context.Context, inputs []DocumentQuestionAnsweringInput) (*DocumentQuestionAnsweringOutput, error) {
	if len(inputs) == 0 {
		return nil, errors.New("document QA requires at least one document question")
	}
	if err := p.Validate(); err != nil {
		return nil, err
	}
	if p.Decoder == nil {
		return nil, errors.New("document QA decoder is not loaded")
	}
	out := &DocumentQuestionAnsweringOutput{Results: make([]DocumentQuestionAnsweringResult, len(inputs))}
	for i, input := range inputs {
		path := input.DocumentPath
		if path == "" {
			path = input.ImagePath
		}
		if strings.TrimSpace(path) == "" || strings.TrimSpace(input.Question) == "" {
			return nil, errors.New("document QA requires a document image and question")
		}
		if len(input.History) > 0 || (input.Role != "" && input.Role != "user") {
			return nil, errors.New("native document QA does not accept conversation history or roles")
		}
		images, err := imageutil.LoadImagesFromPaths(p.SessionContext, []string{path})
		if err != nil {
			return nil, err
		}
		pixels, err := p.processor.pixels(images[0])
		if err != nil {
			return nil, err
		}
		embeddings, err := runNativeTensors(ctx, p.BasePipeline, p.Model, map[string]backends.Tensor{"pixel_values": pixels})
		if err != nil {
			return nil, err
		}
		prompt := "<s_docvqa><s_question>" + input.Question + "</s_question><s_answer>"
		encoded := p.Model.Tokenizer.GoTokenizer.Tokenizer.EncodeWithAnnotations(prompt)
		ids := make([]int64, 0, len(encoded.IDs)+p.MaxLength)
		for _, id := range encoded.IDs {
			ids = append(ids, int64(id))
		}
		// Donut's decoder prompt must not contain the tokenizer's BOS/EOS wrapper.
		if len(ids) > 0 && ids[0] == 0 {
			ids = ids[1:]
		}
		if len(ids) > 0 && ids[len(ids)-1] == p.eosToken {
			ids = ids[:len(ids)-1]
		}
		if len(ids) == 0 {
			return nil, errors.New("empty document QA prompt")
		}
		var generated []uint32
		finished := false
		for range p.MaxLength {
			decoded, err := runNativeTensors(ctx, p.BasePipeline, p.Decoder, map[string]backends.Tensor{
				"input_ids": {Shape: []int64{1, int64(len(ids))}, Data: ids}, "encoder_hidden_states": embeddings["last_hidden_state"],
			})
			if err != nil {
				return nil, err
			}
			token, err := lastTokenArgmax(decoded["logits"])
			if err != nil {
				return nil, err
			}
			if token == p.eosToken {
				finished = true
				break
			}
			ids = append(ids, token)
			generated = append(generated, uint32(token))
		}
		if !finished {
			return nil, fmt.Errorf("document QA answer exceeded max length %d", p.MaxLength)
		}
		text, err := backends.Decode(generated, p.Model.Tokenizer)
		if err != nil {
			return nil, err
		}
		answer, err := donutAnswer(text)
		if err != nil {
			return nil, err
		}
		out.Results[i] = DocumentQuestionAnsweringResult{Answer: answer}
	}
	return out, nil
}

func lastTokenArgmax(tensor backends.Tensor) (int64, error) {
	data, ok := tensor.Data.([]float32)
	if !ok || len(tensor.Shape) != 3 || tensor.Shape[0] != 1 || tensor.Shape[1] <= 0 || tensor.Shape[2] <= 0 || int64(len(data)) != tensor.Shape[1]*tensor.Shape[2] {
		return 0, errors.New("decoder logits must have shape [1,sequence,vocabulary]")
	}
	vocab := int(tensor.Shape[2])
	row := data[len(data)-vocab:]
	best := 0
	for i, value := range row {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return 0, errors.New("decoder logits are nonfinite")
		}
		if value > row[best] {
			best = i
		}
	}
	return int64(best), nil
}

func donutAnswer(text string) (string, error) {
	if end := strings.Index(text, "</s_answer>"); end >= 0 {
		text = text[:end]
	}
	text = strings.TrimSpace(text)
	if text == "" || strings.Contains(text, "<s_") || strings.Contains(text, "</s_") {
		return "", errors.New("donut returned an invalid answer sequence")
	}
	return text, nil
}

func (p donutProcessor) pixels(img image.Image) (backends.Tensor, error) {
	if img == nil || img.Bounds().Empty() {
		return backends.Tensor{}, errors.New("document image is empty")
	}
	if p.AlignLongAxis {
		return backends.Tensor{}, errors.New("rotating Donut processors are not supported")
	}
	w, h := p.Size.Width, p.Size.Height
	size := img.Bounds().Size()
	// Resize the shortest edge, thumbnail to fit, and center-pad like DonutImageProcessor.
	scale := float64(min(w, h)) / float64(min(size.X, size.Y))
	rw, rh := int(float64(size.X)*scale), int(float64(size.Y)*scale)
	thumbnail := min(1, float64(w)/float64(rw), float64(h)/float64(rh))
	rw, rh = max(1, int(float64(rw)*thumbnail)), max(1, int(float64(rh)*thumbnail))
	dst := image.NewRGBA(image.Rect(0, 0, rw, rh))
	draw.BiLinear.Scale(dst, dst.Bounds(), img, img.Bounds(), draw.Src, nil)
	data := make([]float32, 3*w*h)
	// Padding is applied to normalized pixels, so zero means the channel mean.
	xoff, yoff := (w-rw)/2, (h-rh)/2
	for y := range rh {
		for x := range rw {
			r, g, b, _ := dst.At(x, y).RGBA()
			for c, v := range []uint32{r, g, b} {
				data[c*w*h+(y+yoff)*w+x+xoff] = (float32(v)/65535 - p.Mean[c]) / p.Std[c]
			}
		}
	}
	return backends.Tensor{Shape: []int64{1, 3, int64(h), int64(w)}, Data: data}, nil
}
