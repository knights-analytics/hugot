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

const (
	groundedShortestEdge = 800
	groundedLongestEdge  = 1333
	// Grounding DINO pads the class axis to max_text_len. The last index is that pad.
	groundedClassWidth = 256
)

// GroundedObjectDetectionPipeline runs a caption-prompted detector such as
// Grounding DINO. One caption is encoded with the image. Labels come from the
// tokens whose class probability clears the text threshold.
type GroundedObjectDetectionPipeline struct {
	*backends.BasePipeline
	BoxThreshold  float32
	TextThreshold float32
}

func WithGroundedThresholds(box, text float32) backends.PipelineOption[*GroundedObjectDetectionPipeline] {
	return func(p *GroundedObjectDetectionPipeline) error {
		if !groundedThresholdOK(box) || !groundedThresholdOK(text) {
			return errors.New("grounded detection thresholds must be in [0,1]")
		}
		p.BoxThreshold = box
		p.TextThreshold = text
		return nil
	}
}

func NewGroundedObjectDetectionPipeline(ctx context.Context, config backends.PipelineConfig[*GroundedObjectDetectionPipeline], model *backends.Model) (*GroundedObjectDetectionPipeline, error) {
	p := &GroundedObjectDetectionPipeline{BasePipeline: backends.NewBasePipeline(ctx, config, model), BoxThreshold: 0.25, TextThreshold: 0.25}
	for _, option := range config.Options {
		if err := option(p); err != nil {
			return nil, err
		}
	}
	if err := p.Validate(); err != nil {
		return nil, err
	}
	return p, nil
}

func (*GroundedObjectDetectionPipeline) IsGenerative() bool { return false }

func (p *GroundedObjectDetectionPipeline) GetModel() *backends.Model {
	if p == nil || p.BasePipeline == nil {
		return nil
	}
	return p.Model
}

func (p *GroundedObjectDetectionPipeline) GetMetadata() backends.PipelineMetadata {
	return nativeMetadata(p.GetModel())
}

func (p *GroundedObjectDetectionPipeline) GetStatistics() backends.PipelineStatistics {
	if p == nil || p.BasePipeline == nil {
		return backends.PipelineStatistics{}
	}
	return nativeStatistics(p.BasePipeline)
}

func (p *GroundedObjectDetectionPipeline) Validate() error {
	if p == nil || p.BasePipeline == nil || p.Model == nil {
		return errors.New("grounded object detection requires a model")
	}
	if p.Model.Tokenizer == nil {
		return errors.New("grounded object detection requires a tokenizer")
	}
	if !groundedThresholdOK(p.BoxThreshold) || !groundedThresholdOK(p.TextThreshold) {
		return errors.New("grounded detection thresholds must be in [0,1]")
	}
	return requireTensorNames(p.Model,
		[]string{"pixel_values", "pixel_mask", "input_ids", "attention_mask", "token_type_ids"},
		[]string{"logits", "pred_boxes"})
}

// Run is unavailable because []string cannot carry the caption. Use
// RunWithCaption.
func (p *GroundedObjectDetectionPipeline) Run(context.Context, []string) (backends.PipelineBatchOutput, error) {
	return nil, errors.New("grounded object detection requires a caption; use RunWithCaption")
}

func (p *GroundedObjectDetectionPipeline) RunWithCaption(ctx context.Context, inputs []string, caption string) (*ObjectDetectionOutput, error) {
	images, err := imageutil.LoadImagesFromPaths(ctx, inputs)
	if err != nil {
		return nil, fmt.Errorf("failed to load images: %w", err)
	}
	return p.RunWithImagesAndCaption(ctx, images, caption)
}

func (p *GroundedObjectDetectionPipeline) RunWithImagesAndCaption(ctx context.Context, images []image.Image, caption string) (*ObjectDetectionOutput, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	if len(images) == 0 {
		return nil, errors.New("grounded object detection requires at least one image")
	}
	if strings.TrimSpace(caption) == "" {
		return nil, errors.New("grounded object detection requires a caption")
	}
	text, err := groundedCaptionTokens(p.Model.Tokenizer, caption)
	if err != nil {
		return nil, err
	}
	out := &ObjectDetectionOutput{Detections: make([][]Detection, len(images))}
	for i, img := range images {
		if img == nil || img.Bounds().Empty() {
			return nil, errors.New("grounded object detection received an empty image")
		}
		detections, err := p.detect(ctx, img, text)
		if err != nil {
			return nil, err
		}
		out.Detections[i] = detections
	}
	return out, nil
}

type groundedCaption struct {
	ids, attention, types []int64
}

func groundedCaptionTokens(tokenizer *backends.Tokenizer, caption string) (groundedCaption, error) {
	batch := backends.NewBatch(1)
	backends.TokenizeInputs(batch, tokenizer, []string{caption})
	if len(batch.Input) != 1 || len(batch.Input[0].TokenIDs) == 0 {
		return groundedCaption{}, errors.New("grounded object detection caption produced no tokens")
	}
	tokens := batch.Input[0]
	text := groundedCaption{
		ids:       make([]int64, len(tokens.TokenIDs)),
		attention: make([]int64, len(tokens.TokenIDs)),
		types:     make([]int64, len(tokens.TokenIDs)),
	}
	for i, id := range tokens.TokenIDs {
		text.ids[i] = int64(id)
		text.attention[i] = 1
		if i < len(tokens.TypeIDs) {
			text.types[i] = int64(tokens.TypeIDs[i])
		}
	}
	return text, nil
}

func (p *GroundedObjectDetectionPipeline) detect(ctx context.Context, img image.Image, text groundedCaption) ([]Detection, error) {
	pixels, mask, err := groundedImageTensors(p.Model, img)
	if err != nil {
		return nil, err
	}
	shape := []int64{1, int64(len(text.ids))}
	outputs, err := runNativeTensors(ctx, p.BasePipeline, p.Model, map[string]backends.Tensor{
		"pixel_values":   pixels,
		"pixel_mask":     mask,
		"input_ids":      {Shape: shape, Data: text.ids},
		"attention_mask": {Shape: shape, Data: text.attention},
		"token_type_ids": {Shape: shape, Data: text.types},
	})
	if err != nil {
		return nil, err
	}
	logits, err := groundedRows(outputs["logits"], groundedClassWidth)
	if err != nil {
		return nil, fmt.Errorf("grounded detection logits: %w", err)
	}
	boxes, err := groundedRows(outputs["pred_boxes"], 4)
	if err != nil {
		return nil, fmt.Errorf("grounded detection boxes: %w", err)
	}
	return groundedDetections(logits, boxes, text.ids, img.Bounds().Dx(), img.Bounds().Dy(), p.BoxThreshold, p.TextThreshold, func(tokenIDs []int64) (string, error) {
		decoded := make([]uint32, len(tokenIDs))
		for i, id := range tokenIDs {
			if id < 0 {
				return "", errors.New("grounded detection label token is negative")
			}
			decoded[i] = uint32(id)
		}
		return backends.Decode(decoded, p.Model.Tokenizer)
	})
}

func groundedThresholdOK(value float32) bool {
	return !math.IsNaN(float64(value)) && !math.IsInf(float64(value), 0) && value >= 0 && value <= 1
}

// groundedCanvas matches transformers get_size_with_aspect_ratio: shortest edge
// 800, longest at most 1333. The capped short edge uses Python round (half to
// even). An image whose short edge is already that size stays unchanged.
func groundedCanvas(width, height int) (int, int, error) {
	if width <= 0 || height <= 0 {
		return 0, 0, errors.New("grounded detection requires a nonempty image")
	}
	size := groundedShortestEdge
	raw := 0.0
	capped := false
	minSide := float64(min(width, height))
	maxSide := float64(max(width, height))
	if maxSide/minSide*float64(groundedShortestEdge) > groundedLongestEdge {
		raw = float64(groundedLongestEdge) * minSide / maxSide
		capped = true
		size = int(math.RoundToEven(raw))
	}
	if (height <= width && height == size) || (width <= height && width == size) {
		return width, height, nil
	}
	edge := func(along, across int) int {
		if capped {
			return int(raw * float64(along) / float64(across))
		}
		return int(float64(size) * float64(along) / float64(across))
	}
	if width < height {
		return size, edge(height, width), nil
	}
	return edge(width, height), size, nil
}

func groundedImageTensors(model *backends.Model, img image.Image) (backends.Tensor, backends.Tensor, error) {
	canvasW, canvasH, err := groundedCanvas(img.Bounds().Dx(), img.Bounds().Dy())
	if err != nil {
		return backends.Tensor{}, backends.Tensor{}, err
	}
	if err := groundedSpatial(model, canvasH, canvasW); err != nil {
		return backends.Tensor{}, backends.Tensor{}, err
	}
	resized, err := imageutil.ResizeRGB(img, canvasW, canvasH, 2)
	if err != nil {
		return backends.Tensor{}, backends.Tensor{}, err
	}
	values, err := backends.PreprocessImages("NCHW", []image.Image{resized}, nil, []imageutil.NormalizationStep{
		imageutil.RescaleStep(),
		imageutil.ImagenetPixelNormalizationStep(),
	})
	if err != nil {
		return backends.Tensor{}, backends.Tensor{}, err
	}
	data := make([]float32, 0, 3*canvasW*canvasH)
	for _, channel := range values[0] {
		for _, row := range channel {
			data = append(data, row...)
		}
	}
	mask := make([]bool, canvasW*canvasH)
	for i := range mask {
		mask[i] = true
	}
	return backends.Tensor{Shape: []int64{1, 3, int64(canvasH), int64(canvasW)}, Data: data},
		backends.Tensor{Shape: []int64{1, int64(canvasH), int64(canvasW)}, Data: mask}, nil
}

func groundedSpatial(model *backends.Model, height, width int) error {
	for _, input := range model.InputsMeta {
		var h, w int64
		switch input.Name {
		case "pixel_values":
			if len(input.Dimensions) != 4 {
				continue
			}
			h, w = input.Dimensions[2], input.Dimensions[3]
		case "pixel_mask":
			if len(input.Dimensions) != 3 {
				continue
			}
			h, w = input.Dimensions[1], input.Dimensions[2]
		default:
			continue
		}
		if h >= 0 && int(h) != height {
			return fmt.Errorf("grounded detection model expects height %d, image resizes to %d", h, height)
		}
		if w >= 0 && int(w) != width {
			return fmt.Errorf("grounded detection model expects width %d, image resizes to %d", w, width)
		}
	}
	return nil
}

func groundedRows(tensor backends.Tensor, width int) ([][]float32, error) {
	data, ok := tensor.Data.([]float32)
	if !ok || len(tensor.Shape) != 3 || tensor.Shape[0] != 1 || width <= 0 || tensor.Shape[2] != int64(width) {
		return nil, fmt.Errorf("expected shape [1, queries, %d]", width)
	}
	queries := int(tensor.Shape[1])
	if queries < 0 || len(data) != queries*width {
		return nil, errors.New("tensor length does not match its shape")
	}
	rows := make([][]float32, queries)
	for i := range queries {
		rows[i] = data[i*width : (i+1)*width]
	}
	return rows, nil
}

func groundedDetections(logits, boxes [][]float32, inputIDs []int64, width, height int, boxThreshold, textThreshold float32, decode func([]int64) (string, error)) ([]Detection, error) {
	if len(logits) != len(boxes) || width <= 0 || height <= 0 {
		return nil, errors.New("grounded detection outputs must cover the same queries as the source image")
	}
	detections := make([]Detection, 0)
	for i, row := range logits {
		if len(boxes[i]) != 4 || len(row) == 0 {
			return nil, errors.New("grounded detection query is missing a box or class logits")
		}
		score := groundedMaxScore(row)
		if score <= boxThreshold {
			continue
		}
		cx, cy, bw, bh := boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
		label, err := decode(groundedPhrase(row, inputIDs, textThreshold))
		if err != nil {
			return nil, err
		}
		detections = append(detections, Detection{
			Label: strings.TrimSpace(label),
			Score: score,
			Box: [4]float32{
				(cx - bw/2) * float32(width),
				(cy - bh/2) * float32(height),
				(cx + bw/2) * float32(width),
				(cy + bh/2) * float32(height),
			},
		})
	}
	return detections, nil
}

func groundedMaxScore(logits []float32) float32 {
	score := float32(math.Inf(-1))
	for _, logit := range logits {
		if value := groundedSigmoid(logit); value > score {
			score = value
		}
	}
	if math.IsInf(float64(score), -1) {
		return 0
	}
	return score
}

func groundedPhrase(logits []float32, inputIDs []int64, textThreshold float32) []int64 {
	// Class axis 0 and the final padded position are not phrase tokens.
	limit := min(len(inputIDs), len(logits)-1)
	var ids []int64
	for i := 1; i < limit; i++ {
		if groundedSigmoid(logits[i]) > textThreshold {
			ids = append(ids, inputIDs[i])
		}
	}
	return ids
}

func groundedSigmoid(value float32) float32 {
	if math.IsNaN(float64(value)) {
		return float32(math.NaN())
	}
	if value >= 0 {
		z := math.Exp(float64(-value))
		return float32(1 / (1 + z))
	}
	z := math.Exp(float64(value))
	return float32(z / (1 + z))
}
