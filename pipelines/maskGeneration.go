package pipelines

import (
	"context"
	"errors"
	"fmt"
	"image"
	"math"
	"sort"
	"sync/atomic"
	"time"

	"golang.org/x/image/draw"

	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/util/imageutil"
	"github.com/knights-analytics/hugot/util/safeconv"
)

// MaskGenerationPipeline generates automatic SAM proposals, not semantic classes.
type MaskGenerationPipeline struct {
	*backends.BasePipeline
	Decoder                                                                  *backends.Model
	DecoderFilename                                                          string
	PointsPerSide, PointsPerBatch                                            int
	PredictedIOUThreshold, StabilityThreshold, StabilityOffset, NMSThreshold float32
}

type MaskGenerationMask struct {
	Mask  [][]bool
	Score float32
	// PredictedIOU is SAM's unbounded regression output; Score is clipped to [0,1].
	PredictedIOU   float32
	StabilityScore float32
	Box            [4]int
	Area           int
}

type MaskGenerationResult struct {
	Width, Height int
	Masks         []MaskGenerationMask
}

type MaskGenerationOutput struct{ Results []MaskGenerationResult }

func (o *MaskGenerationOutput) GetOutput() []any {
	out := make([]any, len(o.Results))
	for i, result := range o.Results {
		out[i] = result
	}
	return out
}

func NewMaskGenerationPipeline(ctx context.Context, config backends.PipelineConfig[*MaskGenerationPipeline], model *backends.Model) (*MaskGenerationPipeline, error) {
	if model == nil {
		return nil, errors.New("mask generation requires a SAM encoder")
	}
	p := &MaskGenerationPipeline{BasePipeline: backends.NewBasePipeline(ctx, config, model), DecoderFilename: "prompt_encoder_mask_decoder.onnx", PointsPerSide: 32, PointsPerBatch: 64, PredictedIOUThreshold: 0.88, StabilityThreshold: 0.95, StabilityOffset: 1, NMSThreshold: 0.7}
	for _, option := range config.Options {
		if err := option(p); err != nil {
			return nil, err
		}
	}
	if err := p.Validate(); err != nil {
		return nil, err
	}
	if p.Decoder == nil {
		decoder, err := model.LoadGraph(ctx, p.DecoderFilename)
		if err != nil {
			return nil, fmt.Errorf("load SAM decoder: %w", err)
		}
		p.Decoder = decoder
	}
	if err := requireTensorNames(p.Decoder, []string{"image_embeddings", "image_positional_embeddings", "input_points", "input_labels"}, []string{"pred_masks", "iou_scores"}); err != nil {
		return nil, err
	}
	return p, nil
}

func (*MaskGenerationPipeline) IsGenerative() bool          { return false }
func (p *MaskGenerationPipeline) GetModel() *backends.Model { return p.Model }
func (p *MaskGenerationPipeline) GetMetadata() backends.PipelineMetadata {
	return nativeMetadata(p.Model)
}
func (p *MaskGenerationPipeline) GetStatistics() backends.PipelineStatistics {
	return nativeStatistics(p.BasePipeline)
}
func (p *MaskGenerationPipeline) Validate() error {
	if p == nil || p.BasePipeline == nil || p.Model == nil {
		return errors.New("mask generation requires a SAM encoder")
	}
	if err := requireTensorNames(p.Model, []string{"pixel_values"}, []string{"image_embeddings", "image_positional_embeddings"}); err != nil {
		return err
	}
	if p.PointsPerSide <= 0 || p.PointsPerBatch <= 0 {
		return errors.New("SAM point grid and prompt batch must be positive")
	}
	for _, value := range []float32{p.PredictedIOUThreshold, p.StabilityThreshold, p.NMSThreshold} {
		if math.IsNaN(float64(value)) || value < 0 || value > 1 {
			return errors.New("SAM thresholds must be in [0,1]")
		}
	}
	if p.StabilityOffset <= 0 || math.IsNaN(float64(p.StabilityOffset)) || math.IsInf(float64(p.StabilityOffset), 0) {
		return errors.New("SAM stability offset must be finite and positive")
	}
	return nil
}

func (p *MaskGenerationPipeline) Run(ctx context.Context, inputs []string) (backends.PipelineBatchOutput, error) {
	return p.RunPipeline(ctx, inputs)
}
func (p *MaskGenerationPipeline) RunPipeline(ctx context.Context, inputs []string) (*MaskGenerationOutput, error) {
	images, err := imageutil.LoadImagesFromPaths(p.SessionContext, inputs)
	if err != nil {
		return nil, err
	}
	return p.RunWithImages(ctx, images)
}

func (p *MaskGenerationPipeline) RunWithImages(ctx context.Context, images []image.Image) (*MaskGenerationOutput, error) {
	if len(images) == 0 {
		return nil, errors.New("mask generation requires at least one image")
	}
	if err := p.Validate(); err != nil {
		return nil, err
	}
	if p.Decoder == nil {
		return nil, errors.New("SAM decoder is not loaded")
	}
	out := &MaskGenerationOutput{Results: make([]MaskGenerationResult, len(images))}
	for i, img := range images {
		if img == nil || img.Bounds().Empty() {
			return nil, errors.New("SAM requires a nonempty image")
		}
		size := img.Bounds().Size()
		pixels, resized := samPixels(img, 1024)
		embeddings, err := runNativeTensors(ctx, p.BasePipeline, p.Model, map[string]backends.Tensor{"pixel_values": pixels})
		if err != nil {
			return nil, err
		}
		points := samPointGrid(p.PointsPerSide, resized)
		var proposals []MaskGenerationMask
		for start := 0; start < len(points)/2; start += p.PointsPerBatch {
			n := min(p.PointsPerBatch, len(points)/2-start)
			labels := make([]int64, n)
			for j := range labels {
				labels[j] = 1
			}
			decoded, err := runNativeTensors(ctx, p.BasePipeline, p.Decoder, map[string]backends.Tensor{
				"image_embeddings": embeddings["image_embeddings"], "image_positional_embeddings": embeddings["image_positional_embeddings"],
				"input_points": {Shape: []int64{1, int64(n), 1, 2}, Data: points[start*2 : (start+n)*2]},
				"input_labels": {Shape: []int64{1, int64(n), 1}, Data: labels},
			})
			if err != nil {
				return nil, err
			}
			masks, err := p.samProposals(decoded["pred_masks"], decoded["iou_scores"], size, resized)
			if err != nil {
				return nil, err
			}
			proposals = append(proposals, masks...)
		}
		out.Results[i] = MaskGenerationResult{Width: size.X, Height: size.Y, Masks: samNMS(proposals, p.NMSThreshold)}
	}
	return out, nil
}

func samPointGrid(side int, resized image.Point) []float32 {
	points := make([]float32, 0, side*side*2)
	for y := range side {
		for x := range side {
			points = append(points, (float32(x)+0.5)*float32(resized.X)/float32(side), (float32(y)+0.5)*float32(resized.Y)/float32(side))
		}
	}
	return points
}

func samPixels(img image.Image, target int) (backends.Tensor, image.Point) {
	size := img.Bounds().Size()
	scale := float64(target) / float64(max(size.X, size.Y))
	resized := image.Pt(int(float64(size.X)*scale+0.5), int(float64(size.Y)*scale+0.5))
	dst := image.NewRGBA(image.Rect(0, 0, resized.X, resized.Y))
	draw.BiLinear.Scale(dst, dst.Bounds(), img, img.Bounds(), draw.Src, nil)
	data := make([]float32, 3*target*target)
	mean, std := [3]float32{0.485, 0.456, 0.406}, [3]float32{0.229, 0.224, 0.225}
	for y := range resized.Y {
		for x := range resized.X {
			r, g, b, _ := dst.At(x, y).RGBA()
			for c, value := range []uint32{r, g, b} {
				data[c*target*target+y*target+x] = (float32(value)/65535 - mean[c]) / std[c]
			}
		}
	}
	return backends.Tensor{Shape: []int64{1, 3, int64(target), int64(target)}, Data: data}, resized
}

// samProposals upsamples low-resolution logits to the padded encoder size,
// removes padding, then resizes to source coordinates before thresholding.
func (p *MaskGenerationPipeline) samProposals(masks, scores backends.Tensor, size, resized image.Point) ([]MaskGenerationMask, error) {
	logits, ok := masks.Data.([]float32)
	quality, scoresOK := scores.Data.([]float32)
	if !ok || !scoresOK || len(masks.Shape) != 5 || masks.Shape[0] != 1 || len(scores.Shape) != 3 || scores.Shape[0] != 1 || scores.Shape[1] != masks.Shape[1] || scores.Shape[2] != masks.Shape[2] {
		return nil, errors.New("SAM outputs require [1,points,masks,height,width] logits and [1,points,masks] scores")
	}
	h, w := int(masks.Shape[3]), int(masks.Shape[4])
	if h <= 0 || w <= 0 || len(logits) != len(quality)*h*w {
		return nil, errors.New("invalid SAM mask shape")
	}
	var out []MaskGenerationMask
	for i, score := range quality {
		if math.IsNaN(float64(score)) || math.IsInf(float64(score), 0) || score <= p.PredictedIOUThreshold {
			continue
		}
		low := logits[i*h*w : (i+1)*h*w]
		var intersection, union int
		mask := make([][]bool, size.Y)
		box := [4]int{size.X, size.Y, 0, 0}
		area := 0
		for y := range size.Y {
			mask[y] = make([]bool, size.X)
			for x := range size.X {
				// Align-corners=false bilinear interpolation in each resize stage.
				value := samSourceLogit(low, w, h, x, y, size, resized)
				if value > p.StabilityOffset {
					intersection++
				}
				if value > -p.StabilityOffset {
					union++
				}
				if value > 0 {
					mask[y][x] = true
					area++
					box[0] = min(box[0], x)
					box[1] = min(box[1], y)
					box[2] = max(box[2], x)
					box[3] = max(box[3], y)
				}
			}
		}
		if union == 0 {
			continue
		}
		stability := float32(intersection) / float32(union)
		if area > 0 && stability >= p.StabilityThreshold {
			out = append(out, MaskGenerationMask{Mask: mask, Score: min(1, max(0, score)), PredictedIOU: score, StabilityScore: stability, Box: box, Area: area})
		}
	}
	return out, nil
}

func samSourceLogit(low []float32, w, h, x, y int, size, resized image.Point) float32 {
	sx := (float64(x)+0.5)*float64(resized.X)/float64(size.X) - 0.5
	sy := (float64(y)+0.5)*float64(resized.Y)/float64(size.Y) - 0.5
	x0, y0 := int(math.Floor(sx)), int(math.Floor(sy))
	fx, fy := float32(sx-float64(x0)), float32(sy-float64(y0))
	at := func(px, py int) float32 {
		return bilinearLogit(low, w, h, (float64(max(0, min(resized.X-1, px)))+0.5)*float64(w)/1024-0.5, (float64(max(0, min(resized.Y-1, py)))+0.5)*float64(h)/1024-0.5)
	}
	return (at(x0, y0)*(1-fx)+at(x0+1, y0)*fx)*(1-fy) + (at(x0, y0+1)*(1-fx)+at(x0+1, y0+1)*fx)*fy
}

func bilinearLogit(data []float32, w, h int, x, y float64) float32 {
	x = max(0, min(float64(w-1), x))
	y = max(0, min(float64(h-1), y))
	x0, y0 := int(x), int(y)
	x1, y1 := min(x0+1, w-1), min(y0+1, h-1)
	fx, fy := float32(x-float64(x0)), float32(y-float64(y0))
	return (data[y0*w+x0]*(1-fx)+data[y0*w+x1]*fx)*(1-fy) + (data[y1*w+x0]*(1-fx)+data[y1*w+x1]*fx)*fy
}

func samNMS(masks []MaskGenerationMask, threshold float32) []MaskGenerationMask {
	sort.SliceStable(masks, func(i, j int) bool {
		if masks[i].PredictedIOU != masks[j].PredictedIOU {
			return masks[i].PredictedIOU > masks[j].PredictedIOU
		}
		return masks[i].Score > masks[j].Score
	})
	kept := make([]MaskGenerationMask, 0, len(masks))
	for _, mask := range masks {
		keep := true
		for _, previous := range kept {
			if samBoxIOU(mask.Box, previous.Box) > threshold {
				keep = false
				break
			}
		}
		if keep {
			kept = append(kept, mask)
		}
	}
	return kept
}

func samBoxIOU(a, b [4]int) float32 {
	intersection := max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
	union := (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
	if union <= 0 {
		return 0
	}
	return float32(intersection) / float32(union)
}

func requireTensorNames(model *backends.Model, inputs, outputs []string) error {
	if model == nil || model.IsGenerative {
		return errors.New("task requires a non-generative ONNX model")
	}
	for _, group := range []struct {
		names []string
		meta  []backends.InputOutputInfo
	}{{inputs, model.InputsMeta}, {outputs, model.OutputsMeta}} {
		for _, name := range group.names {
			found := false
			for _, info := range group.meta {
				if info.Name == name {
					found = true
					break
				}
			}
			if !found {
				return fmt.Errorf("task model is missing tensor %q", name)
			}
		}
	}
	return nil
}

func nativeMetadata(model *backends.Model) backends.PipelineMetadata {
	if model == nil {
		return backends.PipelineMetadata{}
	}
	out := make([]backends.OutputInfo, len(model.OutputsMeta))
	for i, info := range model.OutputsMeta {
		out[i] = backends.OutputInfo{Name: info.Name, Dimensions: info.Dimensions}
	}
	return backends.PipelineMetadata{OutputsInfo: out}
}
func nativeStatistics(p *backends.BasePipeline) backends.PipelineStatistics {
	out := backends.PipelineStatistics{}
	out.ComputeOnnxStatistics(p.ONNXTimings)
	out.ComputeTokenizerStatistics(p.TokenizerTimings)
	return out
}
func runNativeTensors(ctx context.Context, p *backends.BasePipeline, model *backends.Model, inputs map[string]backends.Tensor) (map[string]backends.Tensor, error) {
	if p.SessionContext != nil {
		if err := p.SessionContext.Err(); err != nil {
			return nil, err
		}
	}
	start := time.Now()
	out, err := model.RunTensors(ctx, inputs)
	atomic.AddUint64(&p.ONNXTimings.NumCalls, 1)
	atomic.AddUint64(&p.ONNXTimings.TotalNS, safeconv.DurationToU64(time.Since(start)))
	return out, err
}
