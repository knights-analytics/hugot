package pipelines

import (
	"errors"
	"fmt"
	"image"
	_ "image/jpeg" // add JPEG decoding support
	_ "image/png"  // add PNG decoding support
	"math"
	"sort"
	"strings"
	"sync/atomic"
	"time"

	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/util/imageutil"
	"github.com/knights-analytics/hugot/util/safeconv"
)

// ObjectDetectionPipeline implements a Hugging Face-like object detection pipeline.
// It supports models that output bounding boxes and class scores.
type ObjectDetectionPipeline struct {
	*backends.BasePipeline
	IDLabelMap     map[int]string
	format         string
	preprocess     []imageutil.PreprocessStep
	normalize      []imageutil.NormalizationStep
	BoxesOutput    string
	ScoresOutput   string
	ScoreThreshold float32
	IouThreshold   float32
	TopK           int
}

func (p *ObjectDetectionPipeline) GetStatistics() backends.PipelineStatistics {
	statistics := backends.PipelineStatistics{}
	backends.ComputeOnnxStatistics(&statistics, p.PipelineTimings)
	return statistics
}

type Detection struct {
	Box   [4]float32 // [xmin, ymin, xmax, ymax] in pixels
	Label string
	Score float32
	Class int
}

type ObjectDetectionOutput struct {
	Detections [][]Detection
}

func (o *ObjectDetectionOutput) GetOutput() []any {
	out := make([]any, len(o.Detections))
	for i, dets := range o.Detections {
		out[i] = any(dets)
	}
	return out
}

// Options.
func WithDetectionPreprocess(steps ...imageutil.PreprocessStep) backends.PipelineOption[*ObjectDetectionPipeline] {
	return func(p *ObjectDetectionPipeline) error { p.preprocess = append(p.preprocess, steps...); return nil }
}

func WithDetectionNormalize(steps ...imageutil.NormalizationStep) backends.PipelineOption[*ObjectDetectionPipeline] {
	return func(p *ObjectDetectionPipeline) error { p.normalize = append(p.normalize, steps...); return nil }
}

func WithDetectionNHWC() backends.PipelineOption[*ObjectDetectionPipeline] {
	return func(p *ObjectDetectionPipeline) error { p.format = "NHWC"; return nil }
}

func WithDetectionNCHW() backends.PipelineOption[*ObjectDetectionPipeline] {
	return func(p *ObjectDetectionPipeline) error { p.format = "NCHW"; return nil }
}

func WithBoxesOutput(name string) backends.PipelineOption[*ObjectDetectionPipeline] {
	return func(p *ObjectDetectionPipeline) error { p.BoxesOutput = name; return nil }
}

func WithScoresOutput(name string) backends.PipelineOption[*ObjectDetectionPipeline] {
	return func(p *ObjectDetectionPipeline) error { p.ScoresOutput = name; return nil }
}

func WithDetectionScoreThreshold(th float32) backends.PipelineOption[*ObjectDetectionPipeline] {
	return func(p *ObjectDetectionPipeline) error { p.ScoreThreshold = th; return nil }
}

func WithDetectionIouThreshold(th float32) backends.PipelineOption[*ObjectDetectionPipeline] {
	return func(p *ObjectDetectionPipeline) error { p.IouThreshold = th; return nil }
}

func WithDetectionTopK(k int) backends.PipelineOption[*ObjectDetectionPipeline] {
	return func(p *ObjectDetectionPipeline) error { p.TopK = k; return nil }
}

// NewObjectDetectionPipeline initializes an object detection pipeline.
func NewObjectDetectionPipeline(config backends.PipelineConfig[*ObjectDetectionPipeline], s *options.Options, model *backends.Model) (*ObjectDetectionPipeline, error) {
	base, err := backends.NewBasePipeline(config, s, model)
	if err != nil {
		return nil, err
	}
	p := &ObjectDetectionPipeline{BasePipeline: base, ScoreThreshold: 0.25, IouThreshold: 0.45, TopK: 100}
	for _, o := range config.Options {
		if err = o(p); err != nil {
			return nil, err
		}
	}
	p.IDLabelMap = model.IDLabelMap
	if p.format == "" {
		fmtDetected, err := backends.DetectImageTensorFormat(model)
		if err != nil {
			return nil, err
		}
		p.format = fmtDetected
	}
	// Set sensible default normalization for vision models (Rescale + ImageNet mean/std)
	if len(p.normalize) == 0 {
		p.normalize = []imageutil.NormalizationStep{
			imageutil.RescaleStep(),
			imageutil.ImagenetPixelNormalizationStep(),
		}
	}
	if err := p.Validate(); err != nil {
		return nil, err
	}
	return p, nil
}

// Interface implementations.
func (p *ObjectDetectionPipeline) GetModel() *backends.Model { return p.Model }

func (p *ObjectDetectionPipeline) GetMetadata() backends.PipelineMetadata {
	outputs := make([]backends.OutputInfo, len(p.Model.OutputsMeta))
	for i, o := range p.Model.OutputsMeta {
		outputs[i] = backends.OutputInfo{Name: o.Name, Dimensions: o.Dimensions}
	}
	return backends.PipelineMetadata{OutputsInfo: outputs}
}

func (p *ObjectDetectionPipeline) GetStats() []string {
	return []string{
		fmt.Sprintf("Statistics for pipeline: %s", p.PipelineName),
		fmt.Sprintf("ONNX: Total time=%s, Execution count=%d, Average query time=%s",
			safeconv.U64ToDuration(p.PipelineTimings.TotalNS),
			p.PipelineTimings.NumCalls,
			time.Duration(float64(p.PipelineTimings.TotalNS)/math.Max(1, float64(p.PipelineTimings.NumCalls)))),
	}
}

func (p *ObjectDetectionPipeline) Validate() error {
	var errs []error
	for _, input := range p.Model.InputsMeta {
		lower := strings.ToLower(input.Name)
		if strings.Contains(lower, "mask") {
			if len(input.Dimensions) != 3 {
				errs = append(errs, fmt.Errorf("input %s must be 3D mask tensor", input.Name))
			}
			continue
		}
		if len(input.Dimensions) != 4 {
			errs = append(errs, fmt.Errorf("input %s must be 4D image tensor", input.Name))
		}
	}
	if p.BoxesOutput == "" || p.ScoresOutput == "" {
		var boxes, scores string
		for _, o := range p.Model.OutputsMeta {
			lower := strings.ToLower(o.Name)
			if boxes == "" && strings.Contains(lower, "box") {
				boxes = o.Name
			}
			if scores == "" && (strings.Contains(lower, "logit") || strings.Contains(lower, "score") || strings.Contains(lower, "class")) {
				scores = o.Name
			}
		}
		if boxes == "" || scores == "" {
			errs = append(errs, fmt.Errorf("could not infer boxes/scores outputs; set WithBoxesOutput/WithScoresOutput"))
		} else {
			p.BoxesOutput, p.ScoresOutput = boxes, scores
		}
	}
	return errors.Join(errs...)
}

// Preprocess images into tensors.
func (p *ObjectDetectionPipeline) Preprocess(batch *backends.PipelineBatch, inputs []image.Image) error {
	processed, err := backends.PreprocessImages(p.format, inputs, p.preprocess, p.normalize)
	if err != nil {
		return err
	}
	return backends.CreateImageTensors(batch, p.Model, processed, p.Runtime)
}

// Forward inference.
func (p *ObjectDetectionPipeline) Forward(batch *backends.PipelineBatch) error {
	start := time.Now()
	if err := backends.RunSessionOnBatch(batch, p.BasePipeline); err != nil {
		return err
	}
	atomic.AddUint64(&p.PipelineTimings.NumCalls, 1)
	atomic.AddUint64(&p.PipelineTimings.TotalNS, safeconv.DurationToU64(time.Since(start)))
	return nil
}

// Postprocess parses boxes/scores, applies thresholds and NMS.
func (p *ObjectDetectionPipeline) Postprocess(batch *backends.PipelineBatch) (*ObjectDetectionOutput, error) {
	// Locate outputs by name in OutputValues
	// We assume order of Model.OutputsMeta corresponds to OutputValues
	boxesIdx, scoresIdx := -1, -1
	for i, meta := range p.Model.OutputsMeta {
		if meta.Name == p.BoxesOutput {
			boxesIdx = i
		}
		if meta.Name == p.ScoresOutput {
			scoresIdx = i
		}
	}
	if boxesIdx < 0 || scoresIdx < 0 {
		return nil, fmt.Errorf("boxes/scores outputs not found")
	}

	boxesAny := batch.OutputValues[boxesIdx]
	scoresAny := batch.OutputValues[scoresIdx]

	// Expected shapes: boxes [batch][num][4], scores [batch][num][num_classes]
	boxes, okB := boxesAny.([][][]float32)
	scores, okS := scoresAny.([][][]float32)
	if !okB || !okS {
		return nil, fmt.Errorf("unexpected output types: boxes=%T scores=%T", boxesAny, scoresAny)
	}

	out := &ObjectDetectionOutput{Detections: make([][]Detection, len(boxes))}
	for b := range boxes {
		dets := decodeDetections(boxes[b], scores[b], p.IDLabelMap, p.ScoreThreshold, p.TopK)
		dets = nonMaxSuppress(dets, p.IouThreshold)
		out.Detections[b] = dets
	}
	return out, nil
}

func decodeDetections(boxes [][]float32, scores [][]float32, labels map[int]string, scoreTh float32, topK int) []Detection {
	type cand struct {
		idx   int
		cls   int
		score float32
	}
	var cands []cand
	// Activate logits with softmax per box and pick best class (skip N/A class id 0).
	for i := range scores {
		probs := softmax(scores[i])
		bestCls := -1
		bestScore := float32(0)
		for c, p := range probs {
			if c == 0 { // skip no-object class
				continue
			}
			if p > bestScore {
				bestScore = p
				bestCls = c
			}
		}
		if bestCls >= 0 && bestScore >= scoreTh {
			cands = append(cands, cand{idx: i, cls: bestCls, score: bestScore})
		}
	}
	sort.Slice(cands, func(i, j int) bool { return cands[i].score > cands[j].score })
	if topK > 0 && topK < len(cands) {
		cands = cands[:topK]
	}
	dets := make([]Detection, 0, len(cands))
	for _, cd := range cands {
		box := convertBoxToCorners(boxes[cd.idx])
		label := fmt.Sprintf("class_%d", cd.cls)
		if labels != nil {
			if l, ok := labels[cd.cls]; ok {
				label = l
			}
		}
		dets = append(dets, Detection{Box: [4]float32{box[0], box[1], box[2], box[3]}, Label: label, Score: cd.score, Class: cd.cls})
	}
	return dets
}

// softmax applies numerically-stable softmax to a slice.
func softmax(x []float32) []float32 {
	if len(x) == 0 {
		return nil
	}
	maxVal := x[0]
	for _, v := range x {
		if v > maxVal {
			maxVal = v
		}
	}
	sum := float64(0)
	out := make([]float32, len(x))
	for i, v := range x {
		e := math.Exp(float64(v - maxVal))
		out[i] = float32(e)
		sum += e
	}
	if sum == 0 {
		return out
	}
	for i := range out {
		out[i] = out[i] / float32(sum)
	}
	return out
}

// convertBoxToCorners assumes input box is [cx, cy, w, h] normalized to [0,1].
func convertBoxToCorners(b []float32) []float32 {
	if len(b) != 4 {
		return []float32{0, 0, 0, 0}
	}
	cx, cy, w, h := b[0], b[1], b[2], b[3]
	x1 := cx - w/2
	y1 := cy - h/2
	x2 := cx + w/2
	y2 := cy + h/2
	return []float32{x1, y1, x2, y2}
}

func iou(a, b [4]float32) float32 {
	ax1, ay1, ax2, ay2 := a[0], a[1], a[2], a[3]
	bx1, by1, bx2, by2 := b[0], b[1], b[2], b[3]
	interX1 := float32(math.Max(float64(ax1), float64(bx1)))
	interY1 := float32(math.Max(float64(ay1), float64(by1)))
	interX2 := float32(math.Min(float64(ax2), float64(bx2)))
	interY2 := float32(math.Min(float64(ay2), float64(by2)))
	iw := interX2 - interX1
	ih := interY2 - interY1
	if iw <= 0 || ih <= 0 {
		return 0
	}
	inter := iw * ih
	areaA := (ax2 - ax1) * (ay2 - ay1)
	areaB := (bx2 - bx1) * (by2 - by1)
	union := areaA + areaB - inter
	if union <= 0 {
		return 0
	}
	return inter / union
}

func nonMaxSuppress(dets []Detection, iouTh float32) []Detection {
	sort.Slice(dets, func(i, j int) bool { return dets[i].Score > dets[j].Score })
	var keep []Detection
	for _, d := range dets {
		suppressed := false
		for _, k := range keep {
			if d.Label == k.Label && iou(d.Box, k.Box) > iouTh {
				suppressed = true
				break
			}
		}
		if !suppressed {
			keep = append(keep, d)
		}
	}
	return keep
}

// Run with file paths.
func (p *ObjectDetectionPipeline) Run(inputs []string) (backends.PipelineBatchOutput, error) {
	return p.RunPipeline(inputs)
}

func (p *ObjectDetectionPipeline) RunPipeline(inputs []string) (*ObjectDetectionOutput, error) {
	var errs []error
	batch := backends.NewBatch(len(inputs))
	defer func(*backends.PipelineBatch) { errs = append(errs, batch.Destroy()) }(batch)
	imgs, err := imageutil.LoadImagesFromPaths(inputs)
	if err != nil {
		return nil, fmt.Errorf("failed to load images: %w", err)
	}
	errs = append(errs, p.Preprocess(batch, imgs))
	if e := errors.Join(errs...); e != nil {
		return nil, e
	}
	errs = append(errs, p.Forward(batch))
	if e := errors.Join(errs...); e != nil {
		return nil, e
	}
	res, postErr := p.Postprocess(batch)
	errs = append(errs, postErr)
	return res, errors.Join(errs...)
}

func (p *ObjectDetectionPipeline) RunWithImages(inputs []image.Image) (*ObjectDetectionOutput, error) {
	var errs []error
	batch := backends.NewBatch(len(inputs))
	defer func(*backends.PipelineBatch) { errs = append(errs, batch.Destroy()) }(batch)
	errs = append(errs, p.Preprocess(batch, inputs))
	if e := errors.Join(errs...); e != nil {
		return nil, e
	}
	errs = append(errs, p.Forward(batch))
	if e := errors.Join(errs...); e != nil {
		return nil, e
	}
	res, postErr := p.Postprocess(batch)
	errs = append(errs, postErr)
	return res, errors.Join(errs...)
}
