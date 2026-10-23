package pipelines

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"sort"

	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/util/imageutil"
)

// ImagePair keeps pair boundaries explicit. Each side must supply exactly one
// decoded image or path. Images may have different original dimensions.
type ImagePair struct {
	Images [2]image.Image `json:"-"`
	Paths  [2]string      `json:"paths"`
}
type Keypoint struct {
	X int `json:"x"`
	Y int `json:"y"`
}
type KeypointMatch struct {
	KeypointImage0 Keypoint `json:"keypoint_image_0"`
	KeypointImage1 Keypoint `json:"keypoint_image_1"`
	Score          float32  `json:"score"`
}
type KeypointMatchingOutput struct{ Matches [][]KeypointMatch }

func (o *KeypointMatchingOutput) GetOutput() []any {
	out := make([]any, len(o.Matches))
	for i := range out {
		out[i] = o.Matches[i]
	}
	return out
}

// KeypointMatchingPipeline accepts the dense/padded FP32 SuperGlue export
// contract. Dynamic keypoint counts are read from runtime output shapes.
type KeypointMatchingPipeline struct {
	*backends.BasePipeline
	Threshold float32
	processor visionProcessor
}
type KeypointMatchingConfig = backends.PipelineConfig[*KeypointMatchingPipeline]
type KeypointMatchingOption = backends.PipelineOption[*KeypointMatchingPipeline]

func WithKeypointMatchingThreshold(threshold float32) KeypointMatchingOption {
	return func(p *KeypointMatchingPipeline) error { p.Threshold = threshold; return nil }
}

func NewKeypointMatchingPipeline(ctx context.Context, config KeypointMatchingConfig, model *backends.Model) (*KeypointMatchingPipeline, error) {
	if model == nil {
		return nil, errors.New("keypoint matching requires a SuperGlue ONNX model")
	}
	p := &KeypointMatchingPipeline{BasePipeline: backends.NewBasePipeline(ctx, config, model)}
	var cfg struct {
		ModelType string `json:"model_type"`
	}
	if err := readNativeConfig(ctx, model.Path, "config.json", &cfg); err != nil {
		return nil, err
	}
	if cfg.ModelType != "superglue" {
		return nil, errors.New("keypoint matching currently requires model_type superglue")
	}
	if err := readNativeConfig(ctx, model.Path, "preprocessor_config.json", &p.processor); err != nil {
		return nil, err
	}
	if p.processor.ProcessorType != "SuperGlueImageProcessor" || !p.processor.Grayscale || p.processor.CenterCrop || p.processor.Size.ShortestEdge != 0 {
		return nil, errors.New("keypoint matching requires the SuperGlue grayscale fixed-size processor")
	}
	for _, option := range config.Options {
		if err := option(p); err != nil {
			return nil, err
		}
	}
	return p, p.Validate()
}

func (*KeypointMatchingPipeline) IsGenerative() bool          { return false }
func (p *KeypointMatchingPipeline) GetModel() *backends.Model { return p.Model }
func (p *KeypointMatchingPipeline) GetMetadata() backends.PipelineMetadata {
	return nativeMetadata(p.Model)
}
func (p *KeypointMatchingPipeline) GetStatistics() backends.PipelineStatistics {
	return nativeStatistics(p.BasePipeline)
}
func (p *KeypointMatchingPipeline) Validate() error {
	if p == nil || p.BasePipeline == nil {
		return errors.New("keypoint matching requires a model")
	}
	if err := requireTensorNames(p.Model, []string{"pixel_values"}, []string{"keypoints", "matches", "matching_scores", "mask"}); err != nil {
		return err
	}
	if !finite32(p.Threshold) || p.Threshold < 0 || p.Threshold > 1 {
		return errors.New("keypoint matching threshold must be between zero and one")
	}
	w, h, err := p.processor.dimensions()
	if err != nil {
		return err
	}
	if len(p.Model.InputsMeta) != 1 {
		return errors.New("SuperGlue graph must have only pixel_values as input")
	}
	if err := tensorContract(p.Model.InputsMeta[0], []int64{1, 2, 3, int64(h), int64(w)}); err != nil {
		return err
	}
	for name, shape := range map[string][]int64{"keypoints": {1, 2, -1, 2}, "matches": {1, 2, -1}, "matching_scores": {1, 2, -1}, "mask": {1, 2, -1}} {
		if err := namedTensorContract(p.Model.OutputsMeta, name, shape); err != nil {
			return err
		}
	}
	return nil
}

// Run accepts one JSON ImagePair per string, e.g. {"paths":["a.png","b.png"]}.
func (p *KeypointMatchingPipeline) Run(ctx context.Context, inputs []string) (backends.PipelineBatchOutput, error) {
	pairs := make([]ImagePair, len(inputs))
	for i, s := range inputs {
		if err := json.Unmarshal([]byte(s), &pairs[i]); err != nil {
			return nil, fmt.Errorf("image pair %d: %w", i, err)
		}
	}
	return p.RunPairs(ctx, pairs)
}
func (p *KeypointMatchingPipeline) RunPairs(ctx context.Context, pairs []ImagePair) (*KeypointMatchingOutput, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	if err := validatePipelineContext(ctx, p.BasePipeline); err != nil {
		return nil, err
	}
	if len(pairs) == 0 {
		return nil, errors.New("keypoint matching requires at least one image pair")
	}
	// Keep the session filesystem while allowing the caller to cancel image I/O.
	inputCtx := ctx
	if p.SessionContext != nil {
		merged, cancel := context.WithCancel(p.SessionContext)
		stop := context.AfterFunc(ctx, cancel)
		defer func() { stop(); cancel() }()
		inputCtx = merged
	}
	out := &KeypointMatchingOutput{Matches: make([][]KeypointMatch, len(pairs))}
	for i, pair := range pairs {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		images := pair.Images
		for side := range images {
			if (images[side] == nil) == (pair.Paths[side] == "") {
				return nil, fmt.Errorf("pair %d side %d requires exactly one image or path", i, side)
			}
			if images[side] == nil {
				loaded, err := imageutil.LoadImagesFromPaths(inputCtx, []string{pair.Paths[side]})
				if err != nil {
					return nil, err
				}
				images[side] = loaded[0]
			}
		}
		pixels, err := p.preprocessPair(images)
		if err != nil {
			return nil, fmt.Errorf("pair %d: %w", i, err)
		}
		outputs, err := runNativeTensors(ctx, p.BasePipeline, p.Model, map[string]backends.Tensor{"pixel_values": pixels})
		if err != nil {
			return nil, err
		}
		out.Matches[i], err = p.decodeMatches(outputs, [2]image.Point{images[0].Bounds().Size(), images[1].Bounds().Size()})
		if err != nil {
			return nil, fmt.Errorf("pair %d: %w", i, err)
		}
	}
	return out, nil
}

func (p *KeypointMatchingPipeline) preprocessPair(images [2]image.Image) (backends.Tensor, error) {
	w, h, err := p.processor.dimensions()
	if err != nil {
		return backends.Tensor{}, err
	}
	data := make([]float32, 0, 2*3*w*h)
	for _, img := range images {
		pixels, err := p.processor.pixels(img)
		if err != nil {
			return backends.Tensor{}, err
		}
		data = append(data, pixels...)
	}
	return backends.Tensor{Shape: []int64{1, 2, 3, int64(h), int64(w)}, Data: data}, nil
}

func (p *KeypointMatchingPipeline) decodeMatches(outputs map[string]backends.Tensor, sizes [2]image.Point) ([]KeypointMatch, error) {
	kp := outputs["keypoints"]
	points, ok := kp.Data.([]float32)
	if !ok || len(kp.Shape) != 4 || kp.Shape[0] != 1 || kp.Shape[1] != 2 || kp.Shape[2] < 0 || kp.Shape[3] != 2 || kp.Shape[2] > int64(len(points))/4 || int64(len(points)) != 4*kp.Shape[2] {
		return nil, errors.New("SuperGlue keypoints must be FP32 [1,2,N,2]")
	}
	n := int(kp.Shape[2])
	for _, name := range []string{"matches", "matching_scores", "mask"} {
		t := outputs[name]
		if len(t.Shape) != 3 || t.Shape[0] != 1 || t.Shape[1] != 2 || t.Shape[2] != int64(n) {
			return nil, fmt.Errorf("SuperGlue %s must have shape [1,2,N]", name)
		}
	}
	matches, ok := outputs["matches"].Data.([]int64)
	if !ok || len(matches) != 2*n {
		return nil, errors.New("SuperGlue matches must contain int64 indices")
	}
	scores, ok := outputs["matching_scores"].Data.([]float32)
	if !ok || len(scores) != 2*n {
		return nil, errors.New("SuperGlue matching_scores must contain FP32 scores")
	}
	mask := make([]bool, 2*n)
	switch v := outputs["mask"].Data.(type) {
	case []int64:
		if len(v) != 2*n {
			return nil, errors.New("invalid SuperGlue mask length")
		}
		for i, m := range v {
			mask[i] = m > 0
		}
	case []bool:
		if len(v) != 2*n {
			return nil, errors.New("invalid SuperGlue mask length")
		}
		copy(mask, v)
	default:
		return nil, errors.New("SuperGlue mask must contain int64 or bool values")
	}
	for _, s := range sizes {
		if s.X <= 0 || s.Y <= 0 {
			return nil, errors.New("invalid original image dimensions")
		}
	}
	// The pinned processor indexes into the compacted valid keypoints in image 1.
	valid1 := make([]int, 0, n)
	for j := 0; j < n; j++ {
		if mask[n+j] {
			valid1 = append(valid1, j)
		}
	}
	result := make([]KeypointMatch, 0)
	for i := 0; i < n; i++ {
		if !mask[i] {
			continue
		}
		if !finite32(scores[i]) {
			return nil, errors.New("SuperGlue scores must be finite")
		}
		j := matches[i]
		if j < 0 || j >= int64(len(valid1)) || scores[i] <= p.Threshold {
			continue
		}
		indices := [2]int{i, n + valid1[int(j)]}
		var coordinates [2]Keypoint
		for side, index := range indices {
			x, y := points[2*index], points[2*index+1]
			if !finite32(x) || !finite32(y) || x < 0 || x > 1 || y < 0 || y > 1 {
				return nil, errors.New("SuperGlue coordinates must be normalized to [0,1]")
			}
			coordinates[side] = Keypoint{X: int(x * float32(sizes[side].X)), Y: int(y * float32(sizes[side].Y))}
		}
		result = append(result, KeypointMatch{KeypointImage0: coordinates[0], KeypointImage1: coordinates[1], Score: scores[i]})
	}
	sort.SliceStable(result, func(i, j int) bool { return result[i].Score > result[j].Score })
	return result, nil
}
