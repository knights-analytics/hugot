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

// ZeroShotObjectDetectionPipeline describes object detection models whose
// categories are supplied at runtime. Models must expose an image/text
// contract; the regular object detector is used when labels are provided by
// the model's ID-to-label map.
type ZeroShotObjectDetectionPipeline struct {
	*ObjectDetectionPipeline
	Labels []string
}

type ZeroShotObjectDetectionOutput = ObjectDetectionOutput

func (p *ZeroShotObjectDetectionPipeline) IsGenerative() bool { return false }

func WithZeroShotObjectLabels(labels []string) backends.PipelineOption[*ZeroShotObjectDetectionPipeline] {
	return func(p *ZeroShotObjectDetectionPipeline) error {
		p.Labels = append([]string(nil), labels...)
		return nil
	}
}

func NewZeroShotObjectDetectionPipeline(ctx context.Context, config backends.PipelineConfig[*ZeroShotObjectDetectionPipeline], model *backends.Model) (*ZeroShotObjectDetectionPipeline, error) {
	baseConfig := backends.PipelineConfig[*ObjectDetectionPipeline]{
		Name: config.Name, ModelPath: config.ModelPath, OnnxFilename: config.OnnxFilename,
	}
	p := &ZeroShotObjectDetectionPipeline{ObjectDetectionPipeline: &ObjectDetectionPipeline{BasePipeline: backends.NewBasePipeline(ctx, baseConfig, model), ScoreThreshold: 0.25, IouThreshold: 0.45, TopK: 100}}
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

func (p *ZeroShotObjectDetectionPipeline) Validate() error {
	if p.Model == nil {
		return errors.New("zero-shot object detection requires a model")
	}
	if len(p.Labels) == 0 {
		return errors.New("zero-shot object detection requires at least one candidate label")
	}
	var errs []error
	if p.Model.Tokenizer == nil {
		errs = append(errs, errors.New("zero-shot object detection requires a tokenizer for candidate labels"))
	}
	for _, input := range p.Model.InputsMeta {
		if isTextInput(input.Name) {
			if len(input.Dimensions) != 2 && len(input.Dimensions) != 3 {
				errs = append(errs, fmt.Errorf("input %s must be a rank-2 or rank-3 image/query token tensor", input.Name))
			}
		} else if len(input.Dimensions) != 4 && !strings.Contains(strings.ToLower(input.Name), "mask") {
			errs = append(errs, fmt.Errorf("input %s must be a rank-4 image tensor", input.Name))
		}
	}
	if p.BoxesOutput == "" || p.ScoresOutput == "" {
		var boxes, scores string
		for _, output := range p.Model.OutputsMeta {
			name := strings.ToLower(output.Name)
			if boxes == "" && strings.Contains(name, "box") {
				boxes = output.Name
			}
			if scores == "" && (strings.Contains(name, "logit") || strings.Contains(name, "score")) {
				scores = output.Name
			}
		}
		if boxes == "" || scores == "" {
			errs = append(errs, errors.New("zero-shot object detection requires box and score outputs"))
		} else {
			p.BoxesOutput, p.ScoresOutput = boxes, scores
		}
	}
	return errors.Join(errs...)
}

// Run is unavailable because []string cannot carry candidate labels. Use
// RunWithLabels for a request-specific label set.
func (p *ZeroShotObjectDetectionPipeline) Run(context.Context, []string) (backends.PipelineBatchOutput, error) {
	return nil, errors.New("zero-shot object detection requires paired image/text inputs; use RunWithLabels")
}

// RunWithLabels validates the paired request. A backend with an explicit
// image/text tensor contract can implement the model-specific execution;
// current generic image execution cannot encode candidate labels safely.
func (p *ZeroShotObjectDetectionPipeline) RunWithLabels(ctx context.Context, inputs []string, labels []string) (*ZeroShotObjectDetectionOutput, error) {
	if len(inputs) == 0 {
		return nil, errors.New("zero-shot object detection requires at least one image")
	}
	if len(labels) == 0 && len(p.Labels) == 0 {
		return nil, errors.New("zero-shot object detection requires at least one candidate label")
	}
	images, err := imageutil.LoadImagesFromPaths(ctx, inputs)
	if err != nil {
		return nil, fmt.Errorf("failed to load images: %w", err)
	}
	return p.RunWithImagesAndLabels(ctx, images, labels)
}

func (p *ZeroShotObjectDetectionPipeline) RunWithImagesAndLabels(ctx context.Context, inputs []image.Image, labels []string) (*ZeroShotObjectDetectionOutput, error) {
	if len(inputs) == 0 {
		return nil, errors.New("zero-shot object detection requires at least one image")
	}
	if len(labels) == 0 && len(p.Labels) == 0 {
		return nil, errors.New("zero-shot object detection requires at least one candidate label")
	}
	if len(labels) == 0 {
		labels = p.Labels
	}
	for _, label := range labels {
		if strings.TrimSpace(label) == "" {
			return nil, errors.New("zero-shot object detection received an empty label")
		}
	}
	for _, img := range inputs {
		if img == nil {
			return nil, errors.New("zero-shot object detection received a nil image")
		}
	}
	return p.run(ctx, inputs, labels)
}

func isTextInput(name string) bool {
	lower := strings.ToLower(name)
	return strings.Contains(lower, "input_ids") || strings.Contains(lower, "attention_mask") || strings.Contains(lower, "token_type") || strings.Contains(lower, "position_ids")
}

func (p *ZeroShotObjectDetectionPipeline) run(ctx context.Context, images []image.Image, labels []string) (*ZeroShotObjectDetectionOutput, error) {
	result, err := backends.RunPipeline(ctx, len(images), func(batch *backends.PipelineBatch) error {
		format, err := backends.DetectImageTensorFormat(p.Model)
		if err != nil {
			return err
		}
		processed, err := backends.PreprocessImages(format, images, nil, nil)
		if err != nil {
			return err
		}
		backends.TokenizeInputs(batch, p.Model.Tokenizer, labels)
		batch.QueryCount = len(labels)
		return p.Model.Backend.CreateImageTextTensors(batch, p.Model, processed, batch.Input)
	}, func(ctx context.Context, batch *backends.PipelineBatch) error {
		return backends.RunSessionOnBatch(ctx, batch, p.BasePipeline)
	}, func(batch *backends.PipelineBatch) (*ZeroShotObjectDetectionOutput, error) {
		boxesIndex, scoresIndex := -1, -1
		for i, output := range p.Model.OutputsMeta {
			if output.Name == p.BoxesOutput {
				boxesIndex = i
			}
			if output.Name == p.ScoresOutput {
				scoresIndex = i
			}
		}
		if boxesIndex < 0 || scoresIndex < 0 {
			return nil, errors.New("zero-shot object detection outputs were not found")
		}
		boxes, boxesOK := batch.OutputValues[boxesIndex].([][][]float32)
		scores, scoresOK := batch.OutputValues[scoresIndex].([][][]float32)
		if !boxesOK || !scoresOK {
			return nil, fmt.Errorf("unsupported zero-shot object detection output types: boxes=%T scores=%T", batch.OutputValues[boxesIndex], batch.OutputValues[scoresIndex])
		}
		result := &ZeroShotObjectDetectionOutput{Detections: make([][]Detection, len(boxes))}
		for b := range boxes {
			if b >= len(scores) {
				continue
			}
			for query, box := range boxes[b] {
				if query >= len(scores[b]) || len(box) != 4 {
					continue
				}
				for labelIndex, value := range scores[b][query] {
					if labelIndex >= len(labels) {
						continue
					}
					score := float32(1 / (1 + math.Exp(float64(-value))))
					if score >= p.ScoreThreshold {
						corners := convertBoxToCorners(box)
						result.Detections[b] = append(result.Detections[b], Detection{
							Label: labels[labelIndex], Score: score,
							Box: [4]float32{corners[0], corners[1], corners[2], corners[3]},
						})
					}
				}
			}
			result.Detections[b] = nonMaxSuppress(result.Detections[b], p.IouThreshold)
		}
		return result, nil
	})
	if err != nil {
		return nil, err
	}
	return result, nil
}
