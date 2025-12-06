package pipelines

import (
	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/util/imageutil"
)

// preprocessPipeline is the minimal interface for pipelines that support image preprocess steps.
// Both FeatureExtractionPipeline (in image mode) and ImageClassificationPipeline implement this.
// It also requires backends.Pipeline so it can be used with backends.PipelineOption generics.
type imagePipeline interface {
	backends.Pipeline
	addPreprocessSteps(...imageutil.PreprocessStep)
	addNormalizationSteps(...imageutil.NormalizationStep)
	setImageFormat(string)
}

// WithPreprocessSteps is a unified option to add image preprocessing steps
// to any pipeline that supports them (e.g., FeatureExtractionPipeline in image mode
// and ImageClassificationPipeline). This avoids conflicting option names.
func WithPreprocessSteps[T imagePipeline](steps ...imageutil.PreprocessStep) backends.PipelineOption[T] {
	return func(p T) error {
		p.addPreprocessSteps(steps...)
		return nil
	}
}

func WithNormalizationSteps[T imagePipeline](steps ...imageutil.NormalizationStep) backends.PipelineOption[T] {
	return func(p T) error {
		p.addNormalizationSteps(steps...)
		return nil
	}
}

func WithNCHWFormat[T imagePipeline]() backends.PipelineOption[T] {
	return func(p T) error {
		p.setImageFormat("NCHW")
		return nil
	}
}

func WithNHWCFormat[T imagePipeline]() backends.PipelineOption[T] {
	return func(p T) error {
		p.setImageFormat("NHWC")
		return nil
	}
}
