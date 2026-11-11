//go:build (GO || ALL) && !TRAINING

package go_test

import (
	"testing"

	testutil "github.com/knights-analytics/hugot/tests"
)

// Image classification

func TestImageClassificationPipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.ImageClassificationPipeline)
}

func TestImageClassificationPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.ImageClassificationPipelineValidation)
}

// Object detection

func TestObjectDetectionPipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.ObjectDetectionPipeline)
}

func TestObjectDetectionPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.ObjectDetectionPipelineValidation)
}

// Zero-shot object detection

func TestZeroShotObjectDetectionPipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.ZeroShotObjectDetectionPipeline)
}

func TestZeroShotObjectDetectionPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.ZeroShotObjectDetectionPipelineValidation)
}

// Grounded object detection

func TestGroundedObjectDetectionPipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.GroundedObjectDetectionPipeline)
}

func TestGroundedObjectDetectionPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.GroundedObjectDetectionPipelineValidation)
}

// Zero-shot image classification

func TestZeroShotImageClassificationPipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.ZeroShotImageClassificationPipeline)
}

func TestZeroShotImageClassificationPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.ZeroShotImageClassificationPipelineValidation)
}

// Image feature extraction

func TestImageFeatureExtractionPipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.ImageFeatureExtractionPipeline)
}

func TestImageFeatureExtractionPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.ImageFeatureExtractionPipelineValidation)
}

// Image segmentation

func TestImageSegmentationPipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.ImageSegmentationPipeline)
}

func TestImageSegmentationPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.ImageSegmentationPipelineValidation)
}

// Depth estimation

func TestDepthEstimationPipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.DepthEstimationPipeline)
}

func TestDepthEstimationPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.DepthEstimationPipelineValidation)
}

// Mask generation

func TestMaskGenerationPipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.MaskGenerationPipeline)
}

func TestMaskGenerationPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.MaskGenerationPipelineValidation)
}

// Background removal

func TestBackgroundRemovalPipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.BackgroundRemovalPipeline)
}

func TestBackgroundRemovalPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.BackgroundRemovalPipelineValidation)
}
