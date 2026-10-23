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
	t.Skip("Skipping test due to known operands description (\"...pq\") has axis '.' appearing more than once issue with Go pipeline")
	runGoPipeline(t, testutil.ZeroShotObjectDetectionPipeline)
}

func TestZeroShotObjectDetectionPipelineValidationGo(t *testing.T) {
	t.Skip("Skipping test due to known operands description (\"...pq\") has axis '.' appearing more than once issue with Go pipeline")
	runGoPipelineValidation(t, testutil.ZeroShotObjectDetectionPipelineValidation)
}

// Grounded object detection

func TestGroundedObjectDetectionPipelineGo(t *testing.T) {
	t.Skip("Skipping test due to known total requested size 6488064 (dimensions=[1 16 12 22 12 128]) doesnt match original size 6809600 (dimensions [1 200 266 128]) issue with Go pipeline")
	runGoPipeline(t, testutil.GroundedObjectDetectionPipeline)
}

func TestGroundedObjectDetectionPipelineValidationGo(t *testing.T) {
	t.Skip("Skipping test due to known total requested size 6488064 (dimensions=[1 16 12 22 12 128]) doesnt match original size 6809600 (dimensions [1 200 266 128]) issue with Go pipeline")
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
	t.Skip("Skipping test due to known compilation failed for Exec issue with Go pipeline")
	runGoPipeline(t, testutil.DepthEstimationPipeline)
}

func TestDepthEstimationPipelineValidationGo(t *testing.T) {
	t.Skip("Skipping test due to known compilation failed for Exec issue with Go pipeline")
	runGoPipelineValidation(t, testutil.DepthEstimationPipelineValidation)
}

// Mask generation

func TestMaskGenerationPipelineGo(t *testing.T) {
	t.Skip("Skipping test due to known total requested size doesnt match original size issue with Go pipeline")
	runGoPipeline(t, testutil.MaskGenerationPipeline)
}

func TestMaskGenerationPipelineValidationGo(t *testing.T) {
	t.Skip("Skipping test due to known total requested size doesnt match original size issue with Go pipeline")
	runGoPipelineValidation(t, testutil.MaskGenerationPipelineValidation)
}

// Background removal

func TestBackgroundRemovalPipelineGo(t *testing.T) {
	t.Skip("Skipping test due to unimplemented ONNX op \"InstanceNormalization\"  issue with Go pipeline")
	runGoPipeline(t, testutil.BackgroundRemovalPipeline)
}

func TestBackgroundRemovalPipelineValidationGo(t *testing.T) {
	t.Skip("Skipping test due to unimplemented ONNX op \"InstanceNormalization\"  issue with Go pipeline")
	runGoPipelineValidation(t, testutil.BackgroundRemovalPipelineValidation)
}
