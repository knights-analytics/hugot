//go:build cgo && (XLA || ALL) && !TRAINING

package xla_test

import (
	"testing"

	testutil "github.com/knights-analytics/hugot/tests"
)

// Image classification

func TestImageClassificationPipelineXLA(t *testing.T) {
	runXLAPipeline(t, testutil.ImageClassificationPipeline)
}

func TestImageClassificationPipelineXLACuda(t *testing.T) {
	runXLAPipelineCuda(t, testutil.ImageClassificationPipeline)
}

func TestImageClassificationPipelineValidationXLA(t *testing.T) {
	runXLAPipelineValidation(t, testutil.ImageClassificationPipelineValidation)
}

// Object detection

func TestObjectDetectionPipelineXLA(t *testing.T) {
	runXLAPipeline(t, testutil.ObjectDetectionPipeline)
}

func TestObjectDetectionPipelineXLACuda(t *testing.T) {
	runXLAPipelineCuda(t, testutil.ObjectDetectionPipeline)
}

func TestObjectDetectionPipelineValidationXLA(t *testing.T) {
	runXLAPipelineValidation(t, testutil.ObjectDetectionPipelineValidation)
}

// Zero-shot object detection

func TestZeroShotObjectDetectionPipelineXLA(t *testing.T) {
	t.Skip("Skipping test due to known operands description (\"...pq\") has axis '.' appearing more than once issue with XLA pipeline")
	runXLAPipeline(t, testutil.ZeroShotObjectDetectionPipeline)
}

func TestZeroShotObjectDetectionPipelineXLACuda(t *testing.T) {
	t.Skip("Skipping test due to known operands description (\"...pq\") has axis '.' appearing more than once issue with XLA pipeline")
	runXLAPipelineCuda(t, testutil.ZeroShotObjectDetectionPipeline)
}

func TestZeroShotObjectDetectionPipelineValidationXLA(t *testing.T) {
	runXLAPipelineValidation(t, testutil.ZeroShotObjectDetectionPipelineValidation)
}

// Image feature extraction

func TestImageFeatureExtractionPipelineXLA(t *testing.T) {
	runXLAPipeline(t, testutil.ImageFeatureExtractionPipeline)
}

func TestImageFeatureExtractionPipelineXLACuda(t *testing.T) {
	runXLAPipelineCuda(t, testutil.ImageFeatureExtractionPipeline)
}

func TestImageFeatureExtractionPipelineValidationXLA(t *testing.T) {
	runXLAPipelineValidation(t, testutil.ImageFeatureExtractionPipelineValidation)
}

// Image segmentation

func TestImageSegmentationPipelineXLA(t *testing.T) {
	runXLAPipeline(t, testutil.ImageSegmentationPipeline)
}

func TestImageSegmentationPipelineXLACuda(t *testing.T) {
	runXLAPipelineCuda(t, testutil.ImageSegmentationPipeline)
}

func TestImageSegmentationPipelineValidationXLA(t *testing.T) {
	runXLAPipelineValidation(t, testutil.ImageSegmentationPipelineValidation)
}

// Depth estimation

func TestDepthEstimationPipelineXLA(t *testing.T) {
	t.Skip("Skipping test due to known compilation failed for Exec issue with XLA pipeline")
	runXLAPipeline(t, testutil.DepthEstimationPipeline)
}

func TestDepthEstimationPipelineXLACuda(t *testing.T) {
	t.Skip("Skipping test due to known compilation failed for Exec issue with XLA pipeline")
	runXLAPipelineCuda(t, testutil.DepthEstimationPipeline)
}

func TestDepthEstimationPipelineValidationXLA(t *testing.T) {
	runXLAPipelineValidation(t, testutil.DepthEstimationPipelineValidation)
}

// Mask generation

func TestMaskGenerationPipelineXLA(t *testing.T) {
	t.Skip("Skipping test due to known total requested size doesnt match original size issue with XLA pipeline")
	runXLAPipeline(t, testutil.MaskGenerationPipeline)
}

func TestMaskGenerationPipelineXLACuda(t *testing.T) {
	t.Skip("Skipping test due to known total requested size doesnt match original size issue with XLA pipeline")
	runXLAPipelineCuda(t, testutil.MaskGenerationPipeline)
}

func TestMaskGenerationPipelineValidationXLA(t *testing.T) {
	t.Skip("Skipping test due to known total requested size doesnt match original size issue with XLA pipeline")
	runXLAPipelineValidation(t, testutil.MaskGenerationPipelineValidation)
}

// Background removal

func TestBackgroundRemovalPipelineXLA(t *testing.T) {
	t.Skip("Skipping test due to known total requested size doesnt match original size issue with XLA pipeline")
	runXLAPipeline(t, testutil.BackgroundRemovalPipeline)
}

func TestBackgroundRemovalPipelineXLACuda(t *testing.T) {
	t.Skip("Skipping test due to known total requested size doesnt match original size issue with XLA pipeline")
	runXLAPipelineCuda(t, testutil.BackgroundRemovalPipeline)
}

func TestBackgroundRemovalPipelineValidationXLA(t *testing.T) {
	t.Skip("Skipping test due to known total requested size doesnt match original size issue with XLA pipeline")
	runXLAPipelineValidation(t, testutil.BackgroundRemovalPipelineValidation)
}

// Zero-shot image classification

func TestZeroShotImageClassificationPipelineXLA(t *testing.T) {
	t.Skip("Skipping test due to unimplemented ONNX op \"InstanceNormalization\" issue with XLA pipeline")
	runXLAPipeline(t, testutil.ZeroShotImageClassificationPipeline)
}

func TestZeroShotImageClassificationPipelineXLACuda(t *testing.T) {
	t.Skip("Skipping test due to unimplemented ONNX op \"InstanceNormalization\" issue with XLA pipeline")
	runXLAPipelineCuda(t, testutil.ZeroShotImageClassificationPipeline)
}

func TestZeroShotImageClassificationPipelineValidationXLA(t *testing.T) {
	t.Skip("Skipping test due to unimplemented ONNX op \"InstanceNormalization\" issue with XLA pipeline")
	runXLAPipelineValidation(t, testutil.ZeroShotImageClassificationPipelineValidation)
}
