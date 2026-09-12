//go:build cgo && (ORT || ALL) && !TRAINING

package ort_test

import (
	"testing"

	testutil "github.com/knights-analytics/hugot/tests"
)

// Image classification

func TestImageClassificationPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.ImageClassificationPipeline)
}

func TestImageClassificationPipelineORTGoMLX(t *testing.T) {
	runORTPipelineGoMLX(t, testutil.ImageClassificationPipeline)
}

func TestImageClassificationPipelineORTCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.ImageClassificationPipeline)
}

func TestImageClassificationPipelineValidationORT(t *testing.T) {
	runORTPipelineValidation(t, testutil.ImageClassificationPipelineValidation)
}

// Object detection

func TestObjectDetectionPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.ObjectDetectionPipeline)
}

func TestObjectDetectionPipelineORTGoMLX(t *testing.T) {
	runORTPipelineGoMLX(t, testutil.ObjectDetectionPipeline)
}

func TestObjectDetectionPipelineORTCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.ObjectDetectionPipeline)
}

func TestObjectDetectionPipelineValidationORT(t *testing.T) {
	runORTPipelineValidation(t, testutil.ObjectDetectionPipelineValidation)
}

// Zero-shot object detection

func TestZeroShotObjectDetectionPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.ZeroShotObjectDetectionPipeline)
}

func TestZeroShotObjectDetectionPipelineORTGoMLX(t *testing.T) {
	t.Skip("Skipping test due to known operands description (\"...pq\") has axis '.' appearing more than once issue with GoMLX")
	runORTPipelineGoMLX(t, testutil.ZeroShotObjectDetectionPipeline)
}

func TestZeroShotObjectDetectionPipelineORTCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.ZeroShotObjectDetectionPipeline)
}

func TestZeroShotObjectDetectionPipelineValidationORT(t *testing.T) {
	runORTPipelineValidation(t, testutil.ZeroShotObjectDetectionPipelineValidation)
}

// Zero-shot image classification

func TestZeroShotImageClassificationPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.ZeroShotImageClassificationPipeline)
}

func TestZeroShotImageClassificationPipelineORTGoMLX(t *testing.T) {
	t.Skip("Skipping test due to known backend \"onnxruntime\" does not support functions (needed for closures) issue with GoMLX")
	runORTPipelineGoMLX(t, testutil.ZeroShotImageClassificationPipeline)
}

func TestZeroShotImageClassificationPipelineORTCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.ZeroShotImageClassificationPipeline)
}

func TestZeroShotImageClassificationPipelineValidationORT(t *testing.T) {
	runORTPipelineValidation(t, testutil.ZeroShotImageClassificationPipelineValidation)
}

// Image feature extraction

func TestImageFeatureExtractionPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.ImageFeatureExtractionPipeline)
}

func TestImageFeatureExtractionPipelineORTGoMLX(t *testing.T) {
	runORTPipelineGoMLX(t, testutil.ImageFeatureExtractionPipeline)
}

func TestImageFeatureExtractionPipelineORTCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.ImageFeatureExtractionPipeline)
}

func TestImageFeatureExtractionPipelineValidationORT(t *testing.T) {
	runORTPipelineValidation(t, testutil.ImageFeatureExtractionPipelineValidation)
}

// Image segmentation

func TestImageSegmentationPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.ImageSegmentationPipeline)
}

func TestImageSegmentationPipelineORTGoMLX(t *testing.T) {
	t.Skip("Skipping test due to known Clamp (22) not implemented for ONNX Runtime backend: op not implemented issue with GoMLX")
	runORTPipelineGoMLX(t, testutil.ImageSegmentationPipeline)
}

func TestImageSegmentationPipelineORTCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.ImageSegmentationPipeline)
}

func TestImageSegmentationPipelineValidationORT(t *testing.T) {
	runORTPipelineValidation(t, testutil.ImageSegmentationPipelineValidation)
}

// Depth estimation

func TestDepthEstimationPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.DepthEstimationPipeline)
}

func TestDepthEstimationPipelineORTGoMLX(t *testing.T) {
	t.Skip("Skipping test due to known compilation failed for Exec issue with GoMLX")
	runORTPipelineGoMLX(t, testutil.DepthEstimationPipeline)
}

func TestDepthEstimationPipelineORTCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.DepthEstimationPipeline)
}

func TestDepthEstimationPipelineValidationORT(t *testing.T) {
	runORTPipelineValidation(t, testutil.DepthEstimationPipelineValidation)
}

// Mask generation

func TestMaskGenerationPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.MaskGenerationPipeline)
}

func TestMaskGenerationPipelineORTGoMLX(t *testing.T) {
	t.Skip("Skipping test due to known Clamp (22) not implemented for ONNX Runtime backend: op not implemented issue with GoMLX")
	runORTPipelineGoMLX(t, testutil.MaskGenerationPipeline)
}

func TestMaskGenerationPipelineORTCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.MaskGenerationPipeline)
}

func TestMaskGenerationPipelineValidationORT(t *testing.T) {
	runORTPipelineValidation(t, testutil.MaskGenerationPipelineValidation)
}

// Background removal

func TestBackgroundRemovalPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.BackgroundRemovalPipeline)
}

func TestBackgroundRemovalPipelineORTGoMLX(t *testing.T) {
	t.Skip("Skipping test due to known Clamp (22) not implemented for ONNX Runtime backend: op not implemented issue with GoMLX")
	runORTPipelineGoMLX(t, testutil.BackgroundRemovalPipeline)
}

func TestBackgroundRemovalPipelineORTCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.BackgroundRemovalPipeline)
}

func TestBackgroundRemovalPipelineValidationORT(t *testing.T) {
	runORTPipelineValidation(t, testutil.BackgroundRemovalPipelineValidation)
}
