package pipelines

import "testing"

func TestZeroShotObjectDetectionIsNotGenerative(t *testing.T) {
	pipeline := &ZeroShotObjectDetectionPipeline{}
	if pipeline.IsGenerative() {
		t.Fatal("zero-shot object detection pipeline must not be generative")
	}
}