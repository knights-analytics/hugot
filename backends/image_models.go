package backends

import (
	"fmt"
)

func CreateImageTensors(batch *PipelineBatch, preprocessed [][][][]float32, runtime string) error {
	switch runtime {
	case "ORT":
		return createImageTensorsORT(batch, preprocessed)
	case "GO", "XLA":
		return createImageTensorsGoXLA(batch, preprocessed)
	default:
		return fmt.Errorf("runtime %s is not supported for image tensors", runtime)
	}
}
