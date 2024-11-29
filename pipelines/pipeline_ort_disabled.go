//go:build NOORT

package pipelines

type ORTSession struct {
	Destroy func() error
}

func createORTPipeline(_ *BasePipeline, _ []byte, _ any) error {
	return nil
}

func createInputTensorsORT(_ *PipelineBatch, _ []InputOutputInfo) error {
	return nil
}

func runORTSessionOnBatch(_ *PipelineBatch, _ *BasePipeline) error {
	return nil
}
