//go:build !XLA && !ALL

package pipelines

type XLASession struct {
	Destroy func()
}

func createXLAPipeline(_ *basePipeline, _ []byte, _ any) error {
	return nil
}

func createInputTensorsXLA(_ *PipelineBatch, _ []InputOutputInfo) error {
	return nil
}

func runXLASessionOnBatch(_ *PipelineBatch, _ *basePipeline) error {
	return nil
}
