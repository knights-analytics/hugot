//go:build NOORT

package pipelines

type ORTSession struct {
	Destroy func() error
}

func createORTPipeline(_ *basePipeline, _ []byte, _ any) error {
	return nil
}

func createInputTensorsORT(_ *PipelineBatch, _ []InputOutputInfo) error {
	return nil
}

func runORTSessionOnBatch(_ *PipelineBatch, _ *basePipeline) error {
	return nil
}
