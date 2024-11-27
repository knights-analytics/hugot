//go:build !GO && !ALL

package pipelines

type GoSession struct{}

func createGoPipeline(_ *BasePipeline, _ []byte, _ any) error {
	return nil
}

func createInputTensorsGo(_ *PipelineBatch, _ []InputOutputInfo) error {
	return nil
}

func runGoSessionOnBatch(_ *PipelineBatch, _ *BasePipeline) error {
	return nil
}
