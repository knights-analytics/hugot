//go:build NOORT && !ALL

package pipelineBackends

import (
	"github.com/knights-analytics/hugot/options"
)

type ORTModel struct {
	Destroy func() error
}

func createORTModelBackend(_ *Model, _ *options.Options) error {
	return nil
}

func createInputTensorsORT(_ *PipelineBatch, _ []InputOutputInfo) error {
	return nil
}

func runORTSessionOnBatch(_ *PipelineBatch, _ *BasePipeline) error {
	return nil
}
