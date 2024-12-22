//go:build !XLA && !ALL

package pipelineBackends

import (
	"github.com/knights-analytics/hugot/options"
)

type XLAModel struct {
	Destroy func()
}

func createXLAModelBackend(_ *Model, _ *options.Options) error {
	return nil
}

func createInputTensorsXLA(_ *PipelineBatch, _ []InputOutputInfo, _ bool) error {
	return nil
}

func runXLASessionOnBatch(_ *PipelineBatch, _ *BasePipeline) error {
	return nil
}

func (xlaModel *XLAModel) Save(_ string) error {
	return nil
}
