//go:build !XLA && !ALL

package pipelines

import (
	_ "github.com/gomlx/gomlx/backends/xla/cpu/static"

	"github.com/knights-analytics/hugot/options"
)

type XLAModel struct {
	Destroy func()
}

func createXLAModelBackend(_ *Model, _ *options.Options) error {
	return nil
}

func createInputTensorsXLA(_ *PipelineBatch, _ []InputOutputInfo) error {
	return nil
}

func runXLASessionOnBatch(_ *PipelineBatch, _ *BasePipeline) error {
	return nil
}
