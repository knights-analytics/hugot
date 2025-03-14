//go:build !XLA && !ALL

package pipelineBackends

import (
	"errors"
	"io"

	"github.com/knights-analytics/hugot/options"
)

type XLAModel struct {
	Destroy func()
}

func createXLAModelBackend(_ *Model, _ *options.Options) error {
	return errors.New("XLA is not enabled")
}

func createInputTensorsXLA(_ *PipelineBatch, _ []InputOutputInfo, _ bool) error {
	return errors.New("XLA is not enabled")
}

func runXLASessionOnBatch(_ *PipelineBatch, _ *BasePipeline) error {
	return errors.New("XLA is not enabled")
}

func (xlaModel *XLAModel) Save(_ io.WriteCloser) error {
	return errors.New("XLA is not enabled")
}
