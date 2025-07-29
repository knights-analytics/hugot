//go:build !ORT && !ALL

package pipelineBackends

import (
	"errors"

	"github.com/knights-analytics/hugot/options"
)

type ORTModel struct {
	Destroy func() error
}

func createORTModelBackend(_ *Model, _ *options.Options) error {
	return errors.New("ORT is not enabled")
}

func createInputTensorsORT(_ *PipelineBatch, _ []InputOutputInfo) error {
	return errors.New("ORT is not enabled")
}

func runORTSessionOnBatch(_ *PipelineBatch, _ *BasePipeline) error {
	return errors.New("ORT is not enabled")
}

func runGenerativeORTSessionOnBatch(_ *PipelineBatch, _ *BasePipeline) error {
	return errors.New("ORT is not enabled")
}
