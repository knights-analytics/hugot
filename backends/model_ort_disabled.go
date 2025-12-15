//go:build !ORT && !ALL

package backends

import (
	"context"
	"errors"

	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/ortgenai"
)

type ORTModel struct {
	Destroy           func() error
	GenerativeSession *ortgenai.Session
}

func createORTModelBackend(_ *Model, _ *options.Options) error {
	return errors.New("ORT is not enabled")
}

func createInputTensorsORT(_ *PipelineBatch, _ *Model) error {
	return errors.New("ORT is not enabled")
}

func runORTSessionOnBatch(_ *PipelineBatch, _ *BasePipeline) error {
	return errors.New("ORT is not enabled")
}

func createImageTensorsORT(_ *PipelineBatch, _ *Model, _ [][][][]float32) error {
	return errors.New("ORT is not enabled")
}

func runGenerativeORTSessionOnBatch(_ context.Context, _ *PipelineBatch, _ *BasePipeline, _ int) (chan SequenceDelta, chan error, error) {
	return nil, nil, errors.New("ORT is not enabled")
}

func createORTGenerativeSession(_ *Model, _ *options.Options) error {
	return errors.New("ORT is not enabled")
}

func CreateMessagesORT(_ *PipelineBatch, _ any) error {
	return errors.New("ORT is not enabled")
}
