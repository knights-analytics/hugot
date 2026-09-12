//go:build !cgo || (!ORT && !ALL)

package backends

import (
	"context"
	"errors"

	"github.com/knights-analytics/hugot/options"
)

type ORTModel struct {
	Generative *generativeORTAdapter
}

type generativeORTAdapter struct{}

func (*generativeORTAdapter) Statistics() PipelineStatistics { return PipelineStatistics{} }

func (m *ORTModel) Close() error { return nil }

func createORTModelBackend(_ *Model, _ *options.Options) error {
	return errors.New("ORT model execution is not available on this platform")
}

func createInputTensorsORT(_ *PipelineBatch, _ *Model) error {
	return errors.New("ORT model execution is not available on this platform")
}

func runORTSessionOnBatch(_ context.Context, _ *PipelineBatch, _ *BasePipeline) error {
	return errors.New("ORT model execution is not available on this platform")
}

func createImageTensorsORT(_ *PipelineBatch, _ *Model, _ [][][][]float32) error {
	return errors.New("ORT model execution is not available on this platform")
}

func createAudioTensorsORT(_ *PipelineBatch, _ *Model, _ [][]float32) error {
	return errors.New("ORT model execution is not available on this platform")
}

func createTabularTensorsORT(_ *PipelineBatch, _ *Model, _ [][]float32) error {
	return errors.New("ORT model execution is not available on this platform")
}

func runGenerativeORTSessionOnBatch(_ context.Context, _ *PipelineBatch, _ *BasePipeline, _ int, _ []string, _ *float64, _ *float64, _ *int, _ []string, _ *Guidance) (chan SequenceDelta, chan error, error) {
	return nil, nil, errors.New("ORT generative inference is not available on this platform")
}

func createORTGenerativeSession(_ context.Context, _ *Model, _ *options.Options) error {
	return errors.New("ORT generative inference is not available on this platform")
}

func CreateMessagesORT(_ *PipelineBatch, _ any, _ string) error {
	return errors.New("ORT generative inference is not available on this platform")
}
