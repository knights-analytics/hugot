//go:build !YZMA && !ALL

package backends

import (
	"context"
	"errors"

	"github.com/knights-analytics/hugot/options"
)

// YZMAModel is an empty placeholder when the YZMA backend is not enabled.
type YZMAModel struct{}

func createYZMAGenerativeSession(_ context.Context, _ *Model, _ *options.Options) error {
	return errors.New("YZMA backend is not enabled; build with -tags YZMA or -tags ALL")
}

func (y *YZMAModel) Generate(_ context.Context, _ [][]Message, _ []string, _ *GenerativeOptions) (chan SequenceDelta, chan error, error) {
	return nil, nil, errors.New("YZMA backend is not enabled; build with -tags YZMA or -tags ALL")
}

func (y *YZMAModel) GetStatistics() PipelineStatistics {
	return PipelineStatistics{}
}

func (y *YZMAModel) Destroy() error {
	return nil
}

func CreateMessagesYZMA(_ *PipelineBatch, _ any, _ string) error {
	return errors.New("YZMA backend is not enabled; build with -tags YZMA or -tags ALL")
}

func runGenerativeYZMAOnBatch(_ context.Context, _ *PipelineBatch, _ *BasePipeline, _ int, _ []string, _ *float64, _ *float64, _ *int, _ []string, _ *Guidance) (chan SequenceDelta, chan error, error) {
	return nil, nil, errors.New("YZMA backend is not enabled; build with -tags YZMA or -tags ALL")
}
