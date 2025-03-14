//go:build !XLA && !ALL

package hugot

import (
	"errors"

	"github.com/knights-analytics/hugot/pipelineBackends"
)

type XLATrainingOptions struct{}

func NewXLATrainingSession[_ pipelineBackends.Pipeline](_ TrainingConfig) (*TrainingSession, error) {
	return nil, errors.New("XLA is not enabled")
}

func TrainXLA(_ *TrainingSession) error {
	return errors.New("XLA is not enabled")
}
