package hugot

import (
	"context"

	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/options"
)

func NewORTTrainingSession[T backends.Pipeline](ctx context.Context, config TrainingConfig) (*TrainingSession, error) {
	s, err := newTrainingSession[T](ctx, options.BackendORT, config)
	if err != nil {
		return nil, err
	}

	return newGoMLXTrainingSession(s)
}
