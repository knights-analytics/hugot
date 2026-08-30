package hugot

import (
	"context"

	_ "github.com/gomlx/compute/gobackend" // Import gobackend

	"github.com/knights-analytics/hugot/options"
)

func NewGoSession(ctx context.Context, opts ...options.WithOption) (*Session, error) {
	return newSession(ctx, options.BackendGo, opts...)
}
