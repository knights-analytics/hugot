package hugot

import (
	"context"

	_ "github.com/gomlx/gomlx/backends/simplego" // Import simplego backend

	"github.com/knights-analytics/hugot/options"
)

func NewGoSession(ctx context.Context, opts ...options.WithOption) (*Session, error) {
	return newSession(ctx, "GO", opts...)
}
