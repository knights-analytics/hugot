//go:build cgo && (XLA || ALL)

package hugot

import (
	"context"

	_ "github.com/gomlx/go-xla/compute/xla" // import XLA backend

	"github.com/knights-analytics/hugot/options"
)

func NewXLASession(ctx context.Context, opts ...options.WithOption) (*Session, error) {
	return newSession(ctx, "XLA", opts...)
}
