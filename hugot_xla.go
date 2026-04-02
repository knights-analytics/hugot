//go:build cgo && (XLA || ALL)

package hugot

import (
	"context"

	"github.com/gomlx/gomlx/backends/xla" // import XLA backend

	"github.com/knights-analytics/hugot/options"
)

func NewXLASession(ctx context.Context, opts ...options.WithOption) (*Session, error) {
	// Disabled for now until we have auto installs globally
	xlaDisableAutoInstall()
	return newSession(ctx, "XLA", opts...)
}

func xlaDisableAutoInstall() {
	xla.EnableAutoInstall(false)
}
