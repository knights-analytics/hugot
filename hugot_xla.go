//go:build XLA || ALL

package hugot

import (
	"github.com/gomlx/gomlx/backends/xla" // import XLA backend

	"github.com/knights-analytics/hugot/options"
)

func NewXLASession(opts ...options.WithOption) (*Session, error) {
	// Disabled for now until we have auto installs globally
	xlaDisableAutoInstall()
	return newSession("XLA", opts...)
}

func xlaDisableAutoInstall() {
	xla.EnableAutoInstall(false)
}
