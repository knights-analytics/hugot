//go:build XLA || ALL

package hugot

import (
	"github.com/knights-analytics/hugot/options"
)

func NewXLASession(opts ...options.WithOption) (*Session, error) {
	return newSession("XLA", nil, opts...)
}
