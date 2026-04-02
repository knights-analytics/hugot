//go:build !cgo || (!XLA && !ALL)

package hugot

import (
	"context"
	"errors"

	"github.com/knights-analytics/hugot/options"
)

func NewXLASession(_ context.Context, _ ...options.WithOption) (*Session, error) {
	return nil, errors.New("to enable XLA, run `go build -tags XLA` or `go build -tags ALL`")
}

func xlaDisableAutoInstall() {}
