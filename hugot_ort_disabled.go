//go:build !ORT && !ALL

package hugot

import (
	"errors"

	"github.com/knights-analytics/hugot/options"
)

func NewORTSession(_ ...options.WithOption) (*Session, error) {
	return nil, errors.New("to enable ORT, run `go build -tags ORT` or `go build -tags ALL`")
}
