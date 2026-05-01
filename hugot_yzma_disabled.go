//go:build !YZMA && !ALL

package hugot

import (
	"context"
	"errors"

	"github.com/knights-analytics/hugot/options"
)

func NewYZMASession(_ context.Context, _ ...options.WithOption) (*Session, error) {
	return nil, errors.New("to enable YZMA, run `go build -tags YZMA` or `go build -tags ALL`")
}
