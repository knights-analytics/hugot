package hugot

import (
	_ "github.com/gomlx/gomlx/backends/simplego" // Import simplego backend

	"github.com/knights-analytics/hugot/options"
)

func NewGoSession(opts ...options.WithOption) (*Session, error) {
	return newSession("GO", opts...)
}
