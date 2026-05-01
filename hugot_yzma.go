//go:build YZMA || ALL

package hugot

import (
	"context"
	"sync"
	"sync/atomic"

	"github.com/knights-analytics/hugot/options"
)

// yzmaSessionCount tracks the number of active YZMA sessions.
// yzma allows multiple sessions sharing the same loaded library.
var (
	yzmaSessionCount atomic.Int32
	yzmaSessionMu    sync.Mutex
)

// NewYZMASession creates a new hugot session backed by the YZMA (llama.cpp) inference engine.
// Multiple sessions may coexist; the llama.cpp shared library is loaded once per process.
//
// Example:
//
//	session, err := hugot.NewYZMASession(ctx,
//	    hugot.WithYZMALibraryPath("/usr/local/lib/llama"),
//	)
func NewYZMASession(ctx context.Context, opts ...options.WithOption) (*Session, error) {
	session, err := newSession(ctx, "YZMA", opts...)
	if err != nil {
		return nil, err
	}

	yzmaSessionCount.Add(1)

	session.environmentDestroy = func() error {
		yzmaSessionCount.Add(-1)
		return nil
	}

	return session, nil
}
