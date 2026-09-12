//go:build cgo && (ORT || ALL)

package hugot

import (
	"context"
	"errors"
	"sync"

	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/options"
)

var (
	ortSessionMu     sync.Mutex
	ortSessionActive bool
)

// NewORTSession creates a session on the ONNX Runtime backend. Only one native ORT session can be
// active at a time, because ONNX Runtime has a single environment per process.
func NewORTSession(ctx context.Context, opts ...options.WithOption) (*Session, error) {
	// Options are parsed first, because whether this session needs the ORT environment at all
	// depends on them. newSession has no global side effects, so nothing needs undoing on failure.
	session, err := newSession(ctx, options.BackendORT, opts...)
	if err != nil {
		return nil, err
	}
	if session.options.UseGoMLX {
		return session, nil
	}

	if !acquireORTSession() {
		return nil, errors.Join(
			errors.New("another session is currently active, and only one session can be active at one time"),
			session.Destroy())
	}

	// set session options and initialise
	if ortErr := session.initialiseORT(); ortErr != nil {
		destroyErr := session.Destroy()
		releaseORTSession()
		return nil, errors.Join(ortErr, destroyErr)
	}

	session.environmentDestroy = func() error {
		releaseORTSession()
		return nil
	}

	return session, err
}

func acquireORTSession() bool {
	ortSessionMu.Lock()
	defer ortSessionMu.Unlock()
	if ortSessionActive {
		return false
	}
	ortSessionActive = true
	return true
}

func releaseORTSession() {
	ortSessionMu.Lock()
	ortSessionActive = false
	ortSessionMu.Unlock()
}

func (s *Session) initialiseORT() error {
	return backends.InitializeORT(s.options)
}
