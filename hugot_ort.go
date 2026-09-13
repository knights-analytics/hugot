//go:build cgo && (ORT || ALL)

package hugot

import (
	"context"
	"errors"
	"sync"

	_ "github.com/gomlx/compute-onnx"

	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/options"
)

var (
	ortSessionMu     sync.Mutex
	ortSessionActive bool
)

func NewORTSession(ctx context.Context, opts ...options.WithOption) (*Session, error) {
	if !acquireORTSession() {
		return nil, errors.New("another session is currently active, and only one session can be active at one time")
	}

	session, err := newSession(ctx, options.BackendORT, opts...)
	if err != nil {
		releaseORTSession()
		return nil, err
	}
	if session.options.UseGoMLX {
		releaseORTSession()
		return session, nil
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
