//go:build cgo && (XLA || ALL) && !TRAINING

package xla_test

import (
	"os"
	"testing"

	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/options"
	testutil "github.com/knights-analytics/hugot/tests"
)

func runXLAPipeline(t *testing.T, run func(*testing.T, *hugot.Session), opts ...options.WithOption) {
	t.Helper()
	session, err := hugot.NewXLASession(t.Context(), opts...)
	testutil.CheckT(t, err)
	defer func() { testutil.CheckT(t, session.Destroy()) }()
	run(t, session)
}

func runXLAPipelineValidation(t *testing.T, run func(*testing.T, *hugot.Session), opts ...options.WithOption) {
	t.Helper()
	runXLAPipeline(t, run, opts...)
}

func runXLAPipelineCuda(t *testing.T, run func(*testing.T, *hugot.Session), opts ...options.WithOption) {
	t.Helper()
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	opts = append([]options.WithOption{options.WithCuda(map[string]string{"device_id": "0"})}, opts...)
	runXLAPipeline(t, run, opts...)
}
