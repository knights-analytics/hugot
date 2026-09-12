//go:build cgo && (ORT || ALL) && !TRAINING

package ort_test

import (
	"os"
	"testing"

	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/options"
	testutil "github.com/knights-analytics/hugot/tests"
)

func runORTPipeline(t *testing.T, run func(*testing.T, *hugot.Session), opts ...options.WithOption) {
	t.Helper()
	session, err := hugot.NewORTSession(t.Context(), opts...)
	testutil.CheckT(t, err)
	defer func() { testutil.CheckT(t, session.Destroy()) }()
	run(t, session)
}

func runORTPipelineGoMLX(t *testing.T, run func(*testing.T, *hugot.Session), opts ...options.WithOption) {
	t.Helper()
	opts = append([]options.WithOption{options.WithGoMLX()}, opts...)
	runORTPipeline(t, run, opts...)
}

func runORTPipelineValidation(t *testing.T, run func(*testing.T, *hugot.Session), opts ...options.WithOption) {
	t.Helper()
	runORTPipeline(t, run, opts...)
}

func runORTPipelineCuda(t *testing.T, run func(*testing.T, *hugot.Session), opts ...options.WithOption) {
	t.Helper()
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	opts = append([]options.WithOption{options.WithCuda(map[string]string{"device_id": "0"})}, opts...)
	runORTPipeline(t, run, opts...)
}

var textClassificationORTOptions = []options.WithOption{
	options.WithTelemetry(),
	options.WithCPUMemArena(true),
	options.WithMemPattern(true),
	options.WithIntraOpNumThreads(1),
	options.WithInterOpNumThreads(1),
}
