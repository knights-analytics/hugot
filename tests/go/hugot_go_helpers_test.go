//go:build (GO || ALL) && !TRAINING

package go_test

import (
	"testing"

	"github.com/knights-analytics/hugot"
	testutil "github.com/knights-analytics/hugot/tests"
)

func runGoPipeline(t *testing.T, run func(*testing.T, *hugot.Session)) {
	t.Helper()
	session, err := hugot.NewGoSession(t.Context())
	testutil.CheckT(t, err)
	defer func() { testutil.CheckT(t, session.Destroy()) }()
	run(t, session)
}

func runGoPipelineValidation(t *testing.T, run func(*testing.T, *hugot.Session)) {
	t.Helper()
	runGoPipeline(t, run)
}
