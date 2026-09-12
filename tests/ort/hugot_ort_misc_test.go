//go:build cgo && (ORT || ALL) && !TRAINING

package ort_test

import (
	"testing"

	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/options"
	testutil "github.com/knights-analytics/hugot/tests"
)

// Tabular pipeline

func TestTabularPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.TabularPipeline)
}

func TestTabularPipelineORTGoMLX(t *testing.T) {
	t.Skip("Currently missing TreeEnsembleClassifier ONNX operator")
	runORTPipelineGoMLX(t, testutil.TabularPipeline)
}

func TestTabularPipelineORTCuda(t *testing.T) {
	runORTPipelineCuda(t, testutil.TabularPipeline)
}

func TestTabularPipelineValidationORT(t *testing.T) {
	runORTPipelineValidation(t, testutil.TabularPipelineValidation)
}

// Thread safety

func TestThreadSafetyORT(t *testing.T) {
	runORTPipeline(t, func(t *testing.T, session *hugot.Session) {
		testutil.ThreadSafety(t, session, 250)
	})
}

func TestThreadSafetyGoMLX(t *testing.T) {
	runORTPipelineGoMLX(t, func(t *testing.T, session *hugot.Session) {
		testutil.ThreadSafety(t, session, 250)
	})
}

func TestThreadSafetyORTCuda(t *testing.T) {
	runORTPipelineCuda(t, func(t *testing.T, session *hugot.Session) {
		testutil.ThreadSafety(t, session, 1000)
	})
}

// No same name

func TestNoSameNamePipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.NoSameNamePipeline)
}

func TestNoSameNameAcrossTypesPipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.NoSameNameAcrossTypesPipeline)
}

func TestClosePipelineORT(t *testing.T) {
	runORTPipeline(t, testutil.DestroyPipelines)
}

// Ort and GoMLX co-existence

// Only one native ORT session can be active at a time. A GoMLX session is allowed alongside a native one.
func TestORTSessionLimitIgnoresGoMLX(t *testing.T) {
	native, err := hugot.NewORTSession(t.Context())
	testutil.CheckT(t, err)

	goMLX, err := hugot.NewORTSession(t.Context(), options.WithGoMLX())
	testutil.CheckT(t, err)
	testutil.CheckT(t, goMLX.Destroy())

	if second, secondErr := hugot.NewORTSession(t.Context()); secondErr == nil {
		testutil.CheckT(t, second.Destroy())
		t.Fatal("a second native ORT session was created while one was active")
	}

	// destroying the native session frees the slot for the next one
	testutil.CheckT(t, native.Destroy())
	native, err = hugot.NewORTSession(t.Context())
	testutil.CheckT(t, err)
	testutil.CheckT(t, native.Destroy())
}
