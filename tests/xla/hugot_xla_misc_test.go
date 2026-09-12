//go:build cgo && (XLA || ALL) && !TRAINING

package xla_test

import (
	"testing"

	"github.com/knights-analytics/hugot"
	testutil "github.com/knights-analytics/hugot/tests"
)

// Tabular pipeline

func TestTabularPipelineXLA(t *testing.T) {
	t.Skip("Currently missing TreeEnsembleClassifier ONNX operator")
	runXLAPipeline(t, testutil.TabularPipeline)
}

func TestTabularPipelineXLACuda(t *testing.T) {
	t.Skip("Currently missing TreeEnsembleClassifier ONNX operator")
	runXLAPipelineCuda(t, testutil.TabularPipeline)
}

func TestTabularPipelineValidationXLA(t *testing.T) {
	runXLAPipelineValidation(t, testutil.TabularPipelineValidation)
}

// Thread safety

func TestThreadSafetyXLA(t *testing.T) {
	runXLAPipeline(t, func(t *testing.T, session *hugot.Session) {
		testutil.ThreadSafety(t, session, 250)
	})
}

func TestThreadSafetyXLACuda(t *testing.T) {
	runXLAPipelineCuda(t, func(t *testing.T, session *hugot.Session) {
		testutil.ThreadSafety(t, session, 1000)
	})
}

// No same name

func TestNoSameNamePipelineXLA(t *testing.T) {
	runXLAPipeline(t, testutil.NoSameNamePipeline)
}

func TestNoSameNameAcrossTypesPipelineXLA(t *testing.T) {
	runXLAPipeline(t, testutil.NoSameNameAcrossTypesPipeline)
}

func TestDestroyPipelineXLA(t *testing.T) {
	runXLAPipeline(t, testutil.DestroyPipelines)
}
