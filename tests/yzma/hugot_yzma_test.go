//go:build (YZMA || ALL) && !TRAINING

package hugot

import (
	"os"
	"testing"

	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/options"
	testutil "github.com/knights-analytics/hugot/tests"
)

func TestTextGenerationPipelineYZMA(t *testing.T) {
	// if os.Getenv("CI") != "" {
	// 	t.SkipNow()
	// }

	t.Run("default session", func(t *testing.T) {
		session, err := hugot.NewYZMASession(t.Context())
		testutil.CheckT(t, err)
		defer func(session *hugot.Session) {
			destroyErr := session.Destroy()
			testutil.CheckT(t, destroyErr)
		}(session)
		testutil.TextGenerationPipeline(t, session)
	})
}

func TestTextGenerationPipelineYZMAWithLibPath(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	libPath := os.Getenv("YZMA_LIB_PATH")
	if libPath == "" {
		t.Skip("YZMA_LIB_PATH not set; skipping library-path override test")
	}

	t.Run("custom lib path", func(t *testing.T) {
		session, err := hugot.NewYZMASession(t.Context(),
			options.WithYZMALibraryPath(libPath),
		)
		testutil.CheckT(t, err)
		defer func(session *hugot.Session) {
			destroyErr := session.Destroy()
			testutil.CheckT(t, destroyErr)
		}(session)
		testutil.TextGenerationPipeline(t, session)
	})
}

func TestTextGenerationPipelineValidationYZMA(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}

	t.Run("validation", func(t *testing.T) {
		session, err := hugot.NewYZMASession(t.Context())
		testutil.CheckT(t, err)
		defer func(session *hugot.Session) {
			destroyErr := session.Destroy()
			testutil.CheckT(t, destroyErr)
		}(session)
		testutil.TextGenerationPipelineValidation(t, session)
	})
}
