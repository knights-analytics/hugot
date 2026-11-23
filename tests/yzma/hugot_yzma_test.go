//go:build (YZMA || ALL) && !TRAINING

package hugot

import (
	"os"
	"testing"

	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/pipelines"
	testutil "github.com/knights-analytics/hugot/tests"
)

func TestTextGenerationPipelineYZMA(t *testing.T) {
	if os.Getenv("YZMA_LIB_PATH") != "" {
		t.Skip("YZMA_LIB_PATH is set; custom library test covers this run")
	}
	modelPath := os.Getenv("YZMA_MODEL_PATH")
	if modelPath == "" {
		t.Skip("YZMA_MODEL_PATH must point to a local GGUF model")
	}

	t.Run("default session", func(t *testing.T) {
		session, err := hugot.NewYZMASession(t.Context())
		testutil.CheckT(t, err)
		defer func(session *hugot.Session) {
			destroyErr := session.Destroy()
			testutil.CheckT(t, destroyErr)
		}(session)
		pipeline, err := session.NewPipeline(hugot.TextGenerationConfig{
			ModelPath: modelPath,
			Name:      "yzmaTextGeneration",
			Options: []hugot.TextGenerationOption{
				pipelines.WithMaxLength(32),
			},
		})
		testutil.CheckT(t, err)
		result, err := pipeline.RunPipeline(t.Context(), []string{"Reply with a short greeting."})
		testutil.CheckT(t, err)
		if len(result.Responses) != 1 || result.Responses[0] == "" {
			t.Fatalf("expected one non-empty generated response, got %#v", result.Responses)
		}
	})
}

func TestTextGenerationPipelineYZMAWithLibPath(t *testing.T) {
	libPath := os.Getenv("YZMA_LIB_PATH")
	if libPath == "" {
		t.Skip("YZMA_LIB_PATH not set; skipping library-path override test")
	}
	modelPath := os.Getenv("YZMA_MODEL_PATH")
	if modelPath == "" {
		t.Skip("YZMA_MODEL_PATH must point to a local GGUF model")
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
		pipeline, err := session.NewPipeline(hugot.TextGenerationConfig{ModelPath: modelPath, Name: "yzmaCustomLib", Options: []hugot.TextGenerationOption{pipelines.WithMaxLength(8)}})
		testutil.CheckT(t, err)
		_, err = pipeline.RunPipeline(t.Context(), []string{"Hello"})
		testutil.CheckT(t, err)
	})
}

func TestImageTextToTextPipelineYZMA(t *testing.T) {
	modelPath := os.Getenv("YZMA_MODEL_PATH")
	imagePath := os.Getenv("YZMA_IMAGE_PATH")
	if modelPath == "" || imagePath == "" {
		t.Skip("YZMA_MODEL_PATH and YZMA_IMAGE_PATH must point to a vision GGUF bundle and image")
	}
	sessionOptions := []options.WithOption{}
	if libPath := os.Getenv("YZMA_LIB_PATH"); libPath != "" {
		sessionOptions = append(sessionOptions, options.WithYZMALibraryPath(libPath))
	}
	if projectorPath := os.Getenv("YZMA_PROJECTOR_PATH"); projectorPath != "" {
		sessionOptions = append(sessionOptions, options.WithYZMAProjectorPath(projectorPath))
	}
	session, err := hugot.NewYZMASession(t.Context(), sessionOptions...)
	testutil.CheckT(t, err)
	defer func() { testutil.CheckT(t, session.Destroy()) }()
	pipeline, err := session.NewPipeline(hugot.ImageTextToTextConfig{
		ModelPath: modelPath,
		Name:      "yzmaImageTextToText",
		Options:   []hugot.ImageTextToTextOption{pipelines.WithImageTextToTextMaxLength(32)},
	})
	testutil.CheckT(t, err)
	result, err := pipeline.RunWithImages(t.Context(), []pipelines.ImageTextPrompt{{
		ImagePath: imagePath,
		Prompt:    "Describe this image briefly.",
	}})
	testutil.CheckT(t, err)
	if len(result.Responses) != 1 || result.Responses[0] == "" {
		t.Fatalf("expected one non-empty generated response, got %#v", result.Responses)
	}
}
