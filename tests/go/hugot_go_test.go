//go:build (GO || ALL) && !TRAINING

package go_test

import (
	"encoding/json"
	"fmt"
	"os"
	"testing"

	"github.com/knights-analytics/hugot"
	testutil "github.com/knights-analytics/hugot/tests"
)

// FEATURE EXTRACTION

func TestFeatureExtractionPipelineGo(t *testing.T) {
	session, err := hugot.NewGoSession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.FeatureExtractionPipeline(t, session)
}

func TestFeatureExtractionPipelineValidationGo(t *testing.T) {
	session, err := hugot.NewGoSession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.FeatureExtractionPipelineValidation(t, session)
}

// Text classification

func TestTextClassificationPipelineGo(t *testing.T) {
	session, err := hugot.NewGoSession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.TextClassificationPipeline(t, session)
}

func TestTextClassificationPipelineMultiGo(t *testing.T) {
	session, err := hugot.NewGoSession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.TextClassificationPipelineMulti(t, session)
}

func TestTextClassificationPipelineValidationGo(t *testing.T) {
	session, err := hugot.NewGoSession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.TextClassificationPipelineValidation(t, session)
}

// Token classification

func TestTokenClassificationPipelineGo(t *testing.T) {
	session, err := hugot.NewGoSession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.TokenClassificationPipeline(t, session)
}

func TestTokenClassificationPipelineValidationGo(t *testing.T) {
	session, err := hugot.NewGoSession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.TokenClassificationPipelineValidation(t, session)
}

// Zero shot

func TestZeroShotClassificationPipelineGo(t *testing.T) {
	session, err := hugot.NewGoSession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.ZeroShotClassificationPipeline(t, session)
}

func TestZeroShotClassificationPipelineValidationGo(t *testing.T) {
	session, err := hugot.NewGoSession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.ZeroShotClassificationPipelineValidation(t, session)
}

// Cross Encoder

func TestCrossEncoderPipelineGo(t *testing.T) {
	session, err := hugot.NewGoSession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.CrossEncoderPipeline(t, session)
}

func TestCrossEncoderPipelineValidationGo(t *testing.T) {
	session, err := hugot.NewGoSession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.CrossEncoderPipelineValidation(t, session)
}

// Image classification

func TestImageClassificationPipelineGo(t *testing.T) {
	session, err := hugot.NewGoSession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.ImageClassificationPipeline(t, session)
}

func TestImageClassificationPipelineValidationGo(t *testing.T) {
	session, err := hugot.NewGoSession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.ImageClassificationPipelineValidation(t, session)
}

// Object detection

func TestObjectDetectionPipelineGo(t *testing.T) {
	session, err := hugot.NewGoSession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.ObjectDetectionPipeline(t, session)
}

func TestObjectDetectionPipelineValidationGo(t *testing.T) {
	session, err := hugot.NewGoSession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.ObjectDetectionPipelineValidation(t, session)
}

// text generation

func TestTextGenerationPipelineGo(t *testing.T) {
	t.Skip("Generative models are not supported yet for Go")
	session, err := hugot.NewGoSession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.TextGenerationPipeline(t, session)
}

func TestTextGenerationPipelineValidationGo(t *testing.T) {
	t.Skip("Generative models are not supported yet for Go")
	session, err := hugot.NewGoSession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.TextGenerationPipelineValidation(t, session)
}

// Tabular

func TestTabularPipelineGo(t *testing.T) {
	t.Skip("Currently missing TreeEnsembleClassifier ONNX operator")
	session, err := hugot.NewGoSession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.TabularPipeline(t, session)
}

// QA

func TestQAPipelineGo(t *testing.T) {
	session, err := hugot.NewGoSession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.QuestionAnsweringPipeline(t, session)
}

// No same name

func TestNoSameNamePipelineGo(t *testing.T) {
	session, err := hugot.NewGoSession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.NoSameNamePipeline(t, session)
}

func TestDestroyPipelineGo(t *testing.T) {
	session, err := hugot.NewGoSession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.DestroyPipelines(t, session)
}

// Thread safety

func TestThreadSafetyGo(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	session, err := hugot.NewGoSession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.ThreadSafety(t, session, 20)
}

// README: test the readme examples

func TestReadmeExample(t *testing.T) {
	t.Helper()
	check := func(err error) {
		if err != nil {
			panic(err.Error())
		}
	}

	// start a new session
	session, err := hugot.NewGoSession(t.Context())
	// For XLA (requires go build tags "XLA" or "ALL"):
	// session, err := NewXLASession()
	// For ORT (requires go build tags "ORT" or "ALL"):
	// session, err := NewORTSession()
	// This looks for the onnxruntime.so library in its default path, e.g. /usr/lib/onnxruntime.so
	// If your onnxruntime.so is somewhere else, you can explicitly set it by using WithOnnxLibraryPath
	// session, err := hugot.NewORTSession(WithOnnxLibraryPath("/path/to/onnxruntime.so"))
	check(err)

	// A successfully created hugot session needs to be destroyed when you're done
	defer func(session *hugot.Session) {
		err := session.Destroy()
		check(err)
	}(session)

	// Let's download an onnx sentiment test classification model in the current directory
	// note: if you compile your library with build flag NODOWNLOAD, this will exclude the downloader.
	// Useful in case you just want the core engine (because you already have the models) and want to
	// drop the dependency on huggingfaceModelDownloader.
	// modelPath, err := DownloadModel("KnightsAnalytics/distilbert-base-uncased-finetuned-sst-2-english", "./models/", NewDownloadOptions())
	// check(err)
	modelPath := "./models/KnightsAnalytics_distilbert-base-uncased-finetuned-sst-2-english"

	// we now create the configuration for the text classification pipeline we want to create.
	// Options to the pipeline can be set here using the Options field
	config := hugot.TextClassificationConfig{
		ModelPath: modelPath,
		Name:      "testPipeline",
	}
	// then we create out pipeline.
	// Note: the pipeline will also be added to the session object so all pipelines can be destroyed at once
	sentimentPipeline, err := hugot.NewPipeline(session, config)
	check(err)

	// we can now use the pipeline for prediction on a batch of strings
	batch := []string{"This movie is disgustingly good !", "The director tried too much"}
	batchResult, err := sentimentPipeline.RunPipeline(t.Context(), batch)
	check(err)

	// and do whatever we want with it :)
	s, err := json.Marshal(batchResult)
	check(err)
	fmt.Println(string(s))
	// OUTPUT: {"ClassificationOutputs":[[{"Label":"POSITIVE","Score":0.9998536}],[{"Label":"NEGATIVE","Score":0.99752176}]]}
}
