package hugot

import (
	"encoding/json"
	"fmt"
	"testing"
)

// FEATURE EXTRACTION

func TestFeatureExtractionPipelineGo(t *testing.T) {
	session, err := NewGoSession()
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	featureExtractionPipeline(t, session)
}

func TestFeatureExtractionPipelineValidationGo(t *testing.T) {
	session, err := NewGoSession()
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	featureExtractionPipelineValidation(t, session)
}

// Text classification

func TestTextClassificationPipelineGo(t *testing.T) {
	session, err := NewGoSession()
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	textClassificationPipeline(t, session)
}

func TestTextClassificationPipelineMultiGo(t *testing.T) {
	session, err := NewGoSession()
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	textClassificationPipelineMulti(t, session)
}

func TestTextClassificationPipelineValidationGo(t *testing.T) {
	session, err := NewGoSession()
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	textClassificationPipelineValidation(t, session)
}

// Token classification

func TestTokenClassificationPipelineGo(t *testing.T) {
	session, err := NewGoSession()
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	tokenClassificationPipeline(t, session)
}

func TestTokenClassificationPipelineValidationGo(t *testing.T) {
	session, err := NewGoSession()
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	tokenClassificationPipelineValidation(t, session)
}

// Zero shot
//
// func TestZeroShotClassificationPipelineGo(t *testing.T) {
//	session, err := NewGoSession()
//	check(t, err)
//	defer func(session *Session) {
//		destroyErr := session.Destroy()
//		check(t, destroyErr)
//	}(session)
//	zeroShotClassificationPipeline(t, session)
// }
//
// func TestZeroShotClassificationPipelineValidationGo(t *testing.T) {
//	session, err := NewGoSession()
//	check(t, err)
//	defer func(session *Session) {
//		destroyErr := session.Destroy()
//		check(t, destroyErr)
//	}(session)
//	zeroShotClassificationPipelineValidation(t, session)
// }

// text generation
func TestTextGenerationPipelineGo(t *testing.T) {
	session, err := NewGoSession()
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	textGenerationPipeline(t, session)
}

func TestTextGenerationPipelineValidationGo(t *testing.T) {
	session, err := NewGoSession()
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	textGenPipelineValidation(t, session)
}

// No same name

func TestNoSameNamePipelineGo(t *testing.T) {
	session, err := NewGoSession()
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	noSameNamePipeline(t, session)
}

func TestDestroyPipelineGo(t *testing.T) {
	session, err := NewGoSession()
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	destroyPipelines(t, session)
}

// Thread safety

func TestThreadSafetyGo(t *testing.T) {
	session, err := NewGoSession()
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	threadSafety(t, session, 50)
}

// README: test the readme examples

func TestReadmeExample(t *testing.T) {
	check := func(err error) {
		if err != nil {
			panic(err.Error())
		}
	}

	// start a new session
	session, err := NewGoSession()
	// For XLA (requires go build tags "XLA" or "ALL"):
	// session, err := NewXLASession()
	// For ORT (requires go build tags "ORT" or "ALL"):
	// session, err := NewORTSession()
	// This looks for the onnxruntime.so library in its default path, e.g. /usr/lib/onnxruntime.so
	// If your onnxruntime.so is somewhere else, you can explicitly set it by using WithOnnxLibraryPath
	// session, err := hugot.NewORTSession(WithOnnxLibraryPath("/path/to/onnxruntime.so"))
	check(err)

	// A successfully created hugot session needs to be destroyed when you're done
	defer func(session *Session) {
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
	config := TextClassificationConfig{
		ModelPath: modelPath,
		Name:      "testPipeline",
	}
	// then we create out pipeline.
	// Note: the pipeline will also be added to the session object so all pipelines can be destroyed at once
	sentimentPipeline, err := NewPipeline(session, config)
	check(err)

	// we can now use the pipeline for prediction on a batch of strings
	batch := []string{"This movie is disgustingly good !", "The director tried too much"}
	batchResult, err := sentimentPipeline.RunPipeline(batch)
	check(err)

	// and do whatever we want with it :)
	s, err := json.Marshal(batchResult)
	check(err)
	fmt.Println(string(s))
	// OUTPUT: {"ClassificationOutputs":[[{"Label":"POSITIVE","Score":0.9998536}],[{"Label":"NEGATIVE","Score":0.99752176}]]}
}
