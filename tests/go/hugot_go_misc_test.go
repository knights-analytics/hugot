//go:build (GO || ALL) && !TRAINING

package go_test

import (
	"encoding/json"
	"fmt"
	"testing"

	"github.com/knights-analytics/hugot"
	testutil "github.com/knights-analytics/hugot/tests"
)

// Tabular pipeline

func TestTabularPipelineGo(t *testing.T) {
	t.Skip("Currently missing TreeEnsembleClassifier ONNX operator")
	runGoPipeline(t, testutil.TabularPipeline)
}

func TestTabularPipelineValidationGo(t *testing.T) {
	runGoPipelineValidation(t, testutil.TabularPipelineValidation)
}

// Thread safety

func TestThreadSafetyGo(t *testing.T) {
	runGoPipeline(t, func(t *testing.T, session *hugot.Session) {
		testutil.ThreadSafety(t, session, 20)
	})
}

// No same name

func TestNoSameNamePipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.NoSameNamePipeline)
}

func TestNoSameNameAcrossTypesPipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.NoSameNameAcrossTypesPipeline)
}

func TestDestroyPipelineGo(t *testing.T) {
	runGoPipeline(t, testutil.DestroyPipelines)
}

// README: test the readme examples

func TestReadmeExample(t *testing.T) {
	t.Helper()
	check := func(err error) {
		if err != nil {
			panic(err.Error())
		}
	}

	session, err := hugot.NewGoSession(t.Context())
	check(err)

	defer func(session *hugot.Session) {
		err := session.Destroy()
		check(err)
	}(session)

	modelPath := testutil.ModelsFolder + "Xenova_distilbert-base-uncased-finetuned-sst-2-english"

	config := hugot.TextClassificationConfig{
		ModelPath: modelPath,
		Name:      "testPipeline",
	}
	// then we create out pipeline.
	// Note: the pipeline will also be added to the session object so all pipelines can be destroyed at once
	sentimentPipeline, err := session.NewPipeline(config)
	check(err)

	batch := []string{"This movie is disgustingly good !", "The director tried too much"}
	batchResult, err := sentimentPipeline.RunPipeline(t.Context(), batch)
	check(err)

	s, err := json.Marshal(batchResult)
	check(err)
	fmt.Println(string(s))
	// OUTPUT: {"ClassificationOutputs":[[{"Label":"POSITIVE","Score":0.9998536}],[{"Label":"NEGATIVE","Score":0.99752176}]]}
}
