package hugot_test

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path"
	"runtime/debug"
	"strings"
	"testing"

	"github.com/phuslu/log"

	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/pipelines"
	"github.com/stretchr/testify/assert"
)

//go:embed testData/tokenExpected.json
var tokenExpectedByte []byte

//go:embed testData/vectors.json
var resultsByte []byte

// Text classification

func TestTextClassificationPipeline(t *testing.T) {
	session, err := hugot.NewSession()
	Check(err)
	defer session.Destroy()
	modelFolder := os.Getenv("TEST_MODELS_FOLDER")
	modelPath := path.Join(modelFolder, "distilbert-base-uncased-finetuned-sst-2-english")
	sentimentPipeline, err := session.NewTextClassificationPipeline(modelPath, "testPipeline")
	Check(err)

	tests := []struct {
		pipeline *pipelines.TextClassificationPipeline
		name     string
		strings  []string
		expected pipelines.TextClassificationOutput
	}{
		{
			pipeline: sentimentPipeline,
			name:     "Basic tests",
			strings:  []string{"This movie is disgustingly good !", "The director tried too much"},
			expected: [][]pipelines.ClassificationOutput{
				{
					{
						Label: "POSITIVE",
						Score: 0.9998536109924316,
					},
				},
				{
					{
						Label: "NEGATIVE",
						Score: 0.9975218176841736,
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := tt.pipeline.Run(tt.strings)
			Check(err)
			for i, expected := range tt.expected {
				checkClassificationOutput(t, expected, result[i])
			}
		})
	}

	// check get stats
	session.GetStats()
}

// Token classification

func TestTokenClassificationPipeline(t *testing.T) {
	session, err := hugot.NewSession()
	Check(err)
	defer session.Destroy()

	modelFolder := os.Getenv("TEST_MODELS_FOLDER")
	modelPath := path.Join(modelFolder, "distilbert-NER")
	pipelineSimple, err := session.NewTokenClassificationPipeline(modelPath, "testPipeline", pipelines.WithSimpleAggregation())
	Check(err)
	pipelineNone, err := session.NewTokenClassificationPipeline(modelPath, "testPipeline", pipelines.WithoutAggregation())
	Check(err)

	var expectedResults map[int]pipelines.TokenClassificationOutput
	Check(json.Unmarshal(tokenExpectedByte, &expectedResults))

	tests := []struct {
		pipeline *pipelines.TokenClassificationPipeline
		name     string
		strings  []string
		expected pipelines.TokenClassificationOutput
	}{
		{
			pipeline: pipelineSimple,
			name:     "Simple aggregation",
			strings:  []string{"My name is Wolfgang and I live in Berlin."},
			expected: expectedResults[0],
		},
		{
			pipeline: pipelineNone,
			name:     "No aggregation",
			strings:  []string{"My name is Wolfgang and I live in Berlin."},
			expected: expectedResults[1],
		},
		{
			pipeline: pipelineSimple,
			name:     "Parsing of batch with different token length",
			strings:  []string{"Microsoft incorporated.", "Yesterday I went to Berlin and met with Jack Brown."},
			expected: expectedResults[2],
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			res, err := tt.pipeline.Run(tt.strings)
			Check(err)
			pipelines.PrintTokenEntities(res)
			for i, predictedEntities := range res {
				assert.Equal(t, len(tt.expected[i]), len(predictedEntities))
				for j, entity := range predictedEntities {
					expectedEntity := tt.expected[i][j]
					assert.Equal(t, expectedEntity.Entity, entity.Entity)
					assert.Equal(t, expectedEntity.Word, entity.Word)
				}
			}
		})
	}
}

// feature extraction

func TestFeatureExtractionPipeline(t *testing.T) {
	session, err := hugot.NewSession()
	Check(err)
	defer session.Destroy()

	modelFolder := os.Getenv("TEST_MODELS_FOLDER")
	modelPath := path.Join(modelFolder, "all-MiniLM-L6-v2")
	pipeline, err := session.NewFeatureExtractionPipeline(modelPath, "testPipeline")
	Check(err)

	var expectedResults map[string][][]float32
	Check(json.Unmarshal(resultsByte, &expectedResults))
	var testResults [][]float32
	var result [][]float32
	// test 'robert smith'

	testResults = expectedResults["test1output"]
	for i := 1; i <= 10; i++ {
		result, err = pipeline.Run([]string{"robert smith"})
		Check(err)
		e := floatsEqual(result[0], testResults[0])
		if e != nil {
			t.Logf("Test 1: The neural network didn't produce the correct result on loop %d: %s\n", i, e)
			t.FailNow()
		}
	}

	// test ['robert smith junior', 'francis ford coppola']
	testResults = expectedResults["test2output"]
	for i := 1; i <= 10; i++ {
		result, err = pipeline.Run([]string{"robert smith junior", "francis ford coppola"})
		Check(err)
		for j, res := range result {
			e := floatsEqual(res, testResults[j])
			if e != nil {
				t.Logf("Test 2: The neural network didn't produce the correct result on loop %d: %s\n", i, e)
				t.FailNow()
			}
		}
	}

	// determinism test to make sure embeddings of a string are not influenced by other strings in the batch
	testPairs := map[string][][]string{}
	testPairs["identity"] = [][]string{{"sinopharm", "yo"}, {"sinopharm", "yo"}}
	testPairs["contextOverlap"] = [][]string{{"sinopharm", "yo"}, {"sinopharm", "yo mama yo"}}
	testPairs["contextDisjoint"] = [][]string{{"sinopharm", "yo"}, {"sinopharm", "another test"}}

	for k, sentencePair := range testPairs {
		// these vectors should be the same
		firstRes, err := pipeline.Run(sentencePair[0])
		Check(err)
		firstEmbedding := firstRes[0]
		secondRes, err := pipeline.Run(sentencePair[1])
		Check(err)
		secondEmbedding := secondRes[0]
		e := floatsEqual(firstEmbedding, secondEmbedding)
		if e != nil {
			t.Logf("Equality failed for determinism test %s test with pairs %s and %s", k, strings.Join(sentencePair[0], ","), strings.Join(sentencePair[1], ","))
			t.Log("First vector", firstEmbedding)
			t.Log("second vector", secondEmbedding)
			t.Fail()
		}
	}
}

// utilities

// Returns an error if any element between a and b don't match.
func floatsEqual(a, b []float32) error {
	if len(a) != len(b) {
		return fmt.Errorf("length mismatch: %d vs %d", len(a), len(b))
	}
	for i := range a {
		diff := a[i] - b[i]
		if diff < 0 {
			diff = -diff
		}
		// Arbitrarily chosen precision. Large enough not to be affected by quantization
		if diff >= 0.000001 {
			return fmt.Errorf("data element %d doesn't match: %.12f vs %.12f",
				i, a[i], b[i])
		}
	}
	return nil
}

func checkClassificationOutput(t *testing.T, inputResult []pipelines.ClassificationOutput, inputExpected []pipelines.ClassificationOutput) {
	assert.Equal(t, len(inputResult), len(inputExpected))
	for i, output := range inputResult {
		resultExpected := inputExpected[i]
		assert.Equal(t, output.Label, resultExpected.Label)
		assert.True(t, almostEqual(float64(output.Score), float64(resultExpected.Score)))
	}
}

func almostEqual(a, b float64) bool {
	return math.Abs(a-b) <= 0.0001
}

func Check(err error) {
	if err != nil {
		stack := strings.Join(strings.Split(string(debug.Stack()), "\n")[5:], "\n")
		log.Fatal().Stack().Err(err).Msg(stack)
	}
}
