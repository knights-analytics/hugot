package hugo_test

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path"
	"strings"
	"testing"

	"github.com/knights-analytics/hugo"
	"github.com/knights-analytics/hugo/pipelines"
	"github.com/knights-analytics/hugo/utils/checks"
	"github.com/stretchr/testify/assert"
)

//go:embed testData/tokenExpected.json
var tokenExpectedByte []byte

//go:embed testData/vectors.json
var resultsByte []byte

// Text classification

func TestTextClassificationPipeline(t *testing.T) {
	session := hugo.NewSession()
	defer session.Destroy()
	modelFolder := os.Getenv("TEST_MODELS_FOLDER")
	modelPath := path.Join(modelFolder, "distilbert-base-uncased-finetuned-sst-2-english")
	sentimentPipeline := session.NewTextClassificationPipeline(modelPath, "testPipeline")

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
			result := tt.pipeline.Run(tt.strings)
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
	session := hugo.NewSession()
	defer session.Destroy()

	modelFolder := os.Getenv("TEST_MODELS_FOLDER")
	modelPath := path.Join(modelFolder, "distilbert-NER")
	pipelineSimple := session.NewTokenClassificationPipeline(modelPath, "testPipeline", pipelines.WithSimpleAggregation())
	pipelineNone := session.NewTokenClassificationPipeline(modelPath, "testPipeline", pipelines.WithoutAggregation())

	var expectedResults map[int]pipelines.TokenClassificationOutput
	checks.Check(json.Unmarshal(tokenExpectedByte, &expectedResults))

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
			res := tt.pipeline.Run(tt.strings)
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
	session := hugo.NewSession()
	defer session.Destroy()

	modelFolder := os.Getenv("TEST_MODELS_FOLDER")
	modelPath := path.Join(modelFolder, "all-MiniLM-L6-v2")
	pipeline := session.NewFeatureExtractionPipeline(modelPath, "testPipeline")

	var expectedResults map[string][][]float32
	checks.Check(json.Unmarshal(resultsByte, &expectedResults))
	var testResults [][]float32
	var result [][]float32
	// test 'robert smith'

	testResults = expectedResults["test1output"]
	for i := 1; i <= 10; i++ {
		result = pipeline.Run([]string{"robert smith"})
		e := floatsEqual(result[0], testResults[0])
		if e != nil {
			t.Logf("Test 1: The neural network didn't produce the correct result on loop %d: %s\n", i, e)
			t.FailNow()
		}
	}

	// test ['robert smith junior', 'francis ford coppola']
	testResults = expectedResults["test2output"]
	for i := 1; i <= 10; i++ {
		result = pipeline.Run([]string{"robert smith junior", "francis ford coppola"})
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
		firstEmbedding := pipeline.Run(sentencePair[0])[0]
		secondEmbedding := pipeline.Run(sentencePair[1])[0]
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
