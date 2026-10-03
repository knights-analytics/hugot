package testutil

import (
	"context"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"math"
	"os"
	"regexp"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/pipelines"
	"github.com/knights-analytics/hugot/testcases/embedded"
	"github.com/knights-analytics/hugot/util/imageutil"
)

const (
	ModelsFolder        = "../../models/"
	TestCasesFolder     = "../../testcases/"
	MultimodalModelPath = ModelsFolder + "microsoft_Phi-3.5-vision-instruct-onnx/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4"
)

// FEATURE EXTRACTION

func RawFeatureExtractionPipeline(t *testing.T, session *hugot.Session) {
	t.Helper()
	config := hugot.FeatureExtractionConfig{
		ModelPath: ModelsFolder + "KnightsAnalytics_all-MiniLM-L6-v2",
		Name:      "rawFeatureExtraction", OnnxFilename: "model.onnx",
		Options: []hugot.FeatureExtractionOption{pipelines.WithNormalization()},
	}
	pipeline, err := session.NewPipeline(config)
	CheckT(t, err)
	reference := loadFeatureReference(t)
	inputs := reference.Text.Inputs
	raw, err := pipeline.RunRaw(t.Context(), inputs)
	CheckT(t, err)
	checkHiddenStateReference(t, raw, reference.Text.hiddenStateReference, reference.Tolerance)
	if raw == nil || len(raw.Dimensions) != 3 || raw.Dimensions[0] != 2 || raw.Dimensions[2] != 384 || len(raw.HiddenStates) != 2 {
		t.Fatal("raw text features must retain batch, token and hidden dimensions")
	}
	pooled, err := pipeline.RunPipeline(t.Context(), inputs)
	CheckT(t, err)
	if pooled == nil || len(pooled.Embeddings) != 2 {
		t.Fatal("pooled text feature batch mismatch")
	}
	batch := backends.NewBatch(len(inputs))
	defer func() { CheckT(t, batch.Destroy()) }()
	backends.TokenizeInputs(batch, pipeline.Model.Tokenizer, inputs)
	if len(batch.Input) != 2 {
		t.Fatal("raw text tokenizer batch mismatch")
	}
	for i, hidden := range raw.HiddenStates {
		if int64(len(hidden)) != raw.Dimensions[1] || len(hidden) < len(batch.Input[i].TokenIDs) {
			t.Fatal("raw text features lost tokens or padding dimensions")
		}
		mean := make([]float32, 384)
		for j, token := range hidden {
			if len(token) != 384 {
				t.Fatal("raw text hidden dimension mismatch")
			}
			for _, value := range token {
				if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
					t.Fatal("raw text features are nonfinite")
				}
			}
			if j < len(batch.Input[i].AttentionMask) && batch.Input[i].AttentionMask[j] != 0 {
				for k, value := range token {
					mean[k] += value
				}
			}
		}
		var norm float64
		for _, value := range mean {
			norm += float64(value) * float64(value)
		}
		if norm == 0 {
			t.Fatal("raw text features have zero masked mean")
		}
		for j := range mean {
			mean[j] /= float32(math.Sqrt(norm))
		}
		assert.InDeltaSlice(t, mean, pooled.Embeddings[i], 1e-5, "raw features reconstruct unchanged normalized embeddings")
	}
	_, err = pipeline.RunRaw(t.Context(), nil)
	assert.Error(t, err)
}

func FeatureExtractionPipeline(t *testing.T, session *hugot.Session) {
	t.Helper()

	modelPath := ModelsFolder + "KnightsAnalytics_all-MiniLM-L6-v2"

	config := hugot.FeatureExtractionConfig{
		ModelPath:    modelPath,
		Name:         "testPipeline",
		OnnxFilename: "model.onnx",
	}
	pipeline, err := session.NewPipeline(config)
	CheckT(t, err)

	var expectedResults map[string][][]float32
	err = json.Unmarshal(embedded.ResultsByte, &expectedResults)
	CheckT(t, err)
	var testResults [][]float32

	// test 'robert smith'
	testResults = expectedResults["test1output"]
	batchResult, err := pipeline.RunPipeline(t.Context(), []string{"robert smith"})
	if err != nil {
		t.Fatal(err)
	}
	for i := range batchResult.Embeddings {
		e := floatsEqual(batchResult.Embeddings[i], testResults[i])
		if e != nil {
			t.Logf("Test 1: The neural network didn't produce the correct result on loop %d: %s\n", i, e)
			t.FailNow()
		}
	}

	// test ['robert smith junior', 'francis ford coppola']
	testResults = expectedResults["test2output"]
	batchResult, err = pipeline.RunPipeline(t.Context(), []string{"robert smith junior", "francis ford coppola"})
	if err != nil {
		t.FailNow()
	}
	for i := range batchResult.Embeddings {
		e := floatsEqual(batchResult.Embeddings[i], testResults[i])
		if e != nil {
			t.Logf("Test 2: The neural network didn't produce the correct result on loop %d: %s\n", i, e)
			t.FailNow()
		}
	}

	// determinism test to make sure embeddings of a string are not influenced by other strings in the batch
	testPairs := map[string][][]string{}
	testPairs["identity"] = [][]string{{"sinopharm", "yo"}, {"sinopharm", "yo"}}
	testPairs["contextOverlap"] = [][]string{{"sinopharm", "yo"}, {"sinopharm", "yo mama yo"}}
	testPairs["contextDisjoint"] = [][]string{{"sinopharm", "yo"}, {"sinopharm", "another test"}}

	for k, sentencePair := range testPairs {
		// these vectors should be the same
		firstBatchResult, err2 := pipeline.RunPipeline(t.Context(), sentencePair[0])
		CheckT(t, err2)
		firstEmbedding := firstBatchResult.Embeddings[0]

		secondBatchResult, err3 := pipeline.RunPipeline(t.Context(), sentencePair[1])
		CheckT(t, err3)
		secondEmbedding := secondBatchResult.Embeddings[0]
		e := floatsEqual(firstEmbedding, secondEmbedding)
		if e != nil {
			t.Logf("Equality failed for determinism test %s test with pairs %s and %s", k, strings.Join(sentencePair[0], ","), strings.Join(sentencePair[1], ","))
			t.Log("First vector", firstEmbedding)
			t.Log("second vector", secondEmbedding)
			t.Fail()
		}
	}

	zero := uint64(0)
	assert.Greater(t, pipeline.ONNXTimings.NumCalls, zero, "PipelineTimings.NumCalls should be greater than 0")
	assert.Greater(t, pipeline.ONNXTimings.TotalNS, zero, "PipelineTimings.TotalNS should be greater than 0")
	assert.Greater(t, pipeline.TokenizerTimings.NumCalls, zero, "TokenizerTimings.NumCalls should be greater than 0")

	// test normalization
	testResults = expectedResults["normalizedOutput"]
	config = hugot.FeatureExtractionConfig{
		ModelPath:    modelPath,
		Name:         "testPipelineNormalise",
		OnnxFilename: "model.onnx",
		Options: []hugot.FeatureExtractionOption{
			pipelines.WithNormalization(),
		},
	}
	pipeline, err = session.NewPipeline(config)
	CheckT(t, err)

	normalizationStrings := []string{"Onnxruntime is a great inference backend"}
	normalizedEmbedding, err := pipeline.RunPipeline(t.Context(), normalizationStrings)
	CheckT(t, err)
	for i, embedding := range normalizedEmbedding.Embeddings {
		e := floatsEqual(embedding, testResults[i])
		if e != nil {
			t.Fatalf("Normalization test failed: %s", normalizationStrings)
		}
	}

	// test getting output by name
	configSentence := hugot.FeatureExtractionConfig{
		ModelPath:    modelPath,
		Name:         "testPipelineSentence",
		OnnxFilename: "model.onnx",
		Options:      []hugot.FeatureExtractionOption{pipelines.WithOutputName("last_hidden_state")},
	}
	pipelineSentence, err := session.NewPipeline(configSentence)
	CheckT(t, err)

	_, err = pipelineSentence.RunPipeline(t.Context(), []string{"Onnxruntime is a great inference backend"})
	if err != nil {
		t.FailNow()
	}
	configSentence = hugot.FeatureExtractionConfig{
		ModelPath:    modelPath,
		Name:         "testPipelineToken",
		OnnxFilename: "model.onnx",
	}
	pipelineToken, err := session.NewPipeline(configSentence)
	CheckT(t, err)
	_, err = pipelineToken.RunPipeline(t.Context(), []string{"Onnxruntime is a great inference backend"})
	if err != nil {
		t.FailNow()
	}
}

func FeatureExtractionPipelineValidation(t *testing.T, session *hugot.Session) {
	t.Helper()

	modelPath := ModelsFolder + "KnightsAnalytics_all-MiniLM-L6-v2"
	config := hugot.FeatureExtractionConfig{
		ModelPath:    modelPath,
		OnnxFilename: "model.onnx",
		Name:         "testPipeline",
	}
	pipeline, err := session.NewPipeline(config)
	CheckT(t, err)

	pipeline.Model.InputsMeta[0].Dimensions = backends.NewShape(-1, -1, -1)

	err = pipeline.Validate()
	assert.Error(t, err)

	pipeline.Model.InputsMeta[0].Dimensions = backends.NewShape(1, 1, 1, 1, 1)
	err = pipeline.Validate()
	assert.Error(t, err)
}

// Text classification

func TextClassificationPipeline(t *testing.T, session *hugot.Session) {
	t.Helper()

	modelPath := ModelsFolder + "Xenova_distilbert-base-uncased-finetuned-sst-2-english"

	config := hugot.TextClassificationConfig{
		ModelPath: modelPath,
		Name:      "testPipelineSimple",
		Options: []hugot.TextClassificationOption{
			pipelines.WithSoftmax(),
		},
	}
	sentimentPipeline, err := session.NewPipeline(config)
	CheckT(t, err)

	test := struct {
		pipeline *pipelines.TextClassificationPipeline
		name     string
		strings  []string
		expected pipelines.TextClassificationOutput
	}{
		pipeline: sentimentPipeline,
		name:     "Basic tests",
		strings:  []string{"This movie is disgustingly good!", "The director tried too much"},
		expected: pipelines.TextClassificationOutput{
			ClassificationOutputs: [][]pipelines.ClassificationOutput{
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

	t.Run(test.name, func(t *testing.T) {
		batchResult, err := test.pipeline.RunPipeline(t.Context(), test.strings)
		CheckT(t, err)
		for i, expected := range test.expected.ClassificationOutputs {
			checkClassificationOutput(t, expected, batchResult.ClassificationOutputs[i])
		}
	})

	// check PrintStatistics
	session.PrintStatistics()
}

func TextClassificationPipelineMulti(t *testing.T, session *hugot.Session) {
	t.Helper()

	modelPathMulti := ModelsFolder + "SamLowe_roberta-base-go_emotions-onnx"

	configMulti := hugot.TextClassificationConfig{
		ModelPath:    modelPathMulti,
		Name:         "testPipelineSimpleMulti",
		OnnxFilename: "model.onnx",
		Options: []hugot.TextClassificationOption{
			pipelines.WithMultiLabel(),
			pipelines.WithSigmoid(),
			pipelines.WithFixedPadding(128),
		},
	}
	sentimentPipelineMulti, err := session.NewPipeline(configMulti)
	CheckT(t, err)

	test := struct {
		pipeline *pipelines.TextClassificationPipeline
		name     string
		strings  []string
		expected pipelines.TextClassificationOutput
	}{
		pipeline: sentimentPipelineMulti,
		name:     "Multiclass pipeline test",
		strings:  []string{"ONNX is seriously fast for small batches. Impressive"},
		expected: pipelines.TextClassificationOutput{
			ClassificationOutputs: [][]pipelines.ClassificationOutput{
				{
					{
						Label: "admiration",
						Score: 0.9217681,
					},
					{
						Label: "amusement",
						Score: 0.001201711,
					},
					{
						Label: "anger",
						Score: 0.001109502,
					},
					{
						Label: "annoyance",
						Score: 0.0034009134,
					},
					{
						Label: "approval",
						Score: 0.05643816,
					},
					{
						Label: "caring",
						Score: 0.0011591336,
					},
					{
						Label: "confusion",
						Score: 0.0018672282,
					},
					{
						Label: "curiosity",
						Score: 0.0026787464,
					},
					{
						Label: "desire",
						Score: 0.00085846696,
					},
					{
						Label: "disappointment",
						Score: 0.0027759627,
					},
					{
						Label: "disapproval",
						Score: 0.004615115,
					},
					{
						Label: "disgust",
						Score: 0.00075303164,
					},
					{
						Label: "embarrassment",
						Score: 0.0003314704,
					},
					{
						Label: "excitement",
						Score: 0.005340109,
					},
					{
						Label: "fear",
						Score: 0.00042834174,
					},
					{
						Label: "gratitude",
						Score: 0.013405683,
					},
					{
						Label: "grief",
						Score: 0.00029952865,
					},
					{
						Label: "joy",
						Score: 0.0026875956,
					},
					{
						Label: "love",
						Score: 0.00092915917,
					},
					{
						Label: "nervousness",
						Score: 0.00012843,
					},
					{
						Label: "optimism",
						Score: 0.006792505,
					},
					{
						Label: "pride",
						Score: 0.0033409835,
					},
					{
						Label: "realization",
						Score: 0.007224476,
					},
					{
						Label: "relief",
						Score: 0.00071489986,
					},
					{
						Label: "remorse",
						Score: 0.00026071363,
					},
					{
						Label: "sadness",
						Score: 0.0009562365,
					},
					{
						Label: "surprise",
						Score: 0.0037120024,
					},
					{
						Label: "neutral",
						Score: 0.04079749,
					},
				},
			},
		},
	}

	t.Run(test.name, func(t *testing.T) {
		batchResult, err := test.pipeline.RunPipeline(t.Context(), test.strings)
		CheckT(t, err)
		for i, expected := range test.expected.ClassificationOutputs {
			checkClassificationOutput(t, expected, batchResult.ClassificationOutputs[i])
		}
	})

	// check GetStatistics
	statistics := session.GetStatistics()
	for m, v := range statistics {
		fmt.Printf("pipeline statistics for: %s\n", m)
		v.Print()
	}
}

func TextClassificationPipelineValidation(t *testing.T, session *hugot.Session) {
	t.Helper()

	modelPath := ModelsFolder + "Xenova_distilbert-base-uncased-finetuned-sst-2-english"

	config := hugot.TextClassificationConfig{
		ModelPath: modelPath,
		Name:      "testPipelineSimple",
		Options: []hugot.TextClassificationOption{
			pipelines.WithSingleLabel(),
		},
	}
	sentimentPipeline, err := session.NewPipeline(config)
	CheckT(t, err)

	t.Run("id-label-map", func(t *testing.T) {
		labelMapInitial := sentimentPipeline.Model.IDLabelMap
		defer func() {
			sentimentPipeline.Model.IDLabelMap = labelMapInitial
		}()
		sentimentPipeline.Model.IDLabelMap = map[int]string{}
		err = sentimentPipeline.Validate()
		assert.Error(t, err)
	})

	t.Run("output-shape", func(t *testing.T) {
		dimensionInitial := sentimentPipeline.Model.OutputsMeta[0].Dimensions
		defer func() {
			sentimentPipeline.Model.OutputsMeta[0].Dimensions = dimensionInitial
		}()
		sentimentPipeline.Model.OutputsMeta[0].Dimensions = backends.NewShape(-1, -1, -1)
		err = sentimentPipeline.Validate()
		assert.Error(t, err)
	})
}

// Zero shot

func ZeroShotClassificationPipeline(t *testing.T, session *hugot.Session) {
	t.Helper()

	modelPath := ModelsFolder + "KnightsAnalytics_deberta-v3-base-zeroshot-v1"

	config := hugot.ZeroShotClassificationConfig{
		ModelPath: modelPath,
		Name:      "testPipeline",
		Options: []backends.PipelineOption[*pipelines.ZeroShotClassificationPipeline]{
			pipelines.WithHypothesisTemplate("This example is {}."),
			pipelines.WithLabels([]string{"fun", "dangerous"}),
			pipelines.WithMultilabel(false), // Gets overridden per test, but included for coverage
		},
	}

	classificationPipeline, err := session.NewPipeline(config)
	CheckT(t, err)

	tests := []struct {
		pipeline   *pipelines.ZeroShotClassificationPipeline
		name       string
		sequences  []string
		labels     []string
		multilabel bool
		expected   pipelines.ZeroShotOutput
	}{
		{
			pipeline:   classificationPipeline,
			name:       "single sequence, single label, no multilabel",
			sequences:  []string{"I am going to the park"},
			labels:     []string{"fun"},
			multilabel: false,
			expected: pipelines.ZeroShotOutput{
				ClassificationOutputs: []pipelines.ZeroShotClassificationOutput{
					{
						Sequence: "I am going to the park",
						SortedValues: []struct {
							Key   string
							Value float64
						}{
							{
								Key:   "fun",
								Value: 0.0009069009101949632,
							},
						},
					},
				},
			},
		},
		{
			pipeline:   classificationPipeline,
			name:       "multiple sequences, multiple labels, no multilabel",
			sequences:  []string{"I am going to the park", "I will watch Interstellar tonight"},
			labels:     []string{"fun", "movie"},
			multilabel: false,
			expected: pipelines.ZeroShotOutput{
				ClassificationOutputs: []pipelines.ZeroShotClassificationOutput{
					{
						Sequence: "I am going to the park",
						SortedValues: []struct {
							Key   string
							Value float64
						}{
							{
								Key:   "fun",
								Value: 0.7746766209602356,
							},
							{
								Key:   "movie",
								Value: 0.2253233790397644,
							},
						},
					},
					{
						Sequence: "I will watch Interstellar tonight",
						SortedValues: []struct {
							Key   string
							Value float64
						}{
							{
								Key:   "movie",
								Value: 0.9984978437423706,
							},
							{
								Key:   "fun",
								Value: 0.001502170693129301,
							},
						},
					},
				},
			},
		},
		{
			pipeline:   classificationPipeline,
			name:       "multiple sequences, multiple labels, multilabel",
			sequences:  []string{"I am going to the park", "I will watch Interstellar tonight"},
			labels:     []string{"fun", "movie"},
			multilabel: true,
			expected: pipelines.ZeroShotOutput{
				ClassificationOutputs: []pipelines.ZeroShotClassificationOutput{
					{
						Sequence: "I am going to the park",
						SortedValues: []struct {
							Key   string
							Value float64
						}{
							{
								Key:   "fun",
								Value: 0.0009069009101949632,
							},
							{
								Key:   "movie",
								Value: 0.00009480675362283364,
							},
						},
					},
					{
						Sequence: "I will watch Interstellar tonight",
						SortedValues: []struct {
							Key   string
							Value float64
						}{
							{
								Key:   "movie",
								Value: 0.9985591769218445,
							},
							{
								Key:   "fun",
								Value: 0.0006653196760453284,
							},
						},
					},
				},
			},
		},
		{
			pipeline:   classificationPipeline,
			name:       "multiple sequences, single label, multilabel",
			sequences:  []string{"I am going to the park", "I will watch Interstellar tonight"},
			labels:     []string{"fun"},
			multilabel: true,
			expected: pipelines.ZeroShotOutput{
				ClassificationOutputs: []pipelines.ZeroShotClassificationOutput{
					{
						Sequence: "I am going to the park",
						SortedValues: []struct {
							Key   string
							Value float64
						}{
							{
								Key:   "fun",
								Value: 0.0009069009101949632,
							},
						},
					},
					{
						Sequence: "I will watch Interstellar tonight",
						SortedValues: []struct {
							Key   string
							Value float64
						}{
							{
								Key:   "fun",
								Value: 0.0006653196760453284,
							},
						},
					},
				},
			},
		},
		{
			pipeline:   classificationPipeline,
			name:       "single sequence, multiple labels, multilabel=false",
			sequences:  []string{"Please don't bother me, I'm in a rush"},
			labels:     []string{"busy", "relaxed", "stressed"},
			multilabel: false,
			expected: pipelines.ZeroShotOutput{
				ClassificationOutputs: []pipelines.ZeroShotClassificationOutput{
					{
						Sequence: "Please don't bother me, I'm in a rush",
						SortedValues: []struct {
							Key   string
							Value float64
						}{
							{
								Key:   "stressed",
								Value: 0.8865461349487305,
							},
							{
								Key:   "busy",
								Value: 0.10629364103078842,
							},
							{
								Key:   "relaxed",
								Value: 0.007160270120948553,
							},
						},
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			classificationPipeline.Labels = tt.labels
			classificationPipeline.Multilabel = tt.multilabel
			batchResult, err := tt.pipeline.RunPipeline(t.Context(), tt.sequences)
			CheckT(t, err)
			assert.Equal(t, len(batchResult.GetOutput()), len(tt.expected.ClassificationOutputs))

			for ind, expected := range tt.expected.ClassificationOutputs {
				expectedResult := expected.SortedValues
				testResult := batchResult.ClassificationOutputs[ind].SortedValues
				assert.Equal(t, len(expectedResult), len(testResult))
				assert.Equal(t, tt.expected.ClassificationOutputs[ind].Sequence, batchResult.ClassificationOutputs[ind].Sequence)
				for i := range testResult {
					assert.True(t, almostEqual(testResult[i].Value, expectedResult[i].Value), fmt.Sprintf("Expected %f, got %f", expectedResult[i].Value, testResult[i].Value))
				}
			}
		})
	}
}

func ZeroShotClassificationPipelineValidation(t *testing.T, session *hugot.Session) {
	t.Helper()

	modelPath := ModelsFolder + "KnightsAnalytics_deberta-v3-base-zeroshot-v1"

	config := hugot.ZeroShotClassificationConfig{
		ModelPath: modelPath,
		Name:      "testZeroShotPipelineValidation",
		Options: []hugot.ZeroShotClassificationOption{
			pipelines.WithLabels([]string{"positive", "negative"}),
		},
	}
	classificationPipeline, err := session.NewPipeline(config)
	CheckT(t, err)

	t.Run("id-label-map", func(t *testing.T) {
		labelMapInitial := classificationPipeline.Model.IDLabelMap
		defer func() {
			classificationPipeline.Model.IDLabelMap = labelMapInitial
		}()
		classificationPipeline.Model.IDLabelMap = map[int]string{}
		assert.Error(t, classificationPipeline.Validate())
	})

	t.Run("output-shape", func(t *testing.T) {
		dimensionInitial := classificationPipeline.Model.OutputsMeta[0].Dimensions
		defer func() {
			classificationPipeline.Model.OutputsMeta[0].Dimensions = dimensionInitial
		}()
		classificationPipeline.Model.OutputsMeta[0].Dimensions = backends.NewShape(-1, -1, -1)
		assert.Error(t, classificationPipeline.Validate())
	})
}

// Token classification

func TokenClassificationPipeline(t *testing.T, session *hugot.Session) {
	t.Helper()

	modelPath := ModelsFolder + "KnightsAnalytics_distilbert-NER"
	configSimple := hugot.TokenClassificationConfig{
		ModelPath: modelPath,
		Name:      "testPipelineSimple",
		Options: []hugot.TokenClassificationOption{
			pipelines.WithSimpleAggregation(),
			pipelines.WithIgnoreLabels([]string{"O"}),
		},
	}
	pipelineSimple, err2 := session.NewPipeline(configSimple)
	CheckT(t, err2)

	configNone := hugot.TokenClassificationConfig{
		ModelPath: modelPath,
		Name:      "testPipelineNone",
		Options: []hugot.TokenClassificationOption{
			pipelines.WithoutAggregation(),
		},
	}
	pipelineNone, err3 := session.NewPipeline(configNone)
	CheckT(t, err3)

	// Split-words enabled pipeline
	configSplit := hugot.TokenClassificationConfig{
		ModelPath: modelPath,
		Name:      "testPipelineSplitWords",
		Options: []hugot.TokenClassificationOption{
			pipelines.WithSimpleAggregation(),
			pipelines.WithIgnoreLabels([]string{"O"}),
			pipelines.WithSplitWords(),
		},
	}
	pipelineSplit, errSplit := session.NewPipeline(configSplit)
	CheckT(t, errSplit)

	var expectedResults map[int]pipelines.TokenClassificationOutput
	err4 := json.Unmarshal(embedded.TokenExpectedByte, &expectedResults)
	CheckT(t, err4)

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
			batchResult, err := tt.pipeline.RunPipeline(t.Context(), tt.strings)
			CheckT(t, err)
			printTokenEntities(batchResult)
			for i, predictedEntities := range batchResult.Entities {
				assert.Equal(t, len(tt.expected.Entities[i]), len(predictedEntities))
				for j, entity := range predictedEntities {
					expectedEntity := tt.expected.Entities[i][j]
					assert.Equal(t, expectedEntity.Entity, entity.Entity)
					assert.Equal(t, expectedEntity.Word, entity.Word)
				}
			}
		})
	}

	// Expect same entities as the simple aggregation for the equivalent sentence for split words
	t.Run("Split words aggregation", func(t *testing.T) {
		words := [][]string{{"My", "name", "is", "Wolfgang", "and", "I", "live", "in", "Berlin", "."}}
		batchResult, err := pipelineSplit.RunWords(t.Context(), words)
		CheckT(t, err)
		printTokenEntities(batchResult)
		expected := expectedResults[0]
		for i, predictedEntities := range batchResult.Entities {
			assert.Equal(t, len(expected.Entities[i]), len(predictedEntities))
			for j, entity := range predictedEntities {
				expectedEntity := expected.Entities[i][j]
				assert.Equal(t, expectedEntity.Entity, entity.Entity)
				assert.Equal(t, expectedEntity.Word, entity.Word)
			}
		}
	})

	t.Run("Split words yields different offsets vs double-space input", func(t *testing.T) {
		// Non-split input with double space changes raw offsets.
		nonSplit := []string{"New  York is great."}
		splitWords := [][]string{{"New", "York", "is", "great", "."}}

		resNonSplit, errNon := pipelineSimple.RunPipeline(t.Context(), nonSplit)
		CheckT(t, errNon)
		resSplit, errSplitRun := pipelineSplit.RunWords(t.Context(), splitWords)
		CheckT(t, errSplitRun)

		// Compare first sequence: entity words may match, but offsets should differ due to space normalization.
		if len(resNonSplit.Entities) > 0 && len(resSplit.Entities) > 0 && len(resNonSplit.Entities[0]) > 0 && len(resSplit.Entities[0]) > 0 {
			eNon := resNonSplit.Entities[0][0]
			eSplit := resSplit.Entities[0][0]
			assert.NotEqual(t, fmt.Sprintf("%d-%d", eNon.Start, eNon.End), fmt.Sprintf("%d-%d", eSplit.Start, eSplit.End))
		}
	})

	// Pre-tokenization splitting on 'X': expect different entity results from the non-split case
	t.Run("Split on X detects entity", func(t *testing.T) {
		nonSplit := []string{"XBerlinXXisXbeautiful."}
		split := [][]string{{"Berlin is", "beautiful", "."}}

		resNonSplit, errNS := pipelineSimple.RunPipeline(t.Context(), nonSplit)
		CheckT(t, errNS)
		resSplit, errSW := pipelineSplit.RunWords(t.Context(), split)
		CheckT(t, errSW)

		gotA := resNonSplit.Entities[0]
		gotB := resSplit.Entities[0]
		// Expect split-words to detect 'Berlin' as an entity, while non-split should not because of the confusing X characters.
		hasBerlin := func(es []pipelines.Entity) bool {
			for _, e := range es {
				if strings.EqualFold(e.Word, "Berlin") {
					return true
				}
			}
			return false
		}
		assert.True(t, !hasBerlin(gotA) || len(gotA) < len(gotB), "expected split-words to surface 'Berlin' or increase entity count")
	})

	t.Run("Word-aggregating strategies merge subwords", func(t *testing.T) {
		for _, tc := range []struct {
			name string
			opt  hugot.TokenClassificationOption
		}{
			{"FIRST", pipelines.WithFirstAggregation()},
			{"MAX", pipelines.WithMaxAggregation()},
			{"AVERAGE", pipelines.WithAverageAggregation()},
		} {
			p, err := session.NewPipeline(hugot.TokenClassificationConfig{
				ModelPath: modelPath,
				Name:      "repro-" + tc.name,
				Options:   []hugot.TokenClassificationOption{tc.opt, pipelines.WithIgnoreLabels([]string{"O"})},
			})
			CheckT(t, err)
			out, err := p.RunPipeline(t.Context(), []string{"Luan Lorenzo"})
			CheckT(t, err)
			assert.NotEmpty(t, out.Entities)
			for _, ents := range out.Entities {
				for _, e := range ents {
					assert.NotEqual(t, "Lu", e.Word)
				}
			}
		}
	})
}

func TokenClassificationPipelineValidation(t *testing.T, session *hugot.Session) {
	t.Helper()

	modelPath := ModelsFolder + "KnightsAnalytics_distilbert-NER"
	configSimple := hugot.TokenClassificationConfig{
		ModelPath: modelPath,
		Name:      "testPipelineSimple",
		Options: []hugot.TokenClassificationOption{
			pipelines.WithSimpleAggregation(),
			pipelines.WithIgnoreLabels([]string{"O"}),
		},
	}
	pipelineSimple, err2 := session.NewPipeline(configSimple)
	CheckT(t, err2)

	t.Run("id-label-map", func(t *testing.T) {
		labelMapInitial := pipelineSimple.IDLabelMap
		defer func() {
			pipelineSimple.IDLabelMap = labelMapInitial
		}()
		pipelineSimple.IDLabelMap = map[int]string{}
		err := pipelineSimple.Validate()
		assert.Error(t, err)
	})

	t.Run("output-shape", func(t *testing.T) {
		dimensionInitial := pipelineSimple.Model.OutputsMeta[0].Dimensions
		defer func() {
			pipelineSimple.Model.OutputsMeta[0].Dimensions = dimensionInitial
		}()
		pipelineSimple.Model.OutputsMeta[0].Dimensions = backends.NewShape(-1, -1, -1)
		err := pipelineSimple.Validate()
		assert.Error(t, err)
	})
}

// Cross Encoder

func CrossEncoderPipeline(t *testing.T, session *hugot.Session) {
	t.Helper()
	config := hugot.CrossEncoderConfig{
		ModelPath: ModelsFolder + "jinaai_jina-reranker-v1-tiny-en",
		Name:      "test-cross-encoder",
	}
	pipeline, err := session.NewPipeline(config)
	CheckT(t, err)

	query := "Organic skincare products for sensitive skin"
	documents := []string{
		"Eco-friendly kitchenware for modern homes",
		"Biodegradable cleaning supplies for eco-conscious consumers",
		"Organic cotton baby clothes for sensitive skin",
		"Natural organic skincare range for sensitive skin",
		"Tech gadgets for smart homes: 2024 edition",
		"Sustainable gardening tools and compost solutions",
		"Sensitive skin-friendly facial cleansers and toners",
		"Organic food wraps and storage solutions",
		"All-natural pet food for dogs with allergies",
		"Yoga mats made from recycled materials",
	}

	type Expected struct {
		Document string
		Score    float32
	}

	expectedRoberta := []Expected{
		{Document: "Natural organic skincare range for sensitive skin", Score: 0.95478064},
		{Document: "Organic cotton baby clothes for sensitive skin", Score: 0.8185698},
		{Document: "Sensitive skin-friendly facial cleansers and toners", Score: 0.5848757},
		{Document: "Organic food wraps and storage solutions", Score: 0.2567817},
		{Document: "Biodegradable cleaning supplies for eco-conscious consumers", Score: 0.22029042},
		{Document: "Yoga mats made from recycled materials", Score: 0.20082192},
		{Document: "Sustainable gardening tools and compost solutions", Score: 0.19299757},
		{Document: "All-natural pet food for dogs with allergies", Score: 0.18836288},
		{Document: "Eco-friendly kitchenware for modern homes", Score: 0.18346606},
		{Document: "Tech gadgets for smart homes: 2024 edition", Score: 0.16224432},
	}

	inputs := append([]string{query}, documents...)
	output, err := pipeline.Run(t.Context(), inputs)
	CheckT(t, err)
	results := output.(*pipelines.CrossEncoderOutput).Results

	for i, expected := range expectedRoberta {
		if expected.Document != results[i].Document {
			t.Errorf("Expected document '%s', got '%s'", expected.Document, results[i].Document)
		}
		if math.Abs(float64(expected.Score-results[i].Score)) > 0.01 {
			t.Errorf("Expected score '%f', got '%f'", expected.Score, results[i].Score)
		}
	}
}

func CrossEncoderPipelineValidation(t *testing.T, session *hugot.Session) {
	t.Helper()
	config := hugot.CrossEncoderConfig{
		ModelPath: ModelsFolder + "jinaai_jina-reranker-v1-tiny-en",
		Name:      "test-cross-encoder-validation",
	}
	pipeline, err := session.NewPipeline(config)
	if err != nil {
		t.Fatalf("Failed to create pipeline: %v", err)
	}

	// 1. Test: output dims length != 2
	pipeline.Model.OutputsMeta[0].Dimensions = backends.NewShape(1)
	err = pipeline.Validate()
	if err == nil {
		t.Errorf("Expected error for output dims length != 2, got %v", err)
	}

	// 2. Test: output dims second dim != 1
	pipeline.Model.OutputsMeta[0].Dimensions = backends.NewShape(2, 3)
	err = pipeline.Validate()
	if err == nil || err.Error() == "" {
		t.Errorf("Expected error for output dims second dim != 1, got %v", err)
	}

	// 3. Test: more than one dynamic dim (-1)
	pipeline.Model.OutputsMeta[0].Dimensions = backends.NewShape(-1, -1)
	err = pipeline.Validate()
	if err == nil || err.Error() == "" {
		t.Errorf("Expected error for more than one dynamic dim, got %v", err)
	}
}

// ImageClassificationPipeline test using Hugging Face ResNet and a sample image.
func ImageClassificationPipeline(t *testing.T, session *hugot.Session) {
	t.Helper()

	modelPath := ModelsFolder + "KnightsAnalytics_resnet50"
	imagePath := ModelsFolder + "imageData/cat.jpg"

	config := hugot.ImageClassificationConfig{
		ModelPath:    modelPath,
		Name:         "testImageClassification",
		OnnxFilename: "resnet50-v1-12.onnx",
		Options: []hugot.ImageClassificationOption{
			pipelines.WithTopK(3),
			pipelines.WithPreprocessSteps[*pipelines.ImageClassificationPipeline](
				imageutil.ResizeBilinearStep(256),
				imageutil.CenterCropStep(224, 224),
			),
			pipelines.WithNormalizationSteps[*pipelines.ImageClassificationPipeline](
				imageutil.RescaleStep(),
				imageutil.ImagenetPixelNormalizationStep(),
			),
		},
	}
	pipeline, err := session.NewPipeline(config)
	CheckT(t, err)

	result, err := pipeline.RunPipeline(t.Context(), []string{imagePath, imagePath})
	CheckT(t, err)

	for i, pred := range result.Predictions[0] {
		fmt.Printf("%d: %s (score: %.4f)\n", i+1, pred.Label, pred.Score)
	}
	if result.Predictions[0][0].Label != "tabby, tabby cat" {
		t.Errorf("Expected label 'tabby, tabby cat', got '%s'", result.Predictions[0][0].Label)
	}
}

func ImageClassificationPipelineValidation(t *testing.T, session *hugot.Session) {
	t.Helper()

	modelPath := ModelsFolder + "/KnightsAnalytics_resnet50"
	config := hugot.ImageClassificationConfig{
		ModelPath: modelPath,
		Name:      "testImageClassification",
	}
	pipeline, err := session.NewPipeline(config)
	CheckT(t, err)

	pipeline.Model.InputsMeta[0].Dimensions = backends.NewShape(-1, -1, -1)

	err = pipeline.Validate()
	assert.Error(t, err)
}

// object detection

func ObjectDetectionPipeline(t *testing.T, session *hugot.Session) {
	t.Helper()

	config := backends.PipelineConfig[*pipelines.ObjectDetectionPipeline]{
		ModelPath: ModelsFolder + "Xenova_detr-resnet-50",
		Name:      "testObjectDetection",
		Options: []backends.PipelineOption[*pipelines.ObjectDetectionPipeline]{
			pipelines.WithNCHWFormat[*pipelines.ObjectDetectionPipeline](),
			pipelines.WithDetectionTopK(50),
			pipelines.WithDetectionScoreThreshold(0.3),
			pipelines.WithDetectionIouThreshold(0.5),
		},
	}

	pipeline, err := session.NewPipeline(config)
	CheckT(t, err)

	// Use a simple cat image similar to classification test style
	inputs := []string{ModelsFolder + "imageData/cat.jpg"}
	result, err := pipeline.RunPipeline(t.Context(), inputs)
	CheckT(t, err)

	if len(result.Detections) == 0 || len(result.Detections[0]) == 0 {
		t.Fatalf("no detections returned")
	}
	// Find a detection labeled cat (COCO index 15)
	foundCat := false
	for _, d := range result.Detections[0] {
		if strings.EqualFold(d.Label, "cat") {
			foundCat = true
			// basic box sanity
			if !(d.Box[0] >= 0 && d.Box[1] >= 0 && d.Box[2] > d.Box[0] && d.Box[3] > d.Box[1]) {
				t.Fatalf("invalid box: %v", d.Box)
			}
			// score should be reasonable
			if d.Score < 0.2 {
				t.Fatalf("cat detection score too low: %.3f", d.Score)
			}
			break
		}
	}
	if !foundCat {
		// fall back to checking top detection label for debug
		top := result.Detections[0][0]
		t.Fatalf("expected a cat detection, top=%s score=%.3f", top.Label, top.Score)
	}
}

func ObjectDetectionPipelineValidation(t *testing.T, session *hugot.Session) {
	t.Helper()

	config := backends.PipelineConfig[*pipelines.ObjectDetectionPipeline]{
		ModelPath: ModelsFolder + "Xenova_detr-resnet-50",
		Name:      "testObjectDetectionValidation",
	}
	pipeline, err := session.NewPipeline(config)
	CheckT(t, err)

	t.Run("input-4d-required", func(t *testing.T) {
		// Corrupt the primary image input to have invalid dims
		original := pipeline.Model.InputsMeta[0].Dimensions
		defer func() { pipeline.Model.InputsMeta[0].Dimensions = original }()
		pipeline.Model.InputsMeta[0].Dimensions = backends.NewShape(-1, -1, -1)
		err = pipeline.Validate()
		assert.Error(t, err)
	})

	t.Run("mask-3d-required", func(t *testing.T) {
		// If a mask input exists, make it invalid length to trigger error
		idx := -1
		for i, in := range pipeline.Model.InputsMeta {
			if strings.Contains(strings.ToLower(in.Name), "mask") {
				idx = i
				break
			}
		}
		if idx >= 0 {
			original := pipeline.Model.InputsMeta[idx].Dimensions
			defer func() { pipeline.Model.InputsMeta[idx].Dimensions = original }()
			pipeline.Model.InputsMeta[idx].Dimensions = backends.NewShape(-1, -1) // invalid
			err = pipeline.Validate()
			assert.Error(t, err)
		}
	})

	t.Run("outputs-must-be-detectable", func(t *testing.T) {
		// Rename outputs so inference of boxes/scores fails
		originals := make([]string, len(pipeline.Model.OutputsMeta))
		for i := range pipeline.Model.OutputsMeta {
			originals[i] = pipeline.Model.OutputsMeta[i].Name
			pipeline.Model.OutputsMeta[i].Name = fmt.Sprintf("out_%d", i)
		}
		defer func() {
			for i := range pipeline.Model.OutputsMeta {
				pipeline.Model.OutputsMeta[i].Name = originals[i]
			}
		}()
		pipeline.BoxesOutput = ""
		pipeline.ScoresOutput = ""
		err = pipeline.Validate()
		assert.Error(t, err)
	})
}

// fill-mask

func FillMaskPipeline(t *testing.T, session *hugot.Session) {
	t.Helper()

	config := hugot.FillMaskConfig{
		ModelPath: ModelsFolder + "Xenova_bert-base-uncased",
		Name:      "testFillMask",
		Options: []hugot.FillMaskOption{
			pipelines.WithMaskToken("[MASK]"),
			pipelines.WithFillMaskTopK(3),
		},
	}
	pipeline, err := session.NewPipeline(config)
	CheckT(t, err)
	result, err := pipeline.RunPipeline(t.Context(), []string{"HuggingFace is [MASK]."})
	CheckT(t, err)
	if len(result.Predictions) != 1 || len(result.Predictions[0]) == 0 {
		t.Fatal("fill-mask inference returned no predictions")
	}
	assert.Greater(t, result.Predictions[0][0].Score, float32(0))

	// Explicit top-3 fill-mask predictions for "HuggingFace is [MASK]."
	expectedPredictions := []pipelines.FillMaskResult{
		{Token: "[MASK]", TokenID: 3819, Sequence: "[CLS] huggingface is perfect . [SEP]", Score: 0.029229378},
		{Token: "[MASK]", TokenID: 2204, Sequence: "[CLS] huggingface is good . [SEP]", Score: 0.022476919},
		{Token: "[MASK]", TokenID: 2157, Sequence: "[CLS] huggingface is right . [SEP]", Score: 0.019856254},
	}
	assert.Len(t, result.Predictions[0], len(expectedPredictions), "TopK=3 should return exactly 3 predictions")
	for i, expected := range expectedPredictions {
		actual := result.Predictions[0][i]
		assert.Equal(t, expected.Token, actual.Token, "prediction %d token", i)
		assert.Equal(t, expected.TokenID, actual.TokenID, "prediction %d tokenID", i)
		assert.Equal(t, expected.Sequence, actual.Sequence, "prediction %d sequence", i)
		assert.InDelta(t, float64(expected.Score), float64(actual.Score), 1e-3, "prediction %d score", i)
	}
}

func FillMaskPipelineValidation(t *testing.T, session *hugot.Session) {
	t.Helper()

	config := hugot.FillMaskConfig{
		ModelPath: ModelsFolder + "Xenova_bert-base-uncased",
		Name:      "testFillMaskValidation",
		Options: []hugot.FillMaskOption{
			pipelines.WithMaskToken("[MASK]"),
		},
	}
	pipeline, err := session.NewPipeline(config)
	CheckT(t, err)

	pipeline.TopK = 0
	assert.Error(t, pipeline.Validate(), "fill-mask validation should reject a non-positive top-k")
}

// image feature extraction

func ImageFeatureExtractionPipeline(t *testing.T, session *hugot.Session) {
	t.Helper()

	config := hugot.ImageFeatureExtractionConfig{
		ModelPath:    ModelsFolder + "Xenova_dino-vits16",
		Name:         "testImageFeatureExtraction",
		OnnxFilename: "model.onnx",
		Options: []hugot.ImageFeatureExtractionOption{
			pipelines.WithPreprocessSteps[*pipelines.ImageFeatureExtractionPipeline](
				imageutil.ResizeStep(224),
				imageutil.CenterCropStep(224, 224),
			),
			pipelines.WithNormalizationSteps[*pipelines.ImageFeatureExtractionPipeline](imageutil.RescaleStep(), imageutil.ImagenetPixelNormalizationStep()),
		},
	}
	pipeline, err := session.NewPipeline(config)
	CheckT(t, err)
	imagePath := ModelsFolder + "imageData/cat.jpg"
	result, err := pipeline.RunPipeline(t.Context(), []string{imagePath, imagePath, ModelsFolder + "imageData/invoice.png"})
	CheckT(t, err)
	if result == nil || len(result.Embeddings) != 3 {
		t.Fatal("image feature extraction returned the wrong batch size")
	}
	assert.Equal(t, "last_hidden_state", pipeline.GetMetadata().OutputsInfo[0].Name)
	for _, embedding := range result.Embeddings {
		assert.Len(t, embedding, 384)
		var norm float64
		for _, value := range embedding {
			assert.False(t, math.IsNaN(float64(value)) || math.IsInf(float64(value), 0))
			norm += float64(value) * float64(value)
		}
		assert.Greater(t, norm, 0.0)
	}
	assert.InDeltaSlice(t, result.Embeddings[0], result.Embeddings[1], 1e-5)
	var difference float64
	for i, value := range result.Embeddings[0] {
		difference += math.Abs(float64(value - result.Embeddings[2][i]))
	}
	assert.Greater(t, difference, 0.1, "different images must produce different hidden features")
	raw, err := pipeline.RunRaw(t.Context(), []string{imagePath, imagePath, ModelsFolder + "imageData/invoice.png"})
	CheckT(t, err)
	if raw == nil || len(raw.HiddenStates) != 3 || len(raw.Dimensions) != 3 {
		t.Fatal("raw image features must retain batch, patch and hidden dimensions")
	}
	assert.EqualValues(t, []int64{3, 197, 384}, raw.Dimensions)
	for i, patches := range raw.HiddenStates {
		mean := make([]float32, 384)
		for _, patch := range patches {
			if len(patch) != 384 {
				t.Fatal("raw image hidden dimension mismatch")
			}
			for j, value := range patch {
				mean[j] += value
			}
		}
		for j := range mean {
			mean[j] /= float32(len(patches))
		}
		assert.InDeltaSlice(t, result.Embeddings[i], mean, 1e-5, "legacy image embeddings remain mean pooled")
	}
	reference := loadFeatureReference(t)
	img := image.NewNRGBA(image.Rect(0, 0, reference.Image.Width, reference.Image.Height))
	for y := range reference.Image.Height {
		for x := range reference.Image.Width {
			img.SetNRGBA(x, y, color.NRGBA{R: reference.Image.RGB[0], G: reference.Image.RGB[1], B: reference.Image.RGB[2], A: 255})
		}
	}
	referenceRaw, err := pipeline.RunRawWithImages(t.Context(), []image.Image{img})
	CheckT(t, err)
	checkHiddenStateReference(t, referenceRaw, reference.Image.hiddenStateReference, reference.Tolerance)
	config.Name = "imageModelPooler"
	config.Options = append(config.Options, pipelines.WithImageModelPooler())
	pooler, err := session.NewPipeline(config)
	hasPooler := false
	for _, output := range pipeline.Model.OutputsMeta {
		hasPooler = hasPooler || output.Name == "pooler_output"
	}
	if !hasPooler {
		assert.Error(t, err, "models without a pooler must not silently mean pool")
		return
	}
	CheckT(t, err)
	modelPooled, err := pooler.RunRaw(t.Context(), []string{imagePath})
	CheckT(t, err)
	if modelPooled == nil || len(modelPooled.Embeddings) != 1 {
		t.Fatal("model pooler did not return a rank-two tensor")
	}
	assert.Equal(t, "pooler_output", modelPooled.OutputName)
	var poolerDifference float64
	for j, value := range modelPooled.Embeddings[0] {
		poolerDifference += math.Abs(float64(value - result.Embeddings[0][j]))
	}
	assert.Greater(t, poolerDifference, 0.01, "model pooling and mean pooling are distinct operations")
}

func ImageFeatureExtractionPipelineValidation(t *testing.T, session *hugot.Session) {
	t.Helper()

	config := hugot.ImageFeatureExtractionConfig{
		ModelPath:    ModelsFolder + "Xenova_dino-vits16",
		Name:         "testImageFeatureExtractionValidation",
		OnnxFilename: "model.onnx",
	}
	pipeline, err := session.NewPipeline(config)
	CheckT(t, err)

	original := pipeline.Model.InputsMeta[0].Dimensions
	defer func() { pipeline.Model.InputsMeta[0].Dimensions = original }()
	pipeline.Model.InputsMeta[0].Dimensions = backends.NewShape(-1, -1, -1)
	assert.Error(t, pipeline.Validate(), "image feature extraction should require four-dimensional image inputs")
}

// image segmentation

func ImageSegmentationPipeline(t *testing.T, session *hugot.Session) {
	t.Helper()
	config := hugot.ImageSegmentationConfig{
		ModelPath: ModelsFolder + "Xenova_segformer-b0-finetuned-ade-512-512",
		Name:      "testImageSegmentation",
	}
	pipeline, err := session.NewPipeline(config)
	CheckT(t, err)
	result, err := pipeline.RunPipeline(t.Context(), []string{ModelsFolder + "imageData/cat.jpg"})
	CheckT(t, err)
	if len(result.Results) != 1 || result.Results[0].Width == 0 || result.Results[0].Height == 0 {
		t.Fatal("image segmentation inference returned no source-sized result")
	}

	// Explicit segmentation result for models/imageData/cat.jpg (source 640x480).
	// Each segment is a full-resolution mask covering 640*480 pixels.
	assert.Equal(t, 640, result.Results[0].Width, "segmentation width")
	assert.Equal(t, 480, result.Results[0].Height, "segmentation height")
	expectedSegments := []struct {
		Label string
		Class int
		Score float32
	}{
		{Label: "wall", Class: 0, Score: 0.011198625},
		{Label: "ceiling", Class: 5, Score: 0.021142391},
		{Label: "bed ", Class: 7, Score: 0.23426121},
		{Label: "person", Class: 12, Score: 0.1132925},
		{Label: "sofa", Class: 23, Score: 0.028294662},
		{Label: "armchair", Class: 30, Score: 0.0052581797},
		{Label: "cushion", Class: 39, Score: 0.053761255},
		{Label: "box", Class: 41, Score: 0.0030891823},
		{Label: "plaything", Class: 108, Score: 0.055784084},
	}
	assert.Len(t, result.Results[0].Segments, len(expectedSegments), "number of segmentation segments")
	for i, expected := range expectedSegments {
		seg := result.Results[0].Segments[i]
		assert.Equal(t, expected.Label, seg.Label, "segment %d label", i)
		assert.Equal(t, expected.Class, seg.Class, "segment %d class", i)
		assert.InDelta(t, float64(expected.Score), float64(seg.Score), 1e-3, "segment %d score", i)
		assert.Len(t, seg.Mask, 480, "segment %d mask rows", i)
		assert.Len(t, seg.Mask[0], 640, "segment %d mask cols", i)
	}
}

func ImageSegmentationPipelineValidation(t *testing.T, session *hugot.Session) {
	t.Helper()

	config := hugot.ImageSegmentationConfig{
		ModelPath: ModelsFolder + "Xenova_segformer-b0-finetuned-ade-512-512",
		Name:      "testImageSegmentationValidation",
		Options: []hugot.ImageSegmentationOption{
			pipelines.WithSegmentationLogitsOutput("segmentation_logits"),
		},
	}
	pipeline, err := session.NewPipeline(config)
	CheckT(t, err)

	original := pipeline.Model.InputsMeta[0].Dimensions
	defer func() { pipeline.Model.InputsMeta[0].Dimensions = original }()
	pipeline.Model.InputsMeta[0].Dimensions = backends.NewShape(-1, -1, -1)
	assert.Error(t, pipeline.Validate(), "image segmentation should require four-dimensional image inputs")
}

// depth estimation

func DepthEstimationPipeline(t *testing.T, session *hugot.Session) {
	t.Helper()
	config := hugot.DepthEstimationConfig{
		ModelPath: ModelsFolder + "Xenova_dpt-large",
		Name:      "testDepthEstimation",
	}
	pipeline, err := session.NewPipeline(config)
	CheckT(t, err)
	result, err := pipeline.RunPipeline(t.Context(), []string{ModelsFolder + "imageData/cat.jpg"})
	CheckT(t, err)
	if len(result.Results) != 1 || len(result.Results[0].DepthMap) == 0 {
		t.Fatal("depth estimation inference returned an empty depth map")
	}
	// Explicit depth-estimation result for models/imageData/cat.jpg (source 640x480).
	dm := result.Results[0]
	assert.Equal(t, 640, dm.Width, "depth width")
	assert.Equal(t, 480, dm.Height, "depth height")
	assert.Len(t, dm.DepthMap, 480, "depth map rows")
	assert.Len(t, dm.DepthMap[0], 640, "depth map cols")
	minDepth, maxDepth := dm.DepthMap[0][0], dm.DepthMap[0][0]
	for _, row := range dm.DepthMap {
		for _, v := range row {
			assert.False(t, math.IsNaN(float64(v)) || math.IsInf(float64(v), 0), "depth value must be finite")
			if v < minDepth {
				minDepth = v
			}
			if v > maxDepth {
				maxDepth = v
			}
		}
	}
	// Depth values for this input range ~[7.02, 30.00].
	assert.InDelta(t, 7.0227566, float64(minDepth), 2.0, "depth map minimum")
	assert.InDelta(t, 30.000578, float64(maxDepth), 2.0, "depth map maximum")
	assert.InDelta(t, 12.303157, float64(dm.DepthMap[0][0]), 0.1, "top-left depth sample")
	assert.InDelta(t, 8.83053, float64(dm.DepthMap[0][639]), 0.1, "top-right depth sample")
}

func DepthEstimationPipelineValidation(t *testing.T, session *hugot.Session) {
	t.Helper()

	config := hugot.DepthEstimationConfig{
		ModelPath: ModelsFolder + "Xenova_dpt-large",
		Name:      "testDepthEstimationValidation",
		Options: []hugot.DepthEstimationOption{
			pipelines.WithDepthOutput("depth"),
		},
	}
	pipeline, err := session.NewPipeline(config)
	CheckT(t, err)

	original := pipeline.Model.InputsMeta[0].Dimensions
	defer func() { pipeline.Model.InputsMeta[0].Dimensions = original }()
	pipeline.Model.InputsMeta[0].Dimensions = backends.NewShape(-1, -1, -1)
	assert.Error(t, pipeline.Validate(), "depth estimation should require four-dimensional image inputs")
}

// audio classification

func AudioClassificationPipeline(t *testing.T, session *hugot.Session) {
	t.Helper()
	config := backends.PipelineConfig[*pipelines.AudioClassificationPipeline]{
		ModelPath: ModelsFolder + "Xenova_wav2vec2-large-xlsr-53-gender-recognition-librispeech",
		Name:      "testAudioClassification",
		Options: []backends.PipelineOption[*pipelines.AudioClassificationPipeline]{
			pipelines.WithAudioTopK(2),
		},
	}
	pipeline, err := session.NewPipeline(config)
	CheckT(t, err)
	result, err := pipeline.RunFiles(t.Context(), []string{ModelsFolder + "audioData/librispeech.wav"})
	CheckT(t, err)
	assert.Len(t, result.Predictions, 1)
	assert.Len(t, result.Predictions[0], 2)
	assert.Equal(t, "male", result.Predictions[0][0].Label)
	assert.Greater(t, result.Predictions[0][0].Score, float32(0.99))
}

func AudioClassificationPipelineValidation(t *testing.T, _ *hugot.Session) {
	t.Helper()
	model := &backends.Model{
		InputsMeta:  []backends.InputOutputInfo{{Name: "input_values", Dimensions: backends.NewShape(-1, -1)}},
		OutputsMeta: []backends.InputOutputInfo{{Name: "logits", Dimensions: backends.NewShape(-1, 2)}},
		IDLabelMap:  map[int]string{0: "speech", 1: "music"},
	}
	pipeline := &pipelines.AudioClassificationPipeline{BasePipeline: &backends.BasePipeline{Model: model}, TopK: 1, IDLabelMap: model.IDLabelMap}
	assert.NoError(t, pipeline.Validate())
	originalDimensions := model.InputsMeta[0].Dimensions
	defer func() { model.InputsMeta[0].Dimensions = originalDimensions }()
	model.InputsMeta[0].Dimensions = backends.NewShape(-1)
	assert.Error(t, pipeline.Validate())
}

// Background removal

func BackgroundRemovalPipeline(t *testing.T, session *hugot.Session) {
	t.Helper()
	config := hugot.BackgroundRemovalConfig{
		ModelPath: ModelsFolder + "Xenova_modnet",
		Name:      "testBackgroundRemoval",
		Options: []hugot.BackgroundRemovalOption{
			pipelines.WithBackgroundRemovalOutput("output"),
			pipelines.WithPreprocessSteps[*pipelines.BackgroundRemovalPipeline](imageutil.ResizeStep(512)),
			pipelines.WithNormalizationSteps[*pipelines.BackgroundRemovalPipeline](
				imageutil.RescaleStep(),
				imageutil.PixelNormalizationStep([3]float32{0.5, 0.5, 0.5}, [3]float32{0.5, 0.5, 0.5}),
			),
		},
	}
	pipeline, err := session.NewPipeline(config)
	CheckT(t, err)
	images, err := imageutil.LoadImagesFromPaths(t.Context(), []string{ModelsFolder + "imageData/portrait.jpg"})
	CheckT(t, err)
	result, err := pipeline.RunWithImages(t.Context(), images)
	CheckT(t, err)
	if len(result.Results) != 1 {
		t.Fatal("background removal returned the wrong batch size")
	}
	w, h := images[0].Bounds().Dx(), images[0].Bounds().Dy()
	mask := result.Results[0].Mask
	assert.Equal(t, w, result.Results[0].Width)
	assert.Equal(t, h, result.Results[0].Height)
	if len(mask) != h {
		t.Fatal("background-removal mask must have source height")
	}
	for _, row := range mask {
		if len(row) != w {
			t.Fatal("background-removal mask must have source width")
		}
		for _, alpha := range row {
			if math.IsNaN(float64(alpha)) || math.IsInf(float64(alpha), 0) || alpha < 0 || alpha > 1 {
				t.Fatal("background-removal mask must contain finite alpha probabilities")
			}
		}
	}
	// The portrait's face is foreground; the upper corners are the pink backdrop.
	assert.Greater(t, mask[h/2][w/2], float32(0.9), "the subject should remain opaque")
	assert.Less(t, mask[h/10][w/10], float32(0.1), "the backdrop should be transparent")
	assert.Less(t, mask[h/10][9*w/10], float32(0.1), "the backdrop should be transparent")
}

func BackgroundRemovalPipelineValidation(t *testing.T, session *hugot.Session) {
	t.Helper()
	config := hugot.BackgroundRemovalConfig{
		ModelPath: ModelsFolder + "Xenova_modnet",
		Name:      "testBackgroundRemovalValidation",
		Options:   []hugot.BackgroundRemovalOption{pipelines.WithBackgroundRemovalOutput("output")},
	}
	pipeline, err := session.NewPipeline(config)
	CheckT(t, err)

	originalDimensions := pipeline.Model.InputsMeta[0].Dimensions
	defer func() { pipeline.Model.InputsMeta[0].Dimensions = originalDimensions }()
	pipeline.Model.InputsMeta[0].Dimensions = backends.NewShape(-1, -1, -1)
	assert.Error(t, pipeline.Validate())
}

// zero-shot image classification

func ZeroShotImageClassificationPipeline(t *testing.T, session *hugot.Session) {
	t.Helper()
	images, err := imageutil.LoadImagesFromPaths(t.Context(), []string{ModelsFolder + "imageData/cat.jpg"})
	CheckT(t, err)
	config := hugot.ZeroShotImageClassificationConfig{
		ModelPath: ModelsFolder + "Xenova_clip-vit-base-patch32",
		Name:      "testZeroShotImageClassification",
		Options: []hugot.ZeroShotImageClassificationOption{
			pipelines.WithImageLabels([]string{"cat", "dog"}),
			pipelines.WithImageTopK(2),
			pipelines.WithPreprocessSteps[*pipelines.ZeroShotImageClassificationPipeline](
				imageutil.ResizeStep(224),
				imageutil.CenterCropStep(224, 224),
			),
			pipelines.WithNormalizationSteps[*pipelines.ZeroShotImageClassificationPipeline](
				imageutil.RescaleStep(),
				imageutil.CLIPPixelNormalizationStep(),
			),
		},
	}
	pipeline, err := session.NewPipeline(config)
	CheckT(t, err)
	result, err := pipeline.RunWithImagesAndLabels(t.Context(), images, []string{"cat", "dog"})
	CheckT(t, err)
	assert.Len(t, result.Predictions, 1)
	assert.Len(t, result.Predictions[0], 2)
	assert.Equal(t, "cat", result.Predictions[0][0].Label)
}

func ZeroShotImageClassificationPipelineValidation(t *testing.T, session *hugot.Session) {
	t.Helper()
	config := hugot.ZeroShotImageClassificationConfig{
		ModelPath: ModelsFolder + "Xenova_clip-vit-base-patch32",
		Name:      "testZeroShotImageClassificationValidation",
		Options: []hugot.ZeroShotImageClassificationOption{
			pipelines.WithImageLabels([]string{"cat", "dog"}),
		},
	}
	pipeline, err := session.NewPipeline(config)
	CheckT(t, err)

	labelsInitial := pipeline.Labels
	defer func() { pipeline.Labels = labelsInitial }()
	pipeline.Labels = nil
	assert.Error(t, pipeline.Validate())
}

// automatic speech recognition

func AutomaticSpeechRecognitionPipeline(t *testing.T, session *hugot.Session) {
	t.Helper()
	config := backends.PipelineConfig[*pipelines.AutomaticSpeechRecognitionPipeline]{
		ModelPath: ModelsFolder + "Xenova_wav2vec2-base-960h",
		Name:      "testAutomaticSpeechRecognition",
	}
	pipeline, err := session.NewPipeline(config)
	CheckT(t, err)
	result, err := pipeline.RunFiles(t.Context(), []string{ModelsFolder + "audioData/librispeech.wav"})
	CheckT(t, err)
	assert.Len(t, result.Text, 1)
	assert.Contains(t, result.Text[0], "MISTER QUILTER",
		"ASR should recognize the speech sample, got: %q", result.Text[0])
}

func AutomaticSpeechRecognitionPipelineValidation(t *testing.T, session *hugot.Session) {
	t.Helper()
	config := backends.PipelineConfig[*pipelines.AutomaticSpeechRecognitionPipeline]{
		ModelPath: ModelsFolder + "Xenova_wav2vec2-base-960h",
		Name:      "testAutomaticSpeechRecognitionValidation",
	}
	pipeline, err := session.NewPipeline(config)
	CheckT(t, err)

	originalDimensions := pipeline.Model.InputsMeta[0].Dimensions
	defer func() { pipeline.Model.InputsMeta[0].Dimensions = originalDimensions }()
	pipeline.Model.InputsMeta[0].Dimensions = backends.NewShape(-1)
	assert.Error(t, pipeline.Validate())
}

// image-to-text

func NativeImageToTextPipeline(t *testing.T, session *hugot.Session) {
	t.Helper()
	config := hugot.ImageToTextConfig{
		ModelPath:    ModelsFolder + "Xenova_vit-gpt2-image-captioning",
		OnnxFilename: "encoder_model_quantized.onnx",
		Name:         "nativeImageToText",
		ModelLoading: backends.ModelLoadingONNX,
	}
	pipeline, err := session.NewPipeline(config)
	CheckT(t, err)
	inputs := []pipelines.ImageTextPrompt{
		{ImagePath: ModelsFolder + "imageData/cat.jpg"},
		{ImagePath: ModelsFolder + "imageData/portrait.jpg"},
	}
	result, err := pipeline.RunWithImages(t.Context(), inputs)
	CheckT(t, err)
	if result == nil || len(result.Responses) != len(inputs) {
		t.Fatal("native captioning did not preserve batch size")
	}
	assert.Contains(t, strings.ToLower(result.Responses[0]), "cat")
	assert.NotEmpty(t, result.Responses[1])
	assert.NotEqual(t, result.Responses[0], result.Responses[1], "captions should be image-conditioned")
	checkCaptionGenerationReference(t, pipeline)
	t.Logf("native captions: %q", result.Responses)
	for i, input := range inputs {
		single, err := pipeline.RunWithImages(t.Context(), []pipelines.ImageTextPrompt{input})
		CheckT(t, err)
		if single == nil || len(single.Responses) != 1 {
			t.Fatal("native captioning returned invalid single-image result")
		}
		assert.Equal(t, result.Responses[i], single.Responses[0], "batch order and deterministic greedy decoding")
	}
	_, err = pipeline.RunWithImages(t.Context(), nil)
	assert.Error(t, err)
	_, err = pipeline.RunWithImages(t.Context(), []pipelines.ImageTextPrompt{{ImagePath: inputs[0].ImagePath, Prompt: "describe"}})
	assert.Error(t, err)
	cancelled, cancel := context.WithCancel(t.Context())
	cancel()
	_, err = pipeline.RunWithImages(cancelled, inputs)
	assert.ErrorIs(t, err, context.Canceled)
	decoder, err := pipeline.Model.LoadGraph(t.Context(), "decoder_model_quantized.onnx")
	CheckT(t, err)
	CheckT(t, session.ClosePipeline(config.Name))
	_, err = decoder.RunTensors(t.Context(), nil)
	assert.ErrorContains(t, err, "closed", "closing the parent must close its decoder graph")
	_, err = pipeline.Model.LoadGraph(t.Context(), "decoder_model_quantized.onnx")
	assert.ErrorContains(t, err, "closed")
	CheckT(t, pipeline.Model.Close())
}

func ImageToTextPipeline(t *testing.T, session *hugot.Session) {
	t.Helper()
	skipGenerativePipeline(t)
	config := hugot.ImageToTextConfig{
		ModelPath: MultimodalModelPath,
		Name:      "imageToText",
	}
	pipeline, err := session.NewPipeline(config)
	CheckT(t, err)
	result, err := pipeline.RunWithImages(t.Context(), []pipelines.ImageTextPrompt{{
		ImagePath: ModelsFolder + "imageData/cat.jpg",
		Prompt:    "Describe this image in one short sentence.",
	}})
	CheckT(t, err)
	if result == nil || len(result.Responses) != 1 {
		t.Fatal("image-to-text returned an invalid response count")
	}
	assert.NotEmpty(t, result.Responses)
	assert.NotEmpty(t, result.Responses[0])
	assert.Contains(t, strings.ToLower(result.Responses[0]), "cat",
		"image-to-text caption should describe the cat, got: %q", result.Responses[0])
}

func ImageToTextPipelineValidation(t *testing.T, _ *hugot.Session) {
	t.Helper()
	model := &backends.Model{IsGenerative: true}
	pipeline, err := pipelines.NewImageToTextPipeline(t.Context(), hugot.ImageToTextConfig{Name: "imageToTextValidation"}, model)
	assert.NoError(t, err)
	pipeline.MaxLength = 0
	assert.Error(t, pipeline.Validate())
}

// image-text-to-text

func ImageTextToTextPipeline(t *testing.T, session *hugot.Session) {
	t.Helper()
	skipGenerativePipeline(t)
	config := hugot.ImageTextToTextConfig{
		ModelPath: MultimodalModelPath,
		Name:      "imageTextToText",
	}
	pipeline, err := session.NewPipeline(config)
	CheckT(t, err)
	result, err := pipeline.RunWithImages(t.Context(), []pipelines.ImageTextPrompt{
		{ImagePath: ModelsFolder + "imageData/cat.jpg", Prompt: "What is shown in this image?"},
		{ImagePath: ModelsFolder + "imageData/portrait.jpg", Prompt: "What is shown in this image?"},
	})
	CheckT(t, err)
	if result == nil || len(result.Responses) != 2 {
		t.Fatal("image-text-to-text returned an invalid response count")
	}
	assert.NotEmpty(t, result.Responses)
	assert.NotEmpty(t, result.Responses[0])
	assert.Contains(t, strings.ToLower(result.Responses[0]), "cat",
		"image-text-to-text answer should reference the cat, got: %q", result.Responses[0])
	assert.NotEmpty(t, result.Responses[1])
	assert.NotEqual(t, result.Responses[0], result.Responses[1], "contrasting images should produce grounded responses")
	assert.True(t, strings.Contains(strings.ToLower(result.Responses[1]), "woman") ||
		strings.Contains(strings.ToLower(result.Responses[1]), "person"), "portrait response should describe the person")
}

func ImageTextToTextPipelineValidation(t *testing.T, _ *hugot.Session) {
	t.Helper()
	model := &backends.Model{IsGenerative: true}
	pipeline, err := pipelines.NewImageTextToTextPipeline(t.Context(), hugot.ImageTextToTextConfig{Name: "imageTextToTextValidation"}, model)
	assert.NoError(t, err)
	pipeline.MaxLength = 0
	assert.Error(t, pipeline.Validate())
}

// No same name

func NoSameNamePipeline(t *testing.T, session *hugot.Session) {
	t.Helper()
	modelPath := ModelsFolder + "/KnightsAnalytics_distilbert-NER"
	configSimple := hugot.TokenClassificationConfig{
		ModelPath: modelPath,
		Name:      "testPipelineSimple",
		Options: []hugot.TokenClassificationOption{
			pipelines.WithSimpleAggregation(),
			pipelines.WithIgnoreLabels([]string{"O"}),
		},
	}
	_, err2 := session.NewPipeline(configSimple)
	if err2 != nil {
		t.FailNow()
	}
	_, err3 := session.NewPipeline(configSimple)
	assert.Error(t, err3)
}

// NoSameNameAcrossTypesPipeline verifies that pipeline names are unique per session across
// pipeline types, not only within a single type. Creating a second pipeline of a different type
// under an already-used name must fail.
func NoSameNameAcrossTypesPipeline(t *testing.T, session *hugot.Session) {
	t.Helper()

	tokenConfig := hugot.TokenClassificationConfig{
		ModelPath: ModelsFolder + "KnightsAnalytics_distilbert-NER",
		Name:      "sharedName",
		Options: []hugot.TokenClassificationOption{
			pipelines.WithSimpleAggregation(),
			pipelines.WithIgnoreLabels([]string{"O"}),
		},
	}
	if _, err := session.NewPipeline(tokenConfig); err != nil {
		t.Fatalf("failed to create token classification pipeline: %s", err)
	}

	textConfig := hugot.TextClassificationConfig{
		ModelPath: ModelsFolder + "Xenova_distilbert-base-uncased-finetuned-sst-2-english",
		Name:      "sharedName",
	}
	_, err := session.NewPipeline(textConfig)
	assert.Error(t, err, "expected an error creating a different pipeline type under an already-used name")
}

func DestroyPipelines(t *testing.T, session *hugot.Session) {
	t.Helper()

	modelPath := ModelsFolder + "/KnightsAnalytics_distilbert-NER"
	configSimple := hugot.TokenClassificationConfig{
		ModelPath: modelPath,
		Name:      "testClosePipeline",
		Options: []hugot.TokenClassificationOption{
			pipelines.WithSimpleAggregation(),
			pipelines.WithIgnoreLabels([]string{"O"}),
		},
	}
	_, err := session.NewPipeline(configSimple)
	CheckT(t, err)

	if len(session.GetModels()) != 1 {
		t.Fatal("Session should have 1 model")
	}

	for _, model := range session.GetModels() {
		if _, ok := model.Pipelines["testClosePipeline"]; !ok {
			t.Fatal("Pipeline alias was not added to the model")
		}
	}

	if err = session.ClosePipeline("testClosePipeline"); err != nil {
		t.Fatal(err)
	}

	if len(session.GetModels()) != 0 {
		t.Fatal("Session should have 0 models")
	}
	p, err := session.GetPipelines[*pipelines.TokenClassificationPipeline]()
	CheckT(t, err)
	if len(p) != 0 {
		t.Fatal("Session should have 0 token classification pipelines")
	}
}

func TextGenerationPipeline(t *testing.T, session *hugot.Session) {
	textGenerationPipeline(t, session, false)
}

func TextGenerationPipelineEngine(t *testing.T, session *hugot.Session) {
	textGenerationPipeline(t, session, true)
}

func textGenerationPipeline(t *testing.T, session *hugot.Session, engineMode bool) {
	t.Helper()
	skipGenerativePipeline(t)
	modelPath := ModelsFolder + "/KnightsAnalytics_qwen3-4B-int4"

	defer func(session *hugot.Session) {
		err := session.Destroy()
		CheckT(t, err)
	}(session)

	// Configure the text generation pipeline
	config := hugot.TextGenerationConfig{
		ModelPath: modelPath,
		Name:      "testPipeline",
		Options: []backends.PipelineOption[*pipelines.TextGenerationPipeline]{
			pipelines.WithMaxLength(2000),
			pipelines.WithSystemPrompt("You are a helpful assistant. Answer with a single very brief sentence."),
		},
	}

	// Create the pipeline
	textGenPipeline, err := session.NewPipeline(config)
	CheckT(t, err)

	tests := []struct {
		name             string
		input            [][]backends.Message
		expectedKeywords []string
	}{
		{
			name: "small test",
			input: [][]backends.Message{
				{
					{Role: "user", Content: "what is the capital of the Netherlands?"},
				},
			},
			expectedKeywords: []string{
				"Amsterdam",
			},
		},
		{
			name: "batched test",
			input: [][]backends.Message{
				{
					{Role: "user", Content: "what is the capital of the Netherlands?"},
				},
				{
					{Role: "user", Content: "who was the first president of the United States?"},
				},
				{
					{Role: "user", Content: "Solve this equation: 2 + 2 = ?"},
				},
			},
			expectedKeywords: []string{
				"Amsterdam",
				"George Washington",
				"4",
			},
		},
	}

	// Execute tests
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			batchResult, err := textGenPipeline.RunMessages(t.Context(), tt.input)
			if batchResult != nil {
				logGenerationFailure(t, tt.name, batchResult.Responses, err)
			}
			CheckT(t, err)
			outputs := batchResult.GetOutput()
			for i := range len(outputs) {
				generatedString := outputs[i].(string)
				fmt.Println(generatedString + "\n")
				if !strings.Contains(generatedString, tt.expectedKeywords[i]) {
					t.Fatalf("Test %s failed: expected keywords '%s' not found in '%s'", tt.name, tt.expectedKeywords[i], generatedString)
				}
			}
		})
	}

	// streaming test
	streamingConfig := hugot.TextGenerationConfig{
		ModelPath: modelPath,
		Name:      "testPipelineStreaming",
		Options: []backends.PipelineOption[*pipelines.TextGenerationPipeline]{
			pipelines.WithMaxLength(2000),
			pipelines.WithStreaming(),
		},
	}
	streamingPipeline, err := session.NewPipeline(streamingConfig)
	CheckT(t, err)

	t.Run("streaming test", func(t *testing.T) {
		input := [][]backends.Message{
			{
				{Role: "system", Content: "you are a helpful assistant."},
				{Role: "user", Content: "Solve this equation: 2 + 2 = ? Be very brief in your explanation."},
			},
		}
		output, err := streamingPipeline.RunMessages(t.Context(), input)
		if output == nil {
			CheckT(t, err)
			return
		}
		var fullAnswer strings.Builder
		for token := range output.TokenStream {
			fullAnswer.WriteString(token.Token)
		}
		if !strings.Contains(fullAnswer.String(), "4") {
			t.Fatalf("Expected answer to contain '4', got '%s'", fullAnswer.String())
		}
		if err != nil {
			t.Logf("streaming text generation failed: %v; partial response: %q", err, fullAnswer.String())
		}
		CheckT(t, err)
	})

	// tools test
	t.Run("tools test", func(t *testing.T) {
		// Lark grammar that constrains both <tool_call> blocks to valid JSON.
		//
		// Requirements:
		//   - <tool_call> and </tool_call> must be "special": false in tokenizer.json so that
		//     llguidance can match them as regular byte sequences. Marking them "special": true
		//     promotes them to control tokens that the guidance system cannot byte-force, causing
		//     "token doesn't satisfy the grammar" errors.
		//   - The grammar expects exactly two tool calls for this query.
		toolGrammar := `start: fun_call fun_call /\n?/
fun_call: "<tool_call>\n" tool_json "\n</tool_call>\n"
tool_json: %json {"anyOf": [` +
			`{"type":"object","required":["name","arguments"],"additionalProperties":false,` +
			`"properties":{"name":{"const":"get_current_time"},"arguments":{"type":"object","properties":{},"additionalProperties":false}}},` +
			`{"type":"object","required":["name","arguments"],"additionalProperties":false,` +
			`"properties":{"name":{"const":"get_weather"},"arguments":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false}}}` +
			`]}`
		guidance := &backends.Guidance{
			Type:           backends.GuidanceTypeLarkGrammar,
			Data:           toolGrammar,
			EnableFFTokens: true,
		}

		config = hugot.TextGenerationConfig{
			ModelPath: modelPath,
			Name:      "testPipelineWithTools",
			Options: []backends.PipelineOption[*pipelines.TextGenerationPipeline]{
				pipelines.WithMaxLength(2000),
				pipelines.WithSystemPrompt("You are a helpful assistant that answers questions using tools."),
				pipelines.WithGuidance(guidance),
			},
		}

		// Create the pipeline
		textGenPipeline, err = session.NewPipeline(config)
		CheckT(t, err)

		// Two minimal Hermes-style tool definitions.
		tools := []string{
			`{
			"type": "function",
			"function": {
				"name": "get_current_time",
				"description": "Returns the current UTC time.",
				"parameters": {
					"type": "object",
					"properties": {},
					"required": []
				}
			}
		}`,
			`{
			"type": "function",
			"function": {
				"name": "get_weather",
				"description": "Returns the current weather for a given city.",
				"parameters": {
					"type": "object",
					"properties": {
						"city": {
							"type": "string",
							"description": "The name of the city."
						}
					},
					"required": ["city"]
				}
			}
		}`,
		}

		messages := [][]backends.Message{
			{
				{Role: "user", Content: "What time is it right now, and what's the weather like in Paris?"},
			},
		}

		ctx, cancel := context.WithTimeout(t.Context(), 5*time.Minute)
		defer cancel()

		toolOutput, err := textGenPipeline.RunMessagesWithOverrides(ctx, messages, tools, nil)
		if engineMode {
			if err == nil || !strings.Contains(err.Error(), "EnableFFTokens") {
				t.Fatalf("expected engine mode to reject unsupported EnableFFTokens guidance, got %v", err)
			}
			return
		}
		if toolOutput != nil {
			logGenerationFailure(t, "tool-call generation", toolOutput.Responses, err)
		}
		CheckT(t, err)
		CheckToolCalls(t, toolOutput.Responses[0])
	})
}

func TextGenerationPipelineValidation(t *testing.T, session *hugot.Session) {
	t.Helper()
	skipGenerativePipeline(t)
	defer func(session *hugot.Session) {
		err := session.Destroy()
		CheckT(t, err)
	}(session)

	// Configure the text generation pipeline
	config := hugot.TextGenerationConfig{
		ModelPath: ModelsFolder + "/KnightsAnalytics_qwen3-4B-int4",
		Name:      "testPipeline",
		Options:   []backends.PipelineOption[*pipelines.TextGenerationPipeline]{},
	}
	pipeline, err := session.NewPipeline(config)
	CheckT(t, err)
	pipeline.MaxLength = -100
	err = pipeline.Validate()
	assert.Error(t, err)
}

// zero-shot object detection

func ZeroShotObjectDetectionPipeline(t *testing.T, session *hugot.Session) {
	t.Helper()
	config := backends.PipelineConfig[*pipelines.ZeroShotObjectDetectionPipeline]{
		ModelPath: ModelsFolder + "Xenova_owlv2-base-patch16",
		Name:      "testZeroShotObjectDetection",
		Options: []backends.PipelineOption[*pipelines.ZeroShotObjectDetectionPipeline]{
			pipelines.WithZeroShotObjectLabels([]string{"cat", "dog"}),
		},
	}
	pipeline, err := session.NewPipeline(config)
	CheckT(t, err)
	result, err := pipeline.RunWithLabels(t.Context(), []string{ModelsFolder + "imageData/cat.jpg"}, nil)
	CheckT(t, err)
	assert.Len(t, result.Detections, 1)
}

func ZeroShotObjectDetectionPipelineValidation(t *testing.T, session *hugot.Session) {
	t.Helper()
	config := backends.PipelineConfig[*pipelines.ZeroShotObjectDetectionPipeline]{
		ModelPath: ModelsFolder + "Xenova_owlv2-base-patch16",
		Name:      "testZeroShotObjectDetectionValidation",
		Options: []backends.PipelineOption[*pipelines.ZeroShotObjectDetectionPipeline]{
			pipelines.WithZeroShotObjectLabels([]string{"cat", "dog"}),
		},
	}
	pipeline, err := session.NewPipeline(config)
	CheckT(t, err)
	assert.NoError(t, pipeline.Validate())
}

// mask generation

func MaskGenerationPipeline(t *testing.T, session *hugot.Session) {
	t.Helper()
	config := backends.PipelineConfig[*pipelines.MaskGenerationPipeline]{
		ModelPath:    ModelsFolder + "Xenova_slimsam-77-uniform",
		OnnxFilename: "vision_encoder.onnx",
		Name:         "testMaskGeneration",
	}
	pipeline, err := session.NewPipeline(config)
	CheckT(t, err)
	result, err := pipeline.RunPipeline(t.Context(), []string{ModelsFolder + "imageData/cat.jpg"})
	CheckT(t, err)
	assert.Len(t, result.Results, 1)
	images, err := imageutil.LoadImagesFromPaths(t.Context(), []string{ModelsFolder + "imageData/cat.jpg"})
	CheckT(t, err)
	proposal := result.Results[0]
	assert.Equal(t, images[0].Bounds().Dx(), proposal.Width)
	assert.Equal(t, images[0].Bounds().Dy(), proposal.Height)
	assert.Greater(t, len(proposal.Masks), 1)
	for _, mask := range proposal.Masks {
		assert.Greater(t, mask.Area, 0)
		assert.GreaterOrEqual(t, mask.Score, pipeline.PredictedIOUThreshold)
		assert.LessOrEqual(t, mask.Score, float32(1))
		assert.GreaterOrEqual(t, mask.StabilityScore, pipeline.StabilityThreshold)
		assert.LessOrEqual(t, mask.StabilityScore, float32(1))
		assert.Len(t, mask.Mask, proposal.Height)
		area := 0
		for _, row := range mask.Mask {
			assert.Len(t, row, proposal.Width)
			for _, foreground := range row {
				if foreground {
					area++
				}
			}
		}
		assert.Equal(t, area, mask.Area)
		assert.GreaterOrEqual(t, mask.Box[0], 0)
		assert.GreaterOrEqual(t, mask.Box[1], 0)
		assert.Less(t, mask.Box[2], proposal.Width)
		assert.Less(t, mask.Box[3], proposal.Height)
	}
}

func MaskGenerationPipelineValidation(t *testing.T, _ *hugot.Session) {
	t.Helper()
	pipeline := &pipelines.MaskGenerationPipeline{
		BasePipeline: &backends.BasePipeline{Model: &backends.Model{}},
	}
	assert.Error(t, pipeline.Validate())
}

// zero-shot audio classification

func ZeroShotAudioClassificationPipeline(t *testing.T, session *hugot.Session) {
	t.Helper()
	config := backends.PipelineConfig[*pipelines.ZeroShotAudioClassificationPipeline]{
		ModelPath: ModelsFolder + "Xenova_larger_clap_music_and_speech",
		Name:      "testZeroShotAudioClassification",
		Options: []backends.PipelineOption[*pipelines.ZeroShotAudioClassificationPipeline]{
			pipelines.WithZeroShotAudioLabels([]string{"a man speaking", "a classical music performance"}),
			pipelines.WithZeroShotAudioTopK(2),
		},
	}
	pipeline, err := session.NewPipeline(config)
	CheckT(t, err)
	result, err := pipeline.RunFiles(t.Context(), []string{ModelsFolder + "audioData/librispeech.wav"})
	CheckT(t, err)
	assert.Len(t, result.Predictions, 1)
	assert.Len(t, result.Predictions[0], 2)
	assert.Equal(t, "a man speaking", result.Predictions[0][0].Label)
	assert.Greater(t, result.Predictions[0][0].Score, result.Predictions[0][1].Score)
}

func ZeroShotAudioClassificationPipelineValidation(t *testing.T, session *hugot.Session) {
	t.Helper()
	config := backends.PipelineConfig[*pipelines.ZeroShotAudioClassificationPipeline]{
		ModelPath: ModelsFolder + "Xenova_larger_clap_music_and_speech",
		Name:      "testZeroShotAudioClassification",
		Options: []backends.PipelineOption[*pipelines.ZeroShotAudioClassificationPipeline]{
			pipelines.WithZeroShotAudioLabels([]string{"speech", "music", "applause"}),
			pipelines.WithZeroShotAudioTopK(2),
		},
	}
	pipeline, err := session.NewPipeline(config)
	CheckT(t, err)
	assert.NoError(t, pipeline.Validate())
	pipeline.Labels = []string{"speech", "music", "applause", "footsteps"}
	assert.NoError(t, pipeline.Validate(), "candidate labels are runtime text inputs, not model output classes")
}

// visual question answering

func NativeVisualQuestionAnsweringPipeline(t *testing.T, session *hugot.Session) {
	t.Helper()
	config := backends.PipelineConfig[*pipelines.VisualQuestionAnsweringPipeline]{
		ModelPath: ModelsFolder + "KnightsAnalytics_vilt-b32-finetuned-vqa", Name: "nativeVisualQuestionAnswering",
		ModelLoading: backends.ModelLoadingONNX,
		Options:      []backends.PipelineOption[*pipelines.VisualQuestionAnsweringPipeline]{pipelines.WithVisualQuestionAnsweringTopK(5)},
	}
	pipeline, err := session.NewPipeline(config)
	CheckT(t, err)
	result, err := pipeline.RunPipeline(t.Context(), []pipelines.VisualQuestionAnsweringInput{
		{ImagePath: ModelsFolder + "imageData/cat.jpg", Question: "What animal is shown in this picture?"},
		{ImagePath: ModelsFolder + "imageData/portrait.jpg", Question: "What animal is shown in this picture?"},
	})
	CheckT(t, err)
	if result == nil || len(result.Answers) != 2 || len(result.Responses) != 2 {
		t.Fatal("native VQA did not preserve batch order and answer count")
	}
	for i, answers := range result.Answers {
		if !assert.Len(t, answers, 5) {
			continue
		}
		assert.Equal(t, answers[0].Answer, result.Responses[i])
		for j, answer := range answers {
			assert.NotEmpty(t, answer.Answer)
			assert.False(t, math.IsNaN(float64(answer.Score)) || math.IsInf(float64(answer.Score), 0))
			assert.GreaterOrEqual(t, answer.Score, float32(0))
			assert.LessOrEqual(t, answer.Score, float32(1))
			if j > 0 {
				assert.GreaterOrEqual(t, answers[j-1].Score, answer.Score)
			}
		}
	}
	assert.Contains(t, strings.ToLower(result.Responses[0]), "cat")
	assert.NotEqual(t, result.Responses[0], result.Responses[1], "changing the image should change the grounded answer")

	var reference struct {
		Tolerance float64 `json:"score_absolute_tolerance"`
		Cases     []struct {
			Width, Height    int
			Question, Answer string
			Tokens           struct {
				IDs       [][]uint32 `json:"input_ids"`
				Types     [][]uint32 `json:"token_type_ids"`
				Attention [][]uint32 `json:"attention_mask"`
			}
			TopIDs []int     `json:"top_ids"`
			Scores []float32 `json:"top_scores"`
		}
	}
	CheckT(t, json.Unmarshal(embedded.ViltReferenceByte, &reference))
	for _, tc := range reference.Cases {
		batch := backends.NewBatch(1)
		backends.TokenizeInputs(batch, pipeline.Model.Tokenizer, []string{tc.Question})
		if len(batch.Input) != 1 || len(tc.Tokens.IDs) != 1 || len(tc.Tokens.Types) != 1 || len(tc.Tokens.Attention) != 1 {
			t.Fatal("invalid VQA tokenizer/reference batch shape")
		}
		assert.EqualValues(t, tc.Tokens.IDs[0], batch.Input[0].TokenIDs)
		assert.EqualValues(t, tc.Tokens.Types[0], batch.Input[0].TypeIDs)
		assert.EqualValues(t, tc.Tokens.Attention[0], batch.Input[0].AttentionMask)
		CheckT(t, batch.Destroy())
		img := image.NewRGBA(image.Rect(0, 0, tc.Width, tc.Height))
		for y := range tc.Height {
			for x := range tc.Width {
				img.SetRGBA(x, y, color.RGBA{R: uint8((3*x + y) % 256), G: uint8((5*y + x) % 256), B: uint8((x + 7*y) % 256), A: 255})
			}
		}
		actual, err := pipeline.RunWithImages(t.Context(), []backends.ImageTextInput{{Image: img, Text: tc.Question}})
		CheckT(t, err)
		if actual == nil || len(actual.Answers) != 1 || len(actual.Responses) != 1 || len(actual.Answers[0]) != len(tc.TopIDs) || len(tc.Scores) != len(tc.TopIDs) {
			t.Fatal("native VQA reference answer shape mismatch")
		}
		assert.Equal(t, tc.Answer, actual.Responses[0])
		for j, answer := range actual.Answers[0] {
			assert.Equal(t, pipeline.Model.IDLabelMap[tc.TopIDs[j]], answer.Answer)
			assert.InDelta(t, tc.Scores[j], answer.Score, reference.Tolerance)
		}
	}
	config.Name = "nativeVisualQuestionAnsweringTopOne"
	config.Options = []backends.PipelineOption[*pipelines.VisualQuestionAnsweringPipeline]{pipelines.WithVisualQuestionAnsweringTopK(1)}
	topOne, err := session.NewPipeline(config)
	CheckT(t, err)
	assert.Same(t, pipeline.GetModel(), topOne.GetModel(), "equivalent ONNX loads should reuse the same model")
	one, err := topOne.RunPipeline(t.Context(), []pipelines.VisualQuestionAnsweringInput{{ImagePath: ModelsFolder + "imageData/cat.jpg", Question: "What animal is shown in this picture?"}})
	CheckT(t, err)
	if one == nil || len(one.Answers) != 1 || len(one.Answers[0]) != 1 {
		t.Fatal("top_k=1 did not return exactly one answer")
	}
	assert.Equal(t, result.Answers[0][0], one.Answers[0][0])
	CheckT(t, session.ClosePipeline(config.Name))
	assert.NoError(t, pipeline.Validate(), "closing a shared pipeline must preserve the native model")
}

func VisualQuestionAnsweringPipeline(t *testing.T, session *hugot.Session) {
	t.Helper()
	skipGenerativePipeline(t)
	config := backends.PipelineConfig[*pipelines.VisualQuestionAnsweringPipeline]{
		ModelPath: MultimodalModelPath,
		Name:      "testVisualQuestionAnswering",
	}
	pipeline, err := session.NewPipeline(config)
	CheckT(t, err)
	result, err := pipeline.RunPipeline(t.Context(), []pipelines.VisualQuestionAnsweringInput{{
		ImagePath: ModelsFolder + "imageData/cat.jpg",
		Question:  "What animal is in this image? Answer with one word.",
	}})
	CheckT(t, err)
	if result == nil || len(result.Responses) != 1 {
		t.Fatal("visual question answering returned an invalid response count")
	}
	assert.NotEmpty(t, result.Responses)
	assert.NotEmpty(t, result.Responses[0])
	assert.Contains(t, strings.ToLower(result.Responses[0]), "cat",
		"the image shows a cat, got: %q", result.Responses[0])
}

func VisualQuestionAnsweringPipelineValidation(t *testing.T, session *hugot.Session) {
	t.Helper()
	skipGenerativePipeline(t)
	config := backends.PipelineConfig[*pipelines.VisualQuestionAnsweringPipeline]{
		ModelPath: ModelsFolder + "KnightsAnalytics_qwen3-4B-int4",
		Name:      "testVisualQuestionAnsweringValidation",
	}
	pipeline, err := session.NewPipeline(config)
	CheckT(t, err)
	pipeline.MaxLength = 0
	assert.Error(t, pipeline.Validate())
}

// document question answering

func DocumentQuestionAnsweringPipeline(t *testing.T, session *hugot.Session) {
	t.Helper()
	config := backends.PipelineConfig[*pipelines.DocumentQuestionAnsweringPipeline]{
		ModelPath:    ModelsFolder + "Xenova_donut-base-finetuned-docvqa",
		OnnxFilename: "encoder_model_quantized.onnx",
		Name:         "testDocumentQuestionAnswering",
	}
	pipeline, err := session.NewPipeline(config)
	CheckT(t, err)
	result, err := pipeline.RunPipeline(t.Context(), []pipelines.DocumentQuestionAnsweringInput{{
		DocumentPath: ModelsFolder + "imageData/invoice.png",
		Question:     "What is the invoice number?",
	}})
	CheckT(t, err)
	if result == nil {
		t.Fatal("document question answering returned no result")
	}
	if assert.Len(t, result.Results, 1) {
		assert.Equal(t, "us-001", strings.ToLower(result.Results[0].Answer))
		assert.Equal(t, result.Results[0], result.GetOutput()[0])
	}
}

func DocumentQuestionAnsweringPipelineValidation(t *testing.T, session *hugot.Session) {
	t.Helper()
	config := backends.PipelineConfig[*pipelines.DocumentQuestionAnsweringPipeline]{
		ModelPath:    ModelsFolder + "Xenova_donut-base-finetuned-docvqa",
		OnnxFilename: "encoder_model_quantized.onnx",
		Name:         "testDocumentQuestionAnsweringValidation",
	}
	pipeline, err := session.NewPipeline(config)
	CheckT(t, err)
	pipeline.MaxLength = 0
	assert.Error(t, pipeline.Validate())
}

// table question answering

func TableQuestionAnsweringPipeline(t *testing.T, session *hugot.Session) {
	t.Helper()
	config := backends.PipelineConfig[*pipelines.TableQuestionAnsweringPipeline]{
		ModelPath: ModelsFolder + "KnightsAnalytics_tapas-base-finetuned-sqa",
		Name:      "testTableQuestionAnswering",
	}
	pipeline, err := session.NewPipeline(config)
	CheckT(t, err)
	result, err := pipeline.RunPipeline(t.Context(), []pipelines.TableQuestionAnsweringInput{{
		Table:    [][]string{{"Name", "Age"}, {"Alice", "30"}, {"Bob", "42"}},
		Question: "How old is Bob?",
	}})
	CheckT(t, err)
	if result == nil {
		t.Fatal("table question answering returned no result")
	}
	if assert.Len(t, result.Results, 1) {
		assert.Equal(t, "42", result.Results[0].Answer)
		assert.Equal(t, []string{"42"}, result.Results[0].Cells)
		assert.Equal(t, [][2]int{{1, 1}}, result.Results[0].Coordinates)
		assert.Equal(t, "NONE", result.Results[0].Aggregator)
		assert.Equal(t, result.Results[0], result.GetOutput()[0])
	}
	table := [][]string{{"Name", "Age"}, {"Alice", "30"}, {"Bob", "42"}}
	questions := []string{"How old is Bob?", "What is his name?", "How old is Alice?"}
	sequential, err := pipeline.RunSequential(t.Context(), table, questions)
	CheckT(t, err)
	if sequential == nil || len(sequential.Results) != len(questions) {
		t.Fatal("sequential TAPAS did not preserve question order")
	}
	for i, answer := range []string{"42", "Bob", "30"} {
		assert.Equal(t, answer, sequential.Results[i].Answer)
		assert.Equal(t, "NONE", sequential.Results[i].Aggregator)
	}
	repeated, err := pipeline.RunSequential(t.Context(), table, questions)
	CheckT(t, err)
	assert.Equal(t, sequential, repeated, "TAPAS conversational state must remain call-local")
	independent, err := pipeline.RunPipeline(t.Context(), []pipelines.TableQuestionAnsweringInput{{Table: table, Question: questions[0]}})
	CheckT(t, err)
	assert.Equal(t, result, independent, "sequential state must not leak into independent inputs")
	_, err = pipeline.RunSequential(t.Context(), table, nil)
	assert.Error(t, err)
	cancelled, cancel := context.WithCancel(t.Context())
	cancel()
	_, err = pipeline.RunSequential(cancelled, table, questions)
	assert.ErrorIs(t, err, context.Canceled)
}

func TableQuestionAnsweringPipelineValidation(t *testing.T, session *hugot.Session) {
	t.Helper()
	config := backends.PipelineConfig[*pipelines.TableQuestionAnsweringPipeline]{
		ModelPath: ModelsFolder + "KnightsAnalytics_tapas-base-finetuned-sqa",
		Name:      "testTableQuestionAnsweringValidation",
	}
	pipeline, err := session.NewPipeline(config)
	CheckT(t, err)
	pipeline.MaxLength = 0
	assert.Error(t, pipeline.Validate())
}

func TableQuestionAnsweringAggregation(t *testing.T, session *hugot.Session) {
	t.Helper()
	config := backends.PipelineConfig[*pipelines.TableQuestionAnsweringPipeline]{
		ModelPath: ModelsFolder + "KnightsAnalytics_tapas-base-finetuned-wtq",
		Name:      "testTableQuestionAnsweringAggregation",
	}
	pipeline, err := session.NewPipeline(config)
	CheckT(t, err)

	type reference struct {
		OnnXContract struct {
			InputNames  []string
			OutputNames []string
		} `json:"onnx_contract"`
		VerificationMetrics struct {
			Atol float64 `json:"atol"`
			Rtol float64 `json:"rtol"`
		} `json:"verification_metrics"`
		ReferenceTestCase struct {
			Table struct {
				Actors []string `json:"Actors"`
				Age    []string `json:"Age"`
			} `json:"table"`
			Queries []string `json:"queries"`
			Inputs  struct {
				InputIDs      [][]int64   `json:"input_ids"`
				AttentionMask [][]int64   `json:"attention_mask"`
				TokenTypeIDs  [][][]int64 `json:"token_type_ids"`
			} `json:"inputs"`
			ExpectedOutputs struct {
				Logits            [][]float32 `json:"logits"`
				LogitsAggregation [][]float32 `json:"logits_aggregation"`
			} `json:"expected_outputs"`
			Interpretation []struct {
				AggregationOp       float32   `json:"aggregation_op"`
				SelectedCellIndices []float32 `json:"selected_cell_indices"`
			} `json:"interpretation"`
		} `json:"reference_test_case"`
	}

	var ref reference
	CheckT(t, json.Unmarshal(embedded.TapasAggregationReferenceByte, &ref))

	// Verify the ONNX contract matches model metadata.
	for _, inName := range ref.OnnXContract.InputNames {
		found := false
		for _, meta := range pipeline.Model.InputsMeta {
			if meta.Name == inName {
				found = true
				break
			}
		}
		assert.True(t, found, "TAPAS WTQ input %q missing from ONNX contract", inName)
	}
	for _, outName := range ref.OnnXContract.OutputNames {
		found := false
		for _, meta := range pipeline.Model.OutputsMeta {
			if meta.Name == outName {
				found = true
				break
			}
		}
		assert.True(t, found, "TAPAS WTQ output %q missing from ONNX contract", outName)
	}

	// Feed the pre-encoded batch tensors directly through the ONNX graph.
	tc := ref.ReferenceTestCase
	n := int64(len(tc.Inputs.InputIDs[0]))
	ttids := make([]int64, 0, len(tc.Inputs.TokenTypeIDs)*int(n)*7)
	for _, batch := range tc.Inputs.TokenTypeIDs {
		for _, row := range batch {
			for _, v := range row {
				ttids = append(ttids, v)
			}
		}
	}
	inputs := map[string]backends.Tensor{
		"input_ids":      {Shape: []int64{int64(len(tc.Inputs.InputIDs)), n}, Data: flatten2d(t, tc.Inputs.InputIDs)},
		"attention_mask": {Shape: []int64{int64(len(tc.Inputs.AttentionMask)), n}, Data: flatten2d(t, tc.Inputs.AttentionMask)},
		"token_type_ids": {Shape: []int64{int64(len(tc.Inputs.TokenTypeIDs)), n, 7}, Data: ttids},
	}
	outputs, err := pipeline.Model.RunTensors(t.Context(), inputs)
	CheckT(t, err)

	agg, ok := outputs["logits_aggregation"]
	if !ok {
		t.Fatal("TAPAS WTQ ONNX graph did not produce logits_aggregation")
	}
	aggVals, ok := agg.Data.([]float32)
	if !ok {
		t.Fatal("logits_aggregation is not []float32")
	}
	batchSize := int64(len(tc.ExpectedOutputs.LogitsAggregation))

	for b := range batchSize {
		rowStart := b * int64(len(tc.ExpectedOutputs.LogitsAggregation[0]))
		rowEnd := rowStart + int64(len(tc.ExpectedOutputs.LogitsAggregation[0]))
		row := aggVals[rowStart:rowEnd]

		// Compare to reference within the pinned tolerance.
		for i, expected := range tc.ExpectedOutputs.LogitsAggregation[b] {
			got := row[int64(i)]
			tol := ref.VerificationMetrics.Rtol*math.Abs(float64(expected)) + ref.VerificationMetrics.Atol
			assert.InDelta(t, float64(expected), float64(got), tol,
				"WTQ aggregation[%d][%d] out of tolerance", b, i)
		}

		// Decode best aggregation index and verify it matches the reference.
		best := 0
		for i, v := range row {
			if v > row[best] {
				best = i
			}
		}
		assert.Equal(t, int(tc.Interpretation[b].AggregationOp), best,
			"WTQ aggregation query %q: best index %d != expected %d",
			tc.Queries[b], best, int(tc.Interpretation[b].AggregationOp))
	}
	t.Logf("WTQ aggregation reference verified for %d queries: %v", len(tc.Queries), tc.Queries)
}

func flatten2d(t *testing.T, vals [][]int64) []int64 {
	t.Helper()
	if len(vals) == 0 {
		return nil
	}
	out := make([]int64, 0, len(vals)*len(vals[0]))
	for _, row := range vals {
		for _, v := range row {
			out = append(out, v)
		}
	}
	return out
}

// text-to-speech

func TextToSpeechPipeline(t *testing.T, session *hugot.Session) {
	t.Helper()
	config := backends.PipelineConfig[*pipelines.TextToSpeechPipeline]{
		ModelPath: ModelsFolder + "Xenova_mms-tts-eng",
		Name:      "testTextToSpeech",
		Options: []backends.PipelineOption[*pipelines.TextToSpeechPipeline]{
			pipelines.WithTextToSpeechSampleRate(16000),
		},
	}
	pipeline, err := session.NewPipeline(config)
	CheckT(t, err)
	result, err := pipeline.RunText(t.Context(), []string{"hello world"})
	CheckT(t, err)
	if result == nil || len(result.Audio) != 1 || len(result.Audio[0].Samples) == 0 {
		t.Fatal("text-to-speech inference returned no audio")
	}
	assert.Equal(t, 16000, result.Audio[0].SampleRate)
	for _, sample := range result.Audio[0].Samples {
		if math.IsNaN(float64(sample)) || math.IsInf(float64(sample), 0) {
			t.Fatal("text-to-speech returned nonfinite audio")
		}
	}
}

func TextToSpeechPipelineValidation(t *testing.T, session *hugot.Session) {
	t.Helper()
	config := backends.PipelineConfig[*pipelines.TextToSpeechPipeline]{
		ModelPath: ModelsFolder + "Xenova_mms-tts-eng",
		Name:      "testTextToSpeechValidation",
	}
	pipeline, err := session.NewPipeline(config)
	CheckT(t, err)
	pipeline.SampleRate = 0
	assert.Error(t, pipeline.Validate())
}

// text-to-audio

func TextToAudioPipeline(t *testing.T, session *hugot.Session) {
	t.Helper()
	config := backends.PipelineConfig[*pipelines.TextToAudioPipeline]{
		ModelPath: ModelsFolder + "Xenova_mms-tts-eng",
		Name:      "testTextToAudio",
		Options: []backends.PipelineOption[*pipelines.TextToAudioPipeline]{
			pipelines.WithTextToAudioSampleRate(16000),
		},
	}
	pipeline, err := session.NewPipeline(config)
	CheckT(t, err)
	result, err := pipeline.RunText(t.Context(), []string{"hello world"})
	CheckT(t, err)
	if result == nil || len(result.Audio) != 1 || len(result.Audio[0].Samples) == 0 {
		t.Fatal("text-to-audio inference returned no audio")
	}
	assert.Equal(t, 16000, result.Audio[0].SampleRate)
	for _, sample := range result.Audio[0].Samples {
		if math.IsNaN(float64(sample)) || math.IsInf(float64(sample), 0) {
			t.Fatal("text-to-audio returned nonfinite audio")
		}
	}
}

func TextToAudioPipelineValidation(t *testing.T, session *hugot.Session) {
	t.Helper()
	config := backends.PipelineConfig[*pipelines.TextToAudioPipeline]{
		ModelPath: ModelsFolder + "Xenova_mms-tts-eng",
		Name:      "testTextToAudioValidation",
	}
	pipeline, err := session.NewPipeline(config)
	CheckT(t, err)
	pipeline.SampleRate = 0
	assert.Error(t, pipeline.Validate())
}

// QUESTION ANSWERING

func QuestionAnsweringPipeline(t *testing.T, session *hugot.Session) {
	t.Helper()

	modelPath := ModelsFolder + "/KnightsAnalytics_distilbert-onnx"

	config := hugot.QuestionAnsweringConfig{
		ModelPath: modelPath,
		Name:      "testQAPipeline",
	}
	pipeline, err := session.NewPipeline(config)
	CheckT(t, err)

	// Context is a JSON document; questions target specific property values.
	contextJSON := `{"product": "coffee maker", "brand": "Acme", "price": 49.99, "currency": "USD", "in_stock": true}`

	inputs := []pipelines.QuestionAnsweringInput{
		{Question: "What is the brand?", Context: contextJSON},
		{Question: "What is the currency?", Context: contextJSON},
	}

	result, err := pipeline.RunPipeline(t.Context(), inputs)
	CheckT(t, err)

	assert.Equal(t, 2, len(result.Outputs), "expected one result per input")

	brand := result.Outputs[0][0]
	assert.Greater(t, brand.Score, float32(0), "brand answer score should be > 0")
	assert.Contains(t, contextJSON[brand.Start:brand.End], brand.Answer, "answer should be a substring of the context")
	assert.Contains(t, strings.ToLower(brand.Answer), "acme", "brand answer should contain 'acme'")

	currency := result.Outputs[1][0]
	assert.Greater(t, currency.Score, float32(0), "currency answer score should be > 0")
	assert.Contains(t, contextJSON[currency.Start:currency.End], currency.Answer, "answer should be a substring of the context")
	assert.Contains(t, strings.ToLower(currency.Answer), "usd", "currency answer should contain 'usd'")

	// WithTopKAnswers(2): verify that 2 ranked answers are returned per input and are ordered by score.

	contextJSON = `{"product": "coffee maker", "brand": "Acme", "price": 49.99, "currency": "USD", "in_stock": true},{"product": "coffee maker", "brand": "Lavazza", "price": 100.99, "currency": "USD", "in_stock": true}`

	inputs = []pipelines.QuestionAnsweringInput{
		{Question: "What is the brand?", Context: contextJSON},
		{Question: "What is the currency?", Context: contextJSON},
	}
	configTopK := hugot.QuestionAnsweringConfig{
		ModelPath: modelPath,
		Name:      "testQAPipelineTopK",
		Options: []hugot.QuestionAnsweringOption{
			pipelines.WithTopKAnswers(2),
		},
	}
	pipelineTopK, err := session.NewPipeline(configTopK)
	CheckT(t, err)

	resultTopK, err := pipelineTopK.RunPipeline(t.Context(), inputs)
	CheckT(t, err)

	assert.Equal(t, 2, len(resultTopK.Outputs), "expected one result set per input")
	for inputIdx, answers := range resultTopK.Outputs {
		assert.Equal(t, 2, len(answers), "expected 2 answers per input with TopK=2")
		assert.GreaterOrEqual(t, answers[0].Score, answers[1].Score, "answers for input %d should be sorted by score descending", inputIdx)
		for answerIdx, answer := range answers {
			assert.Contains(t, contextJSON[answer.Start:answer.End], answer.Answer, "answer %d for input %d should be a substring of the context", answerIdx, inputIdx)
		}
	}
}

func QuestionAnsweringPipelineValidation(t *testing.T, session *hugot.Session) {
	t.Helper()

	modelPath := ModelsFolder + "/KnightsAnalytics_distilbert-onnx"

	config := hugot.QuestionAnsweringConfig{
		ModelPath: modelPath,
		Name:      "testQAPipeline",
	}
	pipeline, err := session.NewPipeline(config)
	CheckT(t, err)

	pipeline.MaxAnswerLength = 0
	assert.Error(t, pipeline.Validate(), "question answering validation should reject a non-positive MaxAnswerLength")
}

// TABULAR

func TabularPipeline(t *testing.T, session *hugot.Session) {
	t.Helper()
	config := backends.PipelineConfig[*pipelines.TabularPipeline]{
		ModelPath: ModelsFolder + "/KnightsAnalytics_iris-decision-tree",
		Name:      "testTabularClassification",
		Options: []backends.PipelineOption[*pipelines.TabularPipeline]{
			pipelines.WithIDLabelMap(map[int]string{
				0: "setosa",
				1: "versicolor",
				2: "virginica",
			}),
		},
	}

	pipeline, err := session.NewPipeline(config)
	CheckT(t, err)

	// Iris classification for an example
	inputs := []string{"[6.1, 2.8, 4.7, 1.2]"}
	result, err := pipeline.Run(t.Context(), inputs)
	CheckT(t, err)
	output := result.GetOutput()
	classification := output[0].(pipelines.TabularClassificationOutput)
	if classification.PredictedClass != "versicolor" {
		t.Errorf("Expected label 'versicolor', got '%s'", classification.PredictedClass)
	}
	for _, prob := range classification.Probabilities {
		if prob.Label == "versicolor" {
			if prob.Score < 0.97 {
				t.Errorf("Expected versicolor probability > 0.97, got '%f'", prob.Score)
			}
		}
	}
}

func TabularPipelineValidation(t *testing.T, session *hugot.Session) {
	t.Helper()
	config := backends.PipelineConfig[*pipelines.TabularPipeline]{
		ModelPath: ModelsFolder + "/KnightsAnalytics_iris-decision-tree",
		Name:      "testTabularClassification",
	}
	pipeline, err := session.NewPipeline(config)
	CheckT(t, err)

	original := pipeline.Model.InputsMeta[0].Dimensions
	defer func() { pipeline.Model.InputsMeta[0].Dimensions = original }()
	pipeline.Model.InputsMeta[0].Dimensions = backends.NewShape(-1)
	assert.Error(t, pipeline.Validate(), "tabular pipeline should require 2D input (batch, features)")
}

// Thread safety

func ThreadSafety(t *testing.T, session *hugot.Session, numEmbeddings int) {
	t.Helper()
	numWorkers := min(runtime.NumCPU(), 6)
	numResults := numWorkers * numEmbeddings

	t.Helper()
	modelPath := ModelsFolder + "/KnightsAnalytics_all-MiniLM-L6-v2"
	config := hugot.FeatureExtractionConfig{
		ModelPath:    modelPath,
		Name:         "testPipeline",
		OnnxFilename: "model.onnx",
	}
	pipeline, err := session.NewPipeline(config)
	CheckT(t, err)

	var expectedResults map[string][][]float32
	err = json.Unmarshal(embedded.ResultsByte, &expectedResults)
	CheckT(t, err)
	expectedResult1 := expectedResults["test1output"]
	expectedResult2 := expectedResults["test2output"]

	outputChannel1 := make(chan [][]float32, numResults)
	outputChannel2 := make(chan [][]float32, numResults)
	errChannel := make(chan error, numWorkers)

	worker := func() {
		for range numEmbeddings {
			batchResult, threadErr := pipeline.RunPipeline(t.Context(), []string{"robert smith"})
			if threadErr != nil {
				errChannel <- threadErr
			}
			outputChannel1 <- batchResult.Embeddings
			batchResult, threadErr = pipeline.RunPipeline(t.Context(), []string{"robert smith junior", "francis ford coppola"})
			if threadErr != nil {
				errChannel <- threadErr
			}
			outputChannel2 <- batchResult.Embeddings
		}
	}

	for range numWorkers {
		go worker()
	}

	correctResults1 := 0
	correctResults2 := 0
loop:
	for {
		if correctResults1 == numResults && correctResults2 == numResults {
			break loop
		}
		select {
		case vectors := <-outputChannel1:
			for i, vector := range vectors {
				e := floatsEqual(vector, expectedResult1[i])
				if e != nil {
					t.Logf("Test 1: The threaded neural network didn't produce the correct result: %s\n", e)
					t.FailNow()
				}
			}
			correctResults1++
		case vectors := <-outputChannel2:
			for i, vector := range vectors {
				e := floatsEqual(vector, expectedResult2[i])
				if e != nil {
					t.Logf("Test 2: The threaded neural network didn't produce the correct result: %s\n", e)
					t.FailNow()
				}
			}
			correctResults2++
		case threadErr := <-errChannel:
			t.Fatal(threadErr)
		}
	}
}

// Utilities

func skipGenerativePipeline(t *testing.T) {
	t.Helper()
	if os.Getenv("CI") != "" || strings.Contains(t.Name(), "Go") || strings.Contains(t.Name(), "XLA") {
		t.SkipNow()
	}
}

func checkClassificationOutput(t *testing.T, inputResult []pipelines.ClassificationOutput, inputExpected []pipelines.ClassificationOutput) {
	t.Helper()
	assert.Equal(t, len(inputResult), len(inputExpected))
	for i, output := range inputResult {
		resultExpected := inputExpected[i]
		assert.Equal(t, output.Label, resultExpected.Label)
		assert.True(t, almostEqual(float64(output.Score), float64(resultExpected.Score)), fmt.Sprintf("Expected %f, got %f", float64(output.Score), float64(resultExpected.Score)))
	}
}

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
		if diff >= 0.01 {
			return fmt.Errorf("data element %d doesn't match: %.12f vs %.12f",
				i, a[i], b[i])
		}
	}
	return nil
}

func almostEqual(a, b float64) bool {
	return math.Abs(a-b) <= 0.0007
}

func CheckT(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatalf("Test failed with error %s", err.Error())
	}
}

func logGenerationFailure(t *testing.T, name string, responses []string, err error) {
	t.Helper()
	if err != nil {
		t.Logf("%s failed: %v; partial responses: %q", name, err, responses)
	}
}

func printTokenEntities(o *pipelines.TokenClassificationOutput) {
	for i, entities := range o.Entities {
		fmt.Printf("Input %d\n", i)
		for _, entity := range entities {
			fmt.Printf("%+v\n", entity)
		}
	}
}

type toolCall struct {
	Arguments map[string]any `json:"arguments"`
	Name      string         `json:"name"`
}

// parseToolCalls extracts all <tool_call>...</tool_call> blocks from s and unmarshals
// each as a toolCall. Uses json.Decoder so that a trailing stray `}` (a common
// int4-quantisation artefact) does not cause the block to be skipped — Decode reads
// exactly one JSON value and stops, leaving trailing garbage unread.
func parseToolCalls(s string) []toolCall {
	re := regexp.MustCompile(`(?s)<tool_call>\s*(.+?)\s*</tool_call>`)
	matches := re.FindAllStringSubmatch(s, -1)
	var calls []toolCall
	for _, m := range matches {
		dec := json.NewDecoder(strings.NewReader(m[1]))
		var tc toolCall
		if err := dec.Decode(&tc); err != nil {
			continue
		}
		calls = append(calls, tc)
	}
	return calls
}

func CheckToolCalls(t *testing.T, output string) {
	t.Helper()
	calls := parseToolCalls(output)
	if len(calls) < 2 {
		t.Fatalf("expected at least 2 tool calls, got %d: %s", len(calls), output)
	}

	names := make(map[string]bool, len(calls))
	for _, c := range calls {
		names[c.Name] = true
	}
	if !names["get_current_time"] {
		t.Errorf("expected get_current_time tool call, got calls: %v", calls)
	}
	if !names["get_weather"] {
		t.Errorf("expected get_weather tool call, got calls: %v", calls)
	}
	for _, c := range calls {
		if c.Name == "get_weather" {
			city, _ := c.Arguments["city"].(string)
			if !strings.EqualFold(city, "Paris") {
				t.Errorf("expected get_weather city=Paris, got %q", city)
			}
		}
	}
}
