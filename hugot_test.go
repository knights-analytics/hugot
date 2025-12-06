package hugot

import (
	"encoding/json"
	"fmt"
	"math"
	"runtime"
	"slices"
	"strings"
	"testing"

	"github.com/gomlx/go-huggingface/hub"
	"github.com/stretchr/testify/assert"

	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/pipelines"
	"github.com/knights-analytics/hugot/testcases/embedded"
	"github.com/knights-analytics/hugot/util/imageutil"
)

// use the system library for the tests.
const onnxRuntimeSharedLibrary = "/usr/lib64/onnxruntime.so"

// test download validation

func TestDownloadValidation(t *testing.T) {
	downloadOptions := NewDownloadOptions()

	// a model with the required files in a subfolder should not error
	_, err := validateDownloadHfModel(hub.New("KnightsAnalytics/distilbert-base-uncased-finetuned-sst-2-english"), downloadOptions)
	assert.NoError(t, err)
	// a model without tokenizer.json or .onnx model should error
	_, err = validateDownloadHfModel(hub.New("ByteDance/SDXL-Lightning"), downloadOptions)
	assert.Error(t, err)
}

// FEATURE EXTRACTION

func featureExtractionPipeline(t *testing.T, session *Session) {
	t.Helper()

	modelPath := "./models/KnightsAnalytics_all-MiniLM-L6-v2"

	config := FeatureExtractionConfig{
		ModelPath:    modelPath,
		Name:         "testPipeline",
		OnnxFilename: "model.onnx",
	}
	pipeline, err := NewPipeline(session, config)
	checkT(t, err)

	var expectedResults map[string][][]float32
	err = json.Unmarshal(embedded.ResultsByte, &expectedResults)
	checkT(t, err)
	var testResults [][]float32

	// test 'robert smith'
	testResults = expectedResults["test1output"]
	batchResult, err := pipeline.RunPipeline([]string{"robert smith"})
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
	batchResult, err = pipeline.RunPipeline([]string{"robert smith junior", "francis ford coppola"})
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
		firstBatchResult, err2 := pipeline.RunPipeline(sentencePair[0])
		checkT(t, err2)
		firstEmbedding := firstBatchResult.Embeddings[0]

		secondBatchResult, err3 := pipeline.RunPipeline(sentencePair[1])
		checkT(t, err3)
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
	assert.Greater(t, pipeline.PipelineTimings.NumCalls, zero, "PipelineTimings.NumCalls should be greater than 0")
	assert.Greater(t, pipeline.PipelineTimings.TotalNS, zero, "PipelineTimings.TotalNS should be greater than 0")
	assert.Greater(t, pipeline.Model.Tokenizer.TokenizerTimings.NumCalls, zero, "TokenizerTimings.NumCalls should be greater than 0")
	assert.Greater(t, pipeline.Model.Tokenizer.TokenizerTimings.TotalNS, zero, "TokenizerTimings.TotalNS should be greater than 0")

	// test normalization
	testResults = expectedResults["normalizedOutput"]
	config = FeatureExtractionConfig{
		ModelPath:    modelPath,
		Name:         "testPipelineNormalise",
		OnnxFilename: "model.onnx",
		Options: []FeatureExtractionOption{
			pipelines.WithNormalization(),
		},
	}
	pipeline, err = NewPipeline(session, config)
	checkT(t, err)

	normalizationStrings := []string{"Onnxruntime is a great inference backend"}
	normalizedEmbedding, err := pipeline.RunPipeline(normalizationStrings)
	checkT(t, err)
	for i, embedding := range normalizedEmbedding.Embeddings {
		e := floatsEqual(embedding, testResults[i])
		if e != nil {
			t.Fatalf("Normalization test failed: %s", normalizationStrings[i])
		}
	}

	// test getting output by name
	configSentence := FeatureExtractionConfig{
		ModelPath:    modelPath,
		Name:         "testPipelineSentence",
		OnnxFilename: "model.onnx",
		Options:      []FeatureExtractionOption{pipelines.WithOutputName("last_hidden_state")},
	}
	pipelineSentence, err := NewPipeline(session, configSentence)
	checkT(t, err)

	_, err = pipelineSentence.RunPipeline([]string{"Onnxruntime is a great inference backend"})
	if err != nil {
		t.FailNow()
	}
	configSentence = FeatureExtractionConfig{
		ModelPath:    modelPath,
		Name:         "testPipelineToken",
		OnnxFilename: "model.onnx",
	}
	pipelineToken, err := NewPipeline(session, configSentence)
	checkT(t, err)
	_, err = pipelineToken.RunPipeline([]string{"Onnxruntime is a great inference backend"})
	if err != nil {
		t.FailNow()
	}
}

func featureExtractionPipelineValidation(t *testing.T, session *Session) {
	t.Helper()

	modelPath := "./models/KnightsAnalytics_all-MiniLM-L6-v2"
	config := FeatureExtractionConfig{
		ModelPath:    modelPath,
		OnnxFilename: "model.onnx",
		Name:         "testPipeline",
	}
	pipeline, err := NewPipeline(session, config)
	checkT(t, err)

	pipeline.Model.InputsMeta[0].Dimensions = backends.NewShape(-1, -1, -1)

	err = pipeline.Validate()
	assert.Error(t, err)

	pipeline.Model.InputsMeta[0].Dimensions = backends.NewShape(1, 1, 1, 1)
	err = pipeline.Validate()
	assert.Error(t, err)
}

// Text classification

func textClassificationPipeline(t *testing.T, session *Session) {
	t.Helper()

	modelPath := "./models/KnightsAnalytics_distilbert-base-uncased-finetuned-sst-2-english"

	config := TextClassificationConfig{
		ModelPath: modelPath,
		Name:      "testPipelineSimple",
		Options: []TextClassificationOption{
			pipelines.WithSoftmax(),
		},
	}
	sentimentPipeline, err := NewPipeline(session, config)
	checkT(t, err)

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
		batchResult, err := test.pipeline.RunPipeline(test.strings)
		checkT(t, err)
		for i, expected := range test.expected.ClassificationOutputs {
			checkClassificationOutput(t, expected, batchResult.ClassificationOutputs[i])
		}
	})

	// check PrintStatistics
	session.PrintStatistics()
}

func textClassificationPipelineMulti(t *testing.T, session *Session) {
	t.Helper()

	modelPathMulti := "./models/KnightsAnalytics_roberta-base-go_emotions"

	configMulti := TextClassificationConfig{
		ModelPath:    modelPathMulti,
		Name:         "testPipelineSimpleMulti",
		OnnxFilename: "model.onnx",
		Options: []TextClassificationOption{
			pipelines.WithMultiLabel(),
			pipelines.WithSigmoid(),
			pipelines.WithFixedPadding(128),
		},
	}
	sentimentPipelineMulti, err := NewPipeline(session, configMulti)
	checkT(t, err)

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
		batchResult, err := test.pipeline.RunPipeline(test.strings)
		checkT(t, err)
		for i, expected := range test.expected.ClassificationOutputs {
			checkClassificationOutput(t, expected, batchResult.ClassificationOutputs[i])
		}
	})

	// check GetStatistics
	statistics := session.GetStatistics()
	for _, v := range statistics {
		v.Print()
	}
}

func textClassificationPipelineValidation(t *testing.T, session *Session) {
	t.Helper()

	modelPath := "./models/KnightsAnalytics_distilbert-base-uncased-finetuned-sst-2-english"

	config := TextClassificationConfig{
		ModelPath: modelPath,
		Name:      "testPipelineSimple",
		Options: []TextClassificationOption{
			pipelines.WithSingleLabel(),
		},
	}
	sentimentPipeline, err := NewPipeline(session, config)
	checkT(t, err)

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

func zeroShotClassificationPipeline(t *testing.T, session *Session) {
	t.Helper()

	modelPath := "./models/KnightsAnalytics_deberta-v3-base-zeroshot-v1"

	config := ZeroShotClassificationConfig{
		ModelPath: modelPath,
		Name:      "testPipeline",
		Options: []backends.PipelineOption[*pipelines.ZeroShotClassificationPipeline]{
			pipelines.WithHypothesisTemplate("This example is {}."),
			pipelines.WithLabels([]string{"fun", "dangerous"}),
			pipelines.WithMultilabel(false), // Gets overridden per test, but included for coverage
		},
	}

	classificationPipeline, err := NewPipeline(session, config)
	checkT(t, err)

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
			batchResult, err := tt.pipeline.RunPipeline(tt.sequences)
			checkT(t, err)
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

func zeroShotClassificationPipelineValidation(t *testing.T, session *Session) {
	t.Helper()

	modelPath := "./models/KnightsAnalytics_deberta-v3-base-zeroshot-v1"

	config := TextClassificationConfig{
		ModelPath: modelPath,
		Name:      "testPipelineSimple",
	}
	sentimentPipeline, err := NewPipeline(session, config)
	checkT(t, err)

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

// Token classification

func tokenClassificationPipeline(t *testing.T, session *Session) {
	t.Helper()

	modelPath := "./models/KnightsAnalytics_distilbert-NER"
	configSimple := TokenClassificationConfig{
		ModelPath: modelPath,
		Name:      "testPipelineSimple",
		Options: []TokenClassificationOption{
			pipelines.WithSimpleAggregation(),
			pipelines.WithIgnoreLabels([]string{"O"}),
		},
	}
	pipelineSimple, err2 := NewPipeline(session, configSimple)
	checkT(t, err2)

	configNone := TokenClassificationConfig{
		ModelPath: modelPath,
		Name:      "testPipelineNone",
		Options: []TokenClassificationOption{
			pipelines.WithoutAggregation(),
		},
	}
	pipelineNone, err3 := NewPipeline(session, configNone)
	checkT(t, err3)

	var expectedResults map[int]pipelines.TokenClassificationOutput
	err4 := json.Unmarshal(embedded.TokenExpectedByte, &expectedResults)
	checkT(t, err4)

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
			batchResult, err := tt.pipeline.RunPipeline(tt.strings)
			checkT(t, err)
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
}

func tokenClassificationPipelineValidation(t *testing.T, session *Session) {
	t.Helper()

	modelPath := "./models/KnightsAnalytics_distilbert-NER"
	configSimple := TokenClassificationConfig{
		ModelPath: modelPath,
		Name:      "testPipelineSimple",
		Options: []TokenClassificationOption{
			pipelines.WithSimpleAggregation(),
			pipelines.WithIgnoreLabels([]string{"O"}),
		},
	}
	pipelineSimple, err2 := NewPipeline(session, configSimple)
	checkT(t, err2)

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

func crossEncoderPipeline(t *testing.T, session *Session) {
	t.Helper()
	config := CrossEncoderConfig{
		ModelPath: "./models/KnightsAnalytics_jina-reranker-v1-tiny-en",
		Name:      "test-cross-encoder",
	}
	pipeline, err := NewPipeline(session, config)
	checkT(t, err)

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
	output, err := pipeline.Run(inputs)
	checkT(t, err)
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

func crossEncoderPipelineValidation(t *testing.T, session *Session) {
	t.Helper()
	config := CrossEncoderConfig{
		ModelPath: "./models/KnightsAnalytics_jina-reranker-v1-tiny-en",
		Name:      "test-cross-encoder-validation",
	}
	pipeline, err := NewPipeline(session, config)
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

// Image classification test using HuggingFace SqueezeNet and a sample image.
func imageClassificationPipeline(t *testing.T, session *Session) {
	t.Helper()

	modelPath := "./models/KnightsAnalytics_resnet50"
	imagePath := "./models/imageData/cat.jpg"

	config := ImageClassificationConfig{
		ModelPath:    modelPath,
		Name:         "testImageClassification",
		OnnxFilename: "squeezenet1.1.onnx",
		Options: []ImageClassificationOption{
			pipelines.WithTopK(3),
			pipelines.WithPreprocessSteps(
				imageutil.ResizeStep(224),
				imageutil.CenterCropStep(224, 224),
			),
			pipelines.WithNormalizationSteps(
				imageutil.ImagenetPixelNormalizationStep(),
				imageutil.RescaleStep(),
			),
		},
	}
	pipeline, err := NewPipeline(session, config)
	checkT(t, err)

	result, err := pipeline.RunPipeline([]string{imagePath, imagePath})
	checkT(t, err)

	for i, pred := range result.Predictions[0] {
		fmt.Printf("%d: %s (score: %.4f)\n", i+1, pred.Label, pred.Score)
	}
	if result.Predictions[0][0].Label != "tabby, tabby cat" {
		t.Errorf("Expected label 'tabby, tabby cat', got '%s'", result.Predictions[0][0].Label)
	}
}

func imageClassificationPipelineValidation(t *testing.T, session *Session) {
	t.Helper()

	modelPath := "./models/KnightsAnalytics_resnet50"
	config := ImageClassificationConfig{
		ModelPath: modelPath,
		Name:      "testImageClassification",
	}
	pipeline, err := NewPipeline(session, config)
	checkT(t, err)

	pipeline.Model.InputsMeta[0].Dimensions = backends.NewShape(-1, -1, -1)

	err = pipeline.Validate()
	assert.Error(t, err)
}

// No same name

func noSameNamePipeline(t *testing.T, session *Session) {
	t.Helper()
	modelPath := "./models/KnightsAnalytics_distilbert-NER"
	configSimple := TokenClassificationConfig{
		ModelPath: modelPath,
		Name:      "testPipelineSimple",
		Options: []TokenClassificationOption{
			pipelines.WithSimpleAggregation(),
			pipelines.WithIgnoreLabels([]string{"O"}),
		},
	}
	_, err2 := NewPipeline(session, configSimple)
	if err2 != nil {
		t.FailNow()
	}
	_, err3 := NewPipeline(session, configSimple)
	assert.Error(t, err3)
}

func destroyPipelines(t *testing.T, session *Session) {
	t.Helper()

	modelPath := "./models/KnightsAnalytics_distilbert-NER"
	configSimple := TokenClassificationConfig{
		ModelPath: modelPath,
		Name:      "testClosePipeline",
		Options: []TokenClassificationOption{
			pipelines.WithSimpleAggregation(),
			pipelines.WithIgnoreLabels([]string{"O"}),
		},
	}
	_, err := NewPipeline(session, configSimple)
	checkT(t, err)

	if len(session.models) != 1 {
		t.Fatal("Session should have 1 model")
	}

	for _, model := range session.models {
		if _, ok := model.Pipelines["testClosePipeline"]; !ok {
			t.Fatal("Pipeline alias was not added to the model")
		}
	}

	if err := ClosePipeline[*pipelines.TokenClassificationPipeline](session, "testClosePipeline"); err != nil {
		t.Fatal(err)
	}

	if len(session.models) != 0 {
		t.Fatal("Session should have 0 models")
	}
	if len(session.tokenClassificationPipelines) != 0 {
		t.Fatal("Session should have 0 token classification pipelines")
	}
}

// Text Generation.
func textGenerationPipeline(t *testing.T, session *Session) {
	t.Helper()

	defer func(session *Session) {
		err := session.Destroy()
		checkT(t, err)
	}(session)

	// Configure the text generation pipeline
	config := TextGenerationConfig{
		ModelPath:    "./models/KnightsAnalytics_Phi-3.5-mini-instruct-onnx",
		Name:         "testPipeline",
		OnnxFilename: "model.onnx",
		Options: []backends.PipelineOption[*pipelines.TextGenerationPipeline]{
			pipelines.WithMaxTokens(200),
			pipelines.WithPhiTemplate(),
			pipelines.WithCustomStopTokens([]int64{32007}), // Phi appears to use a stop token that's different from the one defined in config.json
		},
	}

	// Create the pipeline
	textGenPipeline, err := NewPipeline(session, config)
	checkT(t, err)

	tests := []struct {
		name           string
		input          [][]pipelines.Message
		expectedString []string
	}{
		{
			name: "small test",
			input: [][]pipelines.Message{
				{
					{Role: "system", Content: "you are a helpful assistant."},
					{Role: "user", Content: "what is the capital of the Netherlands?"},
				},
			},
			expectedString: []string{
				"The capital of the Netherlands is Amsterdam. However, it's worth noting that the seat of government is actually located in The Hague, where the royal family and the supreme court are based. Amsterdam is the country's largest city and serves as the political, economic, and cultural center of the Netherlands.",
			},
		},
		{
			name: "batched input, short sequence first, long sequence second",
			input: [][]pipelines.Message{
				{
					{Role: "system", Content: "you are a helpful assistant."},
					{Role: "user", Content: "what is the capital of the Netherlands?"},
				},
				{
					{Role: "system", Content: "you are a helpful assistant."},
					{Role: "user", Content: "who was the first president of the United States?"},
				},
			},
			expectedString: []string{
				"The capital of the Netherlands is Amsterdam. It is the largest city in the country and serves as the political, economic, and cultural center. Amsterdam is known for its historical significance, vibrant cultural scene, and iconic landmarks such as the Anne Frank House and the Van Gogh Museum. The city is also famous for its extensive canal system, which is a UNESCO World Heritage site. The Dutch government's administrative buildings, including the Royal Palace, are located in the historic center of the city. Amsterdam's port is one of the busiest in Europe, contributing to its status as a major global financial hub.",
				"The first president of the United States was George Washington. He served as the president from April 30, 1789, to March 4, 1797. Washington is often referred to as the \"Father of His Country\" for his leadership in the founding of the United States. He was a central figure in the American Revolution and presided over the convention that established the new government, which resulted in the creation of the Constitution. Washington's presidency set many precedents, including the two-term limit, which was later codified in the 22nd Amendment. His leadership and the establishment of a strong federal government were crucial in the early years of the United States.",
			},
		},
		{
			name: "batched input, long sequence first, short sequence second",
			input: [][]pipelines.Message{
				{
					{Role: "system", Content: "you are a helpful assistant."},
					{Role: "user", Content: "Solve this equation: 2 + 2 = ? Be very brief in your explanation."},
				},
				{
					{Role: "system", Content: "you are a helpful assistant."},
					{Role: "user", Content: "Answer in one sentence: what is steel made out of?"},
				},
			},
			expectedString: []string{
				"The sum of 2 and 2 is 4.\n\nExplanation: In arithmetic, when you add two identical positive numbers, you simply double the number. Here, doubling 2 gives you 4.",
				"Steel is primarily made out of iron and carbon, with the addition of other elements such as manganese, chromium, nickel, and sometimes small amounts of other elements to enhance its properties.",
			},
		},
	}

	// Execute tests
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			batchResult, err := textGenPipeline.RunWithTemplate(tt.input)
			checkT(t, err)
			outputString := batchResult.GetOutput()
			for i := range len(outputString) {
				expectedString := tt.expectedString[i]
				generatedString := outputString[i].(string)
				// Generative models have variance with different architectures which builds up per generation, so check for min overlap instead in first 30 words
				expectedWords := strings.Fields(expectedString)
				if len(expectedWords) > 30 {
					expectedWords = expectedWords[:30]
				}
				generatedWords := strings.Fields(generatedString)
				if len(generatedWords) > 30 {
					generatedWords = generatedWords[:30]
				}
				numMismatches := 0
				for _, word := range generatedWords {
					if !slices.Contains(expectedWords, word) {
						numMismatches++
					}
				}
				assert.LessOrEqual(t, numMismatches, len(expectedWords)/8, "Expected generated to have max %d mismatches with expected, but got %d mismatches (got %s wanted %s)", len(expectedWords)/8, numMismatches, expectedString, generatedString)
			}
		})
	}
}

func textGenerationPipelineValidation(t *testing.T, session *Session) {
	t.Helper()

	defer func(session *Session) {
		err := session.Destroy()
		checkT(t, err)
	}(session)

	// Configure the text generation pipeline
	config := TextGenerationConfig{
		ModelPath:    "./models/KnightsAnalytics_Phi-3.5-mini-instruct-onnx",
		Name:         "testPipeline",
		OnnxFilename: "model.onnx",
		Options: []backends.PipelineOption[*pipelines.TextGenerationPipeline]{
			pipelines.WithMaxTokens(1),
		},
	}

	// Create the pipeline
	pipeline, err := NewPipeline(session, config)
	checkT(t, err)

	pipeline.Model.NumHiddenLayers = 0
	err = pipeline.Validate()
	assert.Error(t, err)
	pipeline.Model.NumHiddenLayers = 1

	pipeline.Model.NumKeyValueHeads = 0
	err = pipeline.Validate()
	assert.Error(t, err)
	pipeline.Model.NumKeyValueHeads = 1

	pipeline.Model.HeadDim = 0
	err = pipeline.Validate()
	assert.Error(t, err)
	pipeline.Model.HeadDim = 1
}

// Thread safety

func threadSafety(t *testing.T, session *Session, numEmbeddings int) {
	t.Helper()
	numWorkers := min(runtime.NumCPU(), 6)
	numResults := numWorkers * numEmbeddings

	t.Helper()
	modelPath := "./models/KnightsAnalytics_all-MiniLM-L6-v2"
	config := FeatureExtractionConfig{
		ModelPath:    modelPath,
		Name:         "testPipeline",
		OnnxFilename: "model.onnx",
	}
	pipeline, err := NewPipeline(session, config)
	checkT(t, err)

	var expectedResults map[string][][]float32
	err = json.Unmarshal(embedded.ResultsByte, &expectedResults)
	checkT(t, err)
	expectedResult1 := expectedResults["test1output"]
	expectedResult2 := expectedResults["test2output"]

	outputChannel1 := make(chan [][]float32, numResults)
	outputChannel2 := make(chan [][]float32, numResults)
	errChannel := make(chan error, numWorkers)

	worker := func() {
		for range numEmbeddings {
			batchResult, threadErr := pipeline.RunPipeline([]string{"robert smith"})
			if threadErr != nil {
				errChannel <- threadErr
			}
			outputChannel1 <- batchResult.Embeddings
			batchResult, threadErr = pipeline.RunPipeline([]string{"robert smith junior", "francis ford coppola"})
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

func checkT(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatalf("Test failed with error %s", err.Error())
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
