package hugot

import (
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/knights-analytics/hugot/pipelines"
	util "github.com/knights-analytics/hugot/utils"
)

//go:embed testData/tokenExpected.json
var tokenExpectedByte []byte

//go:embed testData/vectors.json
var resultsByte []byte

// use the system library for the tests
const onnxRuntimeSharedLibrary = "/usr/lib64/onnxruntime.so"

// test download validation

func TestDownloadValidation(t *testing.T) {
	err := validateDownloadHfModel("KnightsAnalytics/distilbert-base-uncased-finetuned-sst-2-english", "main", "")
	assert.NoError(t, err)
	// a model without tokenizer.json or .onnx model should error
	err = validateDownloadHfModel("ByteDance/SDXL-Lightning", "main", "")
	assert.Error(t, err)
}

// Text classification

func TestTextClassificationPipeline(t *testing.T) {
	session, err := NewSession(
		WithOnnxLibraryPath(onnxRuntimeSharedLibrary),
		WithTelemetry(),
		WithCpuMemArena(true),
		WithMemPattern(true),
		WithIntraOpNumThreads(1),
		WithInterOpNumThreads(1),
	)
	check(t, err)
	defer func(session *Session) {
		errDestroy := session.Destroy()
		check(t, errDestroy)
	}(session)
	modelPath := downloadModelIfNotExists(session, "KnightsAnalytics/distilbert-base-uncased-finetuned-sst-2-english", "./models")
	config := TextClassificationConfig{
		ModelPath: modelPath,
		Name:      "testPipelineSimple",
		Options: []TextClassificationOption{
			pipelines.WithSoftmax(),
		},
	}
	sentimentPipeline, err := NewPipeline(session, config)
	check(t, err)

	modelPathMulti := downloadModelIfNotExists(session, "SamLowe/roberta-base-go_emotions-onnx", "./models")
	configMulti := TextClassificationConfig{
		ModelPath:    modelPathMulti,
		Name:         "testPipelineSimpleMulti",
		OnnxFilename: "model.onnx",
		Options: []TextClassificationOption{
			pipelines.WithMultiLabel(),
			pipelines.WithSigmoid(),
		},
	}
	sentimentPipelineMulti, err := NewPipeline(session, configMulti)
	check(t, err)

	tests := []struct {
		pipeline *pipelines.TextClassificationPipeline
		name     string
		strings  []string
		expected pipelines.TextClassificationOutput
	}{
		{
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
		},
		{
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
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			batchResult, err := tt.pipeline.RunPipeline(tt.strings)
			check(t, err)
			for i, expected := range tt.expected.ClassificationOutputs {
				checkClassificationOutput(t, expected, batchResult.ClassificationOutputs[i])
			}
		})
	}

	// check get stats
	session.GetStats()
}

func TestTextClassificationPipelineValidation(t *testing.T) {
	session, err := NewSession(WithOnnxLibraryPath(onnxRuntimeSharedLibrary))
	check(t, err)
	defer func(session *Session) {
		err := session.Destroy()
		check(t, err)
	}(session)
	modelPath := downloadModelIfNotExists(session, "KnightsAnalytics/distilbert-base-uncased-finetuned-sst-2-english", "./models")

	config := TextClassificationConfig{
		ModelPath: modelPath,
		Name:      "testPipelineSimple",
		Options: []TextClassificationOption{
			pipelines.WithSingleLabel(),
		},
	}
	sentimentPipeline, err := NewPipeline(session, config)
	check(t, err)
	sentimentPipeline.IdLabelMap = map[int]string{}
	err = sentimentPipeline.Validate()
	assert.Error(t, err)
	if err != nil {
		errInt := err.(interface{ Unwrap() []error })
		assert.Equal(t, 3, len(errInt.Unwrap()))
	}
	sentimentPipeline.OutputDim = 0
	err = sentimentPipeline.Validate()
	assert.Error(t, err)
	if err != nil {
		errInt := err.(interface{ Unwrap() []error })
		assert.Equal(t, 3, len(errInt.Unwrap()))
	}
}

// Token classification

func TestTokenClassificationPipeline(t *testing.T) {
	session, err := NewSession(WithOnnxLibraryPath(onnxRuntimeSharedLibrary))
	check(t, err)
	defer func(session *Session) {
		err := session.Destroy()
		check(t, err)
	}(session)

	modelPath := downloadModelIfNotExists(session, "KnightsAnalytics/distilbert-NER", "./models")
	configSimple := TokenClassificationConfig{
		ModelPath: modelPath,
		Name:      "testPipelineSimple",
		Options: []TokenClassificationOption{
			pipelines.WithSimpleAggregation(),
			pipelines.WithIgnoreLabels([]string{"O"}),
		},
	}
	pipelineSimple, err2 := NewPipeline(session, configSimple)
	check(t, err2)

	configNone := TokenClassificationConfig{
		ModelPath: modelPath,
		Name:      "testPipelineNone",
		Options: []TokenClassificationOption{
			pipelines.WithoutAggregation(),
		},
	}
	pipelineNone, err3 := NewPipeline(session, configNone)
	check(t, err3)

	var expectedResults map[int]pipelines.TokenClassificationOutput
	err4 := json.Unmarshal(tokenExpectedByte, &expectedResults)
	check(t, err4)

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
			check(t, err)
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

func TestTokenClassificationPipelineValidation(t *testing.T) {
	session, err := NewSession(WithOnnxLibraryPath(onnxRuntimeSharedLibrary))
	check(t, err)
	defer func(session *Session) {
		err := session.Destroy()
		check(t, err)
	}(session)

	modelPath := downloadModelIfNotExists(session, "KnightsAnalytics/distilbert-NER", "./models")
	configSimple := TokenClassificationConfig{
		ModelPath: modelPath,
		Name:      "testPipelineSimple",
		Options: []TokenClassificationOption{
			pipelines.WithSimpleAggregation(),
			pipelines.WithIgnoreLabels([]string{"O"}),
		},
	}
	pipelineSimple, err2 := NewPipeline(session, configSimple)
	check(t, err2)

	pipelineSimple.IdLabelMap = map[int]string{}
	err = pipelineSimple.Validate()
	assert.Error(t, err)
	if err != nil {
		errInt := err.(interface{ Unwrap() []error })
		assert.Equal(t, 2, len(errInt.Unwrap()))
	}
	pipelineSimple.OutputDim = 0
	err = pipelineSimple.Validate()
	assert.Error(t, err)
	if err != nil {
		errInt := err.(interface{ Unwrap() []error })
		assert.Equal(t, 2, len(errInt.Unwrap()))
	}
}

func TestNoSameNamePipeline(t *testing.T) {
	session, err := NewSession(WithOnnxLibraryPath(onnxRuntimeSharedLibrary))
	check(t, err)
	defer func(session *Session) {
		err := session.Destroy()
		check(t, err)
	}(session)

	modelPath := downloadModelIfNotExists(session, "KnightsAnalytics/distilbert-NER", "./models")
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

// feature extraction

func TestFeatureExtractionPipeline(t *testing.T) {
	session, err := NewSession(WithOnnxLibraryPath(onnxRuntimeSharedLibrary))
	check(t, err)
	defer func(session *Session) {
		err := session.Destroy()
		check(t, err)
	}(session)

	modelPath := downloadModelIfNotExists(session, "KnightsAnalytics/all-MiniLM-L6-v2", "./models")

	config := FeatureExtractionConfig{
		ModelPath: modelPath,
		Name:      "testPipeline",
	}
	pipeline, err := NewPipeline(session, config)
	check(t, err)

	var expectedResults map[string][][]float32
	err = json.Unmarshal(resultsByte, &expectedResults)
	check(t, err)
	var testResults [][]float32

	// test 'robert smith'
	testResults = expectedResults["test1output"]
	for i := 1; i <= 10; i++ {
		batchResult, err := pipeline.RunPipeline([]string{"robert smith"})
		check(t, err)
		e := floatsEqual(batchResult.Embeddings[0], testResults[0])
		if e != nil {
			t.Logf("Test 1: The neural network didn't produce the correct result on loop %d: %s\n", i, e)
			t.FailNow()
		}
	}

	// test ['robert smith junior', 'francis ford coppola']
	testResults = expectedResults["test2output"]
	for i := 1; i <= 10; i++ {
		batchResult, err := pipeline.RunPipeline([]string{"robert smith junior", "francis ford coppola"})
		check(t, err)
		for j, res := range batchResult.Embeddings {
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
		firstBatchResult, err2 := pipeline.RunPipeline(sentencePair[0])
		check(t, err2)
		firstEmbedding := firstBatchResult.Embeddings[0]

		secondBatchResult, err3 := pipeline.RunPipeline(sentencePair[1])
		check(t, err3)
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
	assert.Greater(t, pipeline.TokenizerTimings.NumCalls, zero, "TokenizerTimings.NumCalls should be greater than 0")
	assert.Greater(t, pipeline.TokenizerTimings.TotalNS, zero, "TokenizerTimings.TotalNS should be greater than 0")

	// test normalization
	testResults = expectedResults["normalizedOutput"]
	config = FeatureExtractionConfig{
		ModelPath: modelPath,
		Name:      "testPipelineNormalise",
		Options: []FeatureExtractionOption{
			pipelines.WithNormalization(),
		},
	}
	pipeline, err = NewPipeline(session, config)
	check(t, err)
	normalizationStrings := []string{"Onnxruntime is a great inference backend"}
	normalizedEmbedding, err := pipeline.RunPipeline(normalizationStrings)
	check(t, err)
	for i, embedding := range normalizedEmbedding.Embeddings {
		e := floatsEqual(embedding, testResults[i])
		if e != nil {
			t.Fatalf("Normalization test failed: %s", normalizationStrings[i])
		}
	}
}

func TestFeatureExtractionPipelineValidation(t *testing.T) {
	session, err := NewSession(WithOnnxLibraryPath(onnxRuntimeSharedLibrary))
	check(t, err)
	defer func(session *Session) {
		err := session.Destroy()
		check(t, err)
	}(session)

	modelPath := downloadModelIfNotExists(session, "KnightsAnalytics/all-MiniLM-L6-v2", "./models")
	config := FeatureExtractionConfig{
		ModelPath: modelPath,
		Name:      "testPipeline",
	}
	pipeline, err := NewPipeline(session, config)
	check(t, err)

	pipeline.OutputDim = 0
	err = pipeline.Validate()
	assert.Error(t, err)
}

// README: test the readme examples

func TestReadmeExample(t *testing.T) {
	check := func(err error) {
		if err != nil {
			panic(err.Error())
		}
	}

	// start a new session. This looks for the onnxruntime.so library in its default path, e.g. /usr/lib/onnxruntime.so
	session, err := NewSession()
	// if your onnxruntime.so is somewhere else, you can explicitly set it by using WithOnnxLibraryPath
	// session, err := hugot.NewSession(WithOnnxLibraryPath("/path/to/onnxruntime.so"))
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
	modelPath, err := session.DownloadModel("KnightsAnalytics/distilbert-base-uncased-finetuned-sst-2-english", "./", NewDownloadOptions())
	check(err)

	defer func(modelPath string) {
		err := util.FileSystem.Delete(context.Background(), modelPath)
		if err != nil {
			t.FailNow()
		}
	}(modelPath)

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
	// {"ClassificationOutputs":[[{"Label":"POSITIVE","Score":0.9998536}],[{"Label":"NEGATIVE","Score":0.99752176}]]}
}

func TestCuda(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	opts := []WithOption{
		WithOnnxLibraryPath("/usr/lib64/onnxruntime-gpu/libonnxruntime.so"),
		WithCuda(map[string]string{
			"device_id": "0",
		}),
	}

	session, err := NewSession(opts...)
	check(t, err)

	defer func(session *Session) {
		errDestroy := session.Destroy()
		if errDestroy != nil {
			panic(errDestroy)
		}
	}(session)

	modelPath := downloadModelIfNotExists(session, "KnightsAnalytics/all-MiniLM-L6-v2", "./models")
	config := FeatureExtractionConfig{
		ModelPath: modelPath,
		Name:      "benchmarkEmbedding",
	}
	pipelineEmbedder, err2 := NewPipeline(session, config)
	check(t, err2)
	res, err := pipelineEmbedder.Run([]string{"Test with cuda", "Test with cuda 1"})
	check(t, err)
	fmt.Println(res.GetOutput())
}

// Benchmarks

func runBenchmarkEmbedding(strings *[]string, cuda bool) {
	var opts []WithOption
	switch cuda {
	case true:
		opts = []WithOption{
			WithOnnxLibraryPath("/usr/lib64/onnxruntime-gpu/libonnxruntime.so"),
			WithCuda(map[string]string{
				"device_id": "0",
			}),
		}
	default:
		opts = []WithOption{WithOnnxLibraryPath("/usr/lib64/onnxruntime.so")}
	}
	session, err := NewSession(opts...)
	if err != nil {
		panic(err)
	}

	defer func(session *Session) {
		errDestroy := session.Destroy()
		if errDestroy != nil {
			panic(errDestroy)
		}
	}(session)

	modelPath := downloadModelIfNotExists(session, "KnightsAnalytics/all-MiniLM-L6-v2", "./models")
	config := FeatureExtractionConfig{
		ModelPath: modelPath,
		Name:      "benchmarkEmbedding",
	}
	pipelineEmbedder, err2 := NewPipeline(session, config)
	if err2 != nil {
		panic(err2)
	}
	res, err := pipelineEmbedder.Run(*strings)
	if err != nil {
		panic(err)
	}
	fmt.Println(len(res.GetOutput()))
}

func BenchmarkCudaEmbedding(b *testing.B) {
	if os.Getenv("CI") != "" {
		b.SkipNow()
	}
	p := make([]string, 30000)
	for i := 0; i < 30000; i++ {
		p[i] = "The goal of this library is to provide an easy, scalable, and hassle-free way to run huggingface transformer pipelines in golang applications."
	}
	for i := 0; i < b.N; i++ {
		runBenchmarkEmbedding(&p, true)
	}
}

func BenchmarkCPUEmbedding(b *testing.B) {
	if os.Getenv("CI") != "" {
		b.SkipNow()
	}
	p := make([]string, 30000)
	for i := 0; i < 30000; i++ {
		p[i] = "The goal of this library is to provide an easy, scalable, and hassle-free way to run huggingface transformer pipelines in golang applications."
	}
	for i := 0; i < b.N; i++ {
		runBenchmarkEmbedding(&p, false)
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
	t.Helper()
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

func check(t *testing.T, err error) {
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
