//go:build (GO || ALL) && !TRAINING

package hugot

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/pipelines"
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

func TestZeroShotClassificationPipelineGo(t *testing.T) {
	session, err := NewGoSession()
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	zeroShotClassificationPipeline(t, session)
}

func TestZeroShotClassificationPipelineValidationGo(t *testing.T) {
	session, err := NewGoSession()
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	zeroShotClassificationPipelineValidation(t, session)
}

// Cross Encoder

func TestCrossEncoderPipelineGo(t *testing.T) {
	session, err := NewGoSession()
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	crossEncoderPipeline(t, session)
}

func TestCrossEncoderPipelineValidationGo(t *testing.T) {
	session, err := NewGoSession()
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	crossEncoderPipelineValidation(t, session)
}

// Image classification

func TestImageClassificationPipelineGo(t *testing.T) {
	session, err := NewGoSession()
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	imageClassificationPipeline(t, session)
}

func TestImageClassificationPipelineValidationGo(t *testing.T) {
	session, err := NewGoSession()
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	imageClassificationPipelineValidation(t, session)
}

// Object detection

func TestObjectDetectionPipelineGo(t *testing.T) {
	t.Skip("Currently fails due to unsupported constant in XLA backend")
	session, err := NewGoSession()
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	objectDetectionPipeline(t, session)
}

func TestObjectDetectionPipelineValidationGo(t *testing.T) {
	t.Skip("Currently fails due to unsupported constant in XLA backend")
	session, err := NewGoSession()
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	objectDetectionPipelineValidation(t, session)
}

// text generation

func TestTextGenerationPipelineGo(t *testing.T) {
	t.Skip("Generative models are not supported yet for Go")
	session, err := NewGoSession()
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	textGenerationPipeline(t, session)
}

func TestTextGenerationPipelineValidationGo(t *testing.T) {
	t.Skip("Generative models are not supported yet for Go")
	session, err := NewGoSession()
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	textGenerationPipelineValidation(t, session)
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
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	session, err := NewGoSession()
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	threadSafety(t, session, 20)
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

// ============================================================================
// Seq2Seq Pipeline Tests (Go Runtime)
// ============================================================================

func TestSeq2SeqPipelineGo(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow() // Skip in CI - requires model download
	}
	session, err := NewGoSession()
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	seq2seqPipelineGo(t, session)
}

func TestSeq2SeqPipelineValidationGo(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	session, err := NewGoSession()
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	seq2seqPipelineValidationGo(t, session)
}

func seq2seqPipelineGo(t *testing.T, session *Session) {
	t.Helper()

	// Model path - expects a T5/doc2query model exported with fastT5
	modelPath := "./models/doc2query-t5-small"

	// Check if model exists
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skipf("Model not found at %s. Run export_t5_onnx.py to create it.", modelPath)
	}

	// Test 1: Basic greedy decoding
	t.Run("GreedyDecoding", func(t *testing.T) {
		config := Seq2SeqConfig{
			ModelPath: modelPath,
			Name:      "testSeq2SeqGreedyGo",
			Options: []backends.PipelineOption[*pipelines.Seq2SeqPipeline]{
				pipelines.WithSeq2SeqMaxTokens(32),
			},
		}
		pipeline, err := NewPipeline(session, config)
		checkT(t, err)

		// Test with a simple input
		inputs := []string{"Python is an interpreted, high-level programming language."}
		output, err := pipeline.RunPipeline(inputs)
		checkT(t, err)

		// Verify output structure
		assert.Len(t, output.GeneratedTexts, 1, "Should have 1 output for 1 input")
		assert.Len(t, output.GeneratedTexts[0], 1, "Should have 1 sequence per input (greedy)")

		// Output should be non-empty and reasonable
		generatedText := output.GeneratedTexts[0][0]
		t.Logf("Generated query (Go): %s", generatedText)
		assert.NotEmpty(t, generatedText, "Generated text should not be empty")

		// For doc2query, output should contain some relevant keywords
		lowercaseOutput := strings.ToLower(generatedText)
		hasRelevantTerm := strings.Contains(lowercaseOutput, "python") ||
			strings.Contains(lowercaseOutput, "programming") ||
			strings.Contains(lowercaseOutput, "language") ||
			strings.Contains(lowercaseOutput, "interpreted")
		assert.True(t, hasRelevantTerm, "Generated query should contain relevant terms")

		// Check statistics were recorded
		zero := uint64(0)
		stats := pipeline.GetStatistics()
		assert.Greater(t, stats.TokenizerExecutionCount, zero, "Tokenizer should have been called")
		assert.Greater(t, stats.OnnxExecutionCount, zero, "ONNX should have been called")
	})

	// Test 2: Batched input
	t.Run("BatchedInput", func(t *testing.T) {
		config := Seq2SeqConfig{
			ModelPath: modelPath,
			Name:      "testSeq2SeqBatchGo",
			Options: []backends.PipelineOption[*pipelines.Seq2SeqPipeline]{
				pipelines.WithSeq2SeqMaxTokens(32),
			},
		}
		pipeline, err := NewPipeline(session, config)
		checkT(t, err)

		inputs := []string{
			"Python is an interpreted programming language.",
			"Machine learning enables computers to learn from data.",
		}
		output, err := pipeline.RunPipeline(inputs)
		checkT(t, err)

		// Verify batch processing
		assert.Len(t, output.GeneratedTexts, 2, "Should have 2 outputs for 2 inputs")
		assert.NotEmpty(t, output.GeneratedTexts[0][0], "First output should not be empty")
		assert.NotEmpty(t, output.GeneratedTexts[1][0], "Second output should not be empty")

		t.Logf("Query 1 (Go): %s", output.GeneratedTexts[0][0])
		t.Logf("Query 2 (Go): %s", output.GeneratedTexts[1][0])
	})

	// Test 3: Sampling mode with top-p
	t.Run("SamplingMode", func(t *testing.T) {
		config := Seq2SeqConfig{
			ModelPath: modelPath,
			Name:      "testSeq2SeqSamplingGo",
			Options: []backends.PipelineOption[*pipelines.Seq2SeqPipeline]{
				pipelines.WithSeq2SeqMaxTokens(32),
				pipelines.WithSampling(0.95, 0.7),
			},
		}
		pipeline, err := NewPipeline(session, config)
		checkT(t, err)

		// Verify sampling config
		assert.True(t, pipeline.DoSample, "DoSample should be true")
		assert.InDelta(t, 0.95, pipeline.TopP, 0.001)
		assert.InDelta(t, 0.7, pipeline.Temperature, 0.001)

		inputs := []string{"Python is an interpreted programming language."}
		output, err := pipeline.RunPipeline(inputs)
		checkT(t, err)

		assert.NotEmpty(t, output.GeneratedTexts[0][0], "Sampled output should not be empty")
		t.Logf("Sampled query (Go): %s", output.GeneratedTexts[0][0])
	})

	// Test 4: Session management
	t.Run("SessionManagement", func(t *testing.T) {
		config := Seq2SeqConfig{
			ModelPath: modelPath,
			Name:      "testSessionMgmtGo",
			Options: []backends.PipelineOption[*pipelines.Seq2SeqPipeline]{
				pipelines.WithSeq2SeqMaxTokens(16),
			},
		}
		pipeline, err := NewPipeline(session, config)
		checkT(t, err)

		// Should be able to retrieve pipeline by name
		retrieved, err := GetPipeline[*pipelines.Seq2SeqPipeline](session, "testSessionMgmtGo")
		checkT(t, err)
		assert.Equal(t, pipeline, retrieved)

		// Should error for non-existent pipeline
		_, err = GetPipeline[*pipelines.Seq2SeqPipeline](session, "nonexistent")
		assert.Error(t, err)

		// Close pipeline should work
		err = ClosePipeline[*pipelines.Seq2SeqPipeline](session, "testSessionMgmtGo")
		checkT(t, err)

		// After closing, should not be retrievable
		_, err = GetPipeline[*pipelines.Seq2SeqPipeline](session, "testSessionMgmtGo")
		assert.Error(t, err)
	})

	// Test 5: LMQG/FLAN-T5 model support (if model is available)
	lmqgModelPath := "./models/flan-t5-small-qg"
	if _, err := os.Stat(lmqgModelPath); err == nil {
		t.Run("LMQGQuestionGeneration", func(t *testing.T) {
			config := Seq2SeqConfig{
				ModelPath: lmqgModelPath,
				Name:      "testLMQGPipelineGo",
				Options: []backends.PipelineOption[*pipelines.Seq2SeqPipeline]{
					pipelines.WithSeq2SeqMaxTokens(32),
				},
			}
			pipeline, err := NewPipeline(session, config)
			checkT(t, err)

			// Test with LMQG format using helper
			pairs := []pipelines.AnswerContextPair{
				{
					Answer:  "Beyonce",
					Context: "Beyonce further expanded her acting career, starring as blues singer Etta James in the 2008 musical biopic, Cadillac Records.",
				},
			}
			output, err := pipeline.RunQuestionGeneration(pairs)
			checkT(t, err)

			assert.Len(t, output.GeneratedTexts, 1)
			generatedQuestion := output.GeneratedTexts[0][0]
			t.Logf("Generated question (Go): %s", generatedQuestion)
			assert.NotEmpty(t, generatedQuestion)
		})
	}
}

func seq2seqPipelineValidationGo(t *testing.T, session *Session) {
	t.Helper()

	modelPath := "./models/doc2query-t5-small"

	// Check if model exists
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skipf("Model not found at %s. Run export_t5_onnx.py to create it.", modelPath)
	}

	config := Seq2SeqConfig{
		ModelPath: modelPath,
		Name:      "testSeq2SeqValidationGo",
	}
	pipeline, err := NewPipeline(session, config)
	checkT(t, err)

	// Test validation with invalid EOS tokens
	t.Run("NoEosTokens", func(t *testing.T) {
		origEos := pipeline.EosTokenIDs
		defer func() { pipeline.EosTokenIDs = origEos }()
		pipeline.EosTokenIDs = map[int64]bool{}
		err := pipeline.Validate()
		assert.Error(t, err, "Should fail with no EOS tokens")
	})

	// Test validation with invalid vocab size
	t.Run("ZeroVocabSize", func(t *testing.T) {
		origVocab := pipeline.VocabSize
		defer func() { pipeline.VocabSize = origVocab }()
		pipeline.VocabSize = 0
		err := pipeline.Validate()
		assert.Error(t, err, "Should fail with zero vocab size")
	})

	// Test validation with invalid max tokens
	t.Run("InvalidMaxTokens", func(t *testing.T) {
		origMax := pipeline.MaxNewTokens
		defer func() { pipeline.MaxNewTokens = origMax }()
		pipeline.MaxNewTokens = 0
		err := pipeline.Validate()
		assert.Error(t, err, "Should fail with zero max tokens")
	})

	// Test validation with nil encoder
	t.Run("NilEncoder", func(t *testing.T) {
		origEncoder := pipeline.EncoderModel
		defer func() { pipeline.EncoderModel = origEncoder }()
		pipeline.EncoderModel = nil
		err := pipeline.Validate()
		assert.Error(t, err, "Should fail with nil encoder")
	})

	// Test validation with nil decoder-init
	t.Run("NilDecoderInit", func(t *testing.T) {
		origDecInit := pipeline.DecoderInitModel
		defer func() { pipeline.DecoderInitModel = origDecInit }()
		pipeline.DecoderInitModel = nil
		err := pipeline.Validate()
		assert.Error(t, err, "Should fail with nil decoder-init")
	})

	// Test validation with nil decoder
	t.Run("NilDecoder", func(t *testing.T) {
		origDec := pipeline.DecoderModel
		defer func() { pipeline.DecoderModel = origDec }()
		pipeline.DecoderModel = nil
		err := pipeline.Validate()
		assert.Error(t, err, "Should fail with nil decoder")
	})

	// Test validation with nil tokenizer
	t.Run("NilTokenizer", func(t *testing.T) {
		origTok := pipeline.Tokenizer
		defer func() { pipeline.Tokenizer = origTok }()
		pipeline.Tokenizer = nil
		err := pipeline.Validate()
		assert.Error(t, err, "Should fail with nil tokenizer")
	})
}

// ============================================================================
// Seq2Seq Benchmarks (Go Runtime)
// ============================================================================

func BenchmarkSeq2SeqGreedyGo(b *testing.B) {
	if os.Getenv("CI") != "" {
		b.SkipNow()
	}

	modelPath := "./models/doc2query-t5-small"
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		b.Skipf("Model not found at %s", modelPath)
	}

	session, err := NewGoSession()
	if err != nil {
		b.Fatal(err)
	}
	defer session.Destroy()

	config := Seq2SeqConfig{
		ModelPath: modelPath,
		Name:      "benchSeq2SeqGreedyGo",
		Options: []backends.PipelineOption[*pipelines.Seq2SeqPipeline]{
			pipelines.WithSeq2SeqMaxTokens(32),
		},
	}
	pipeline, err := NewPipeline(session, config)
	if err != nil {
		b.Fatal(err)
	}

	inputs := []string{"Python is an interpreted, high-level programming language."}

	b.ResetTimer()
	for b.Loop() {
		_, err := pipeline.RunPipeline(inputs)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSeq2SeqSamplingGo(b *testing.B) {
	if os.Getenv("CI") != "" {
		b.SkipNow()
	}

	modelPath := "./models/doc2query-t5-small"
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		b.Skipf("Model not found at %s", modelPath)
	}

	session, err := NewGoSession()
	if err != nil {
		b.Fatal(err)
	}
	defer session.Destroy()

	config := Seq2SeqConfig{
		ModelPath: modelPath,
		Name:      "benchSeq2SeqSamplingGo",
		Options: []backends.PipelineOption[*pipelines.Seq2SeqPipeline]{
			pipelines.WithSeq2SeqMaxTokens(32),
			pipelines.WithSampling(0.95, 0.7),
		},
	}
	pipeline, err := NewPipeline(session, config)
	if err != nil {
		b.Fatal(err)
	}

	inputs := []string{"Python is an interpreted, high-level programming language."}

	b.ResetTimer()
	for b.Loop() {
		_, err := pipeline.RunPipeline(inputs)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSeq2SeqBatchGo(b *testing.B) {
	if os.Getenv("CI") != "" {
		b.SkipNow()
	}

	modelPath := "./models/doc2query-t5-small"
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		b.Skipf("Model not found at %s", modelPath)
	}

	session, err := NewGoSession()
	if err != nil {
		b.Fatal(err)
	}
	defer session.Destroy()

	config := Seq2SeqConfig{
		ModelPath: modelPath,
		Name:      "benchSeq2SeqBatchGo",
		Options: []backends.PipelineOption[*pipelines.Seq2SeqPipeline]{
			pipelines.WithSeq2SeqMaxTokens(32),
		},
	}
	pipeline, err := NewPipeline(session, config)
	if err != nil {
		b.Fatal(err)
	}

	// Batch of 10 documents
	inputs := make([]string, 10)
	for i := range inputs {
		inputs[i] = fmt.Sprintf("Document %d: Python is a popular programming language for data science and machine learning.", i+1)
	}

	b.ResetTimer()
	for b.Loop() {
		_, err := pipeline.RunPipeline(inputs)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkSeq2SeqLMQGGo benchmarks the LMQG/FLAN-T5 model for question generation with Go runtime.
func BenchmarkSeq2SeqLMQGGo(b *testing.B) {
	if os.Getenv("CI") != "" {
		b.SkipNow()
	}

	modelPath := "./models/flan-t5-small-qg"
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		b.Skipf("LMQG model not found at %s", modelPath)
	}

	session, err := NewGoSession()
	if err != nil {
		b.Fatal(err)
	}
	defer session.Destroy()

	config := Seq2SeqConfig{
		ModelPath: modelPath,
		Name:      "benchLMQGGo",
		Options: []backends.PipelineOption[*pipelines.Seq2SeqPipeline]{
			pipelines.WithSeq2SeqMaxTokens(32),
		},
	}
	pipeline, err := NewPipeline(session, config)
	if err != nil {
		b.Fatal(err)
	}

	pairs := []pipelines.AnswerContextPair{
		{
			Answer:  "Python",
			Context: "Python is a high-level programming language known for its readability and versatility.",
		},
	}

	b.ResetTimer()
	for b.Loop() {
		_, err := pipeline.RunQuestionGeneration(pairs)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkSeq2SeqLMQGBatchGo benchmarks batch question generation with LMQG using Go runtime.
func BenchmarkSeq2SeqLMQGBatchGo(b *testing.B) {
	if os.Getenv("CI") != "" {
		b.SkipNow()
	}

	modelPath := "./models/flan-t5-small-qg"
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		b.Skipf("LMQG model not found at %s", modelPath)
	}

	session, err := NewGoSession()
	if err != nil {
		b.Fatal(err)
	}
	defer session.Destroy()

	config := Seq2SeqConfig{
		ModelPath: modelPath,
		Name:      "benchLMQGBatchGo",
		Options: []backends.PipelineOption[*pipelines.Seq2SeqPipeline]{
			pipelines.WithSeq2SeqMaxTokens(32),
		},
	}
	pipeline, err := NewPipeline(session, config)
	if err != nil {
		b.Fatal(err)
	}

	// Batch of 5 question generation tasks
	pairs := []pipelines.AnswerContextPair{
		{Answer: "Python", Context: "Python is a high-level programming language."},
		{Answer: "Go", Context: "Go is a statically typed language developed by Google."},
		{Answer: "Rust", Context: "Rust is a systems programming language focused on safety."},
		{Answer: "JavaScript", Context: "JavaScript is the programming language of the web."},
		{Answer: "TypeScript", Context: "TypeScript is a typed superset of JavaScript."},
	}

	b.ResetTimer()
	for b.Loop() {
		_, err := pipeline.RunQuestionGeneration(pairs)
		if err != nil {
			b.Fatal(err)
		}
	}
}
