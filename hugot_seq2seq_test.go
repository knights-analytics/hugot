//go:build ORT || ALL

package hugot

import (
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/pipelines"
)

// ============================================================================
// Unit Tests (No model required)
// ============================================================================

func TestSeq2SeqPipelineInterfaceMethods(t *testing.T) {
	// Test pipeline interface accessor methods (no model required)
	pipeline := &pipelines.Seq2SeqPipeline{
		PipelineName:        "test-pipeline",
		Runtime:             "ORT",
		MaxNewTokens:        64,
		NumReturnSeqs:       3,
		DoSample:            true,
		TopP:                0.95,
		Temperature:         0.8,
		RepetitionPenalty:   1.1,
		DecoderStartTokenID: 0,
		EosTokenIDs:         map[int64]bool{1: true, 2: true},
		PadTokenID:          0,
		NumDecoderLayers:    6,
		VocabSize:           32000,
	}

	assert.Equal(t, "ORT", pipeline.GetRuntime())
	assert.Equal(t, 64, pipeline.GetMaxNewTokens())
	assert.Equal(t, 3, pipeline.GetNumReturnSeqs())
	assert.True(t, pipeline.GetDoSample())
	assert.InDelta(t, 0.95, pipeline.GetTopP(), 0.001)
	assert.InDelta(t, 0.8, pipeline.GetTemperature(), 0.001)
	assert.InDelta(t, 1.1, pipeline.GetRepetitionPenalty(), 0.001)
	assert.Equal(t, int64(0), pipeline.GetDecoderStartTokenID())
	assert.Equal(t, map[int64]bool{1: true, 2: true}, pipeline.GetEosTokenIDs())
	assert.Equal(t, int64(0), pipeline.GetPadTokenID())
	assert.Equal(t, 6, pipeline.GetNumDecoderLayers())
	assert.Equal(t, 32000, pipeline.GetVocabSize())

	// Test GetGenerationConfig
	genConfig := pipeline.GetGenerationConfig()
	assert.Equal(t, 64, genConfig["max_new_tokens"])
	assert.Equal(t, 3, genConfig["num_return_seqs"])
	assert.True(t, genConfig["do_sample"].(bool))
}

func TestSeq2SeqPipelineValidateWithoutModels(t *testing.T) {
	// Test validation logic without loading actual models
	pipeline := &pipelines.Seq2SeqPipeline{
		MaxNewTokens: 64,
		VocabSize:    32000,
		EosTokenIDs:  map[int64]bool{1: true},
	}

	// Should fail - no encoder
	err := pipeline.Validate()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "encoder")

	// Add mock encoder, should still fail - no decoder-init
	pipeline.EncoderModel = &backends.Model{}
	err = pipeline.Validate()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "decoder-init")

	// Add mock decoder-init, should still fail - no decoder
	pipeline.DecoderInitModel = &backends.Model{}
	err = pipeline.Validate()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "decoder model")

	// Add mock decoder, should still fail - no tokenizer
	pipeline.DecoderModel = &backends.Model{}
	err = pipeline.Validate()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "tokenizer")

	// Add mock tokenizer - should pass now
	pipeline.Tokenizer = &backends.Tokenizer{}
	err = pipeline.Validate()
	assert.NoError(t, err)
}

// ============================================================================
// ORT-Specific Tests (Require model)
// ============================================================================

func TestSeq2SeqPipelineORT(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow() // Skip in CI - requires model download
	}
	opts := []options.WithOption{options.WithOnnxLibraryPath(onnxRuntimeSharedLibrary)}
	session, err := NewORTSession(opts...)
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	seq2seqPipeline(t, session)
}

func TestSeq2SeqPipelineORTCuda(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	opts := []options.WithOption{
		options.WithOnnxLibraryPath("/usr/lib64/onnxruntime-gpu/libonnxruntime.so"),
		options.WithCuda(map[string]string{
			"device_id": "0",
		}),
	}
	session, err := NewORTSession(opts...)
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	seq2seqPipeline(t, session)
}

func TestSeq2SeqPipelineValidationORT(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	opts := []options.WithOption{options.WithOnnxLibraryPath(onnxRuntimeSharedLibrary)}
	session, err := NewORTSession(opts...)
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	seq2seqPipelineValidation(t, session)
}

func TestSeq2SeqLMQGHelpers(t *testing.T) {
	// Test LMQG input formatting helpers (no model required)
	t.Run("FormatLMQGInput", func(t *testing.T) {
		result := pipelines.FormatLMQGInput("Beyonce", "Beyonce starred as Etta James in Cadillac Records.")
		expected := "generate question: <hl> Beyonce <hl> Beyonce starred as Etta James in Cadillac Records."
		assert.Equal(t, expected, result)
	})

	t.Run("FormatLMQGInputBatch", func(t *testing.T) {
		pairs := []pipelines.AnswerContextPair{
			{Answer: "Python", Context: "Python is a programming language."},
			{Answer: "Go", Context: "Go is a compiled language."},
		}
		results := pipelines.FormatLMQGInputBatch(pairs)
		assert.Len(t, results, 2)
		assert.Equal(t, "generate question: <hl> Python <hl> Python is a programming language.", results[0])
		assert.Equal(t, "generate question: <hl> Go <hl> Go is a compiled language.", results[1])
	})

	t.Run("EmptyPairs", func(t *testing.T) {
		results := pipelines.FormatLMQGInputBatch([]pipelines.AnswerContextPair{})
		assert.Len(t, results, 0)
	})
}

func TestSeq2SeqBatchInterface(t *testing.T) {
	// Test Seq2SeqBatch methods (no model required)
	t.Run("NewSeq2SeqBatch", func(t *testing.T) {
		batch := pipelines.NewSeq2SeqBatch(3)
		assert.Equal(t, 3, batch.GetSize())
		assert.Len(t, batch.GetGeneratedTokens(), 3)
		assert.Len(t, batch.GetFinished(), 3)
		assert.Equal(t, 0, batch.GetFinishedCount())
	})

	t.Run("BatchSettersGetters", func(t *testing.T) {
		batch := pipelines.NewSeq2SeqBatch(2)

		// Test input setters/getters
		tokenIDs := [][]int64{{1, 2, 3}, {4, 5, 6}}
		batch.InputTokenIDs = tokenIDs
		assert.Equal(t, tokenIDs, batch.GetInputTokenIDs())

		attentionMask := [][]int64{{1, 1, 1}, {1, 1, 0}}
		batch.InputAttentionMask = attentionMask
		assert.Equal(t, attentionMask, batch.GetInputAttentionMask())

		batch.MaxInputLength = 3
		assert.Equal(t, 3, batch.GetMaxInputLength())

		// Test encoder state
		encoderStates := "mock_encoder_states"
		batch.SetEncoderHiddenStates(encoderStates)
		assert.Equal(t, encoderStates, batch.GetEncoderHiddenStates())

		encoderMask := "mock_encoder_mask"
		batch.SetEncoderAttentionMask(encoderMask)
		assert.Equal(t, encoderMask, batch.GetEncoderAttentionMask())

		// Test past key values
		pkv := []any{"kv1", "kv2"}
		batch.SetPastKeyValues(pkv)
		assert.Equal(t, pkv, batch.GetPastKeyValues())

		// Test logits
		logits := "mock_logits"
		batch.SetLogits(logits)
		assert.Equal(t, logits, batch.GetLogits())

		// Test generated tokens
		genTokens := [][]int64{{10, 11}, {20, 21}}
		batch.SetGeneratedTokens(genTokens)
		assert.Equal(t, genTokens, batch.GetGeneratedTokens())

		// Test finished state
		finished := []bool{true, false}
		batch.SetFinished(finished)
		assert.Equal(t, finished, batch.GetFinished())

		batch.SetFinishedCount(1)
		assert.Equal(t, 1, batch.GetFinishedCount())
	})

	t.Run("BatchDestroy", func(t *testing.T) {
		batch := pipelines.NewSeq2SeqBatch(1)

		// Default destroy functions should not error
		err := batch.Destroy()
		assert.NoError(t, err)

		// Custom destroy functions
		encoderDestroyed := false
		decoderDestroyed := false
		batch.SetDestroyEncoder(func() error {
			encoderDestroyed = true
			return nil
		})
		batch.SetDestroyDecoder(func() error {
			decoderDestroyed = true
			return nil
		})

		err = batch.Destroy()
		assert.NoError(t, err)
		assert.True(t, encoderDestroyed)
		assert.True(t, decoderDestroyed)
	})
}

func TestSeq2SeqOutputInterface(t *testing.T) {
	// Test Seq2SeqOutput methods (no model required)
	t.Run("GetOutput", func(t *testing.T) {
		output := &pipelines.Seq2SeqOutput{
			GeneratedTexts: [][]string{
				{"query 1", "query 2"},
				{"query 3"},
			},
			GeneratedTokens: [][][]uint32{
				{{1, 2}, {3, 4}},
				{{5, 6}},
			},
		}

		result := output.GetOutput()
		assert.Len(t, result, 2)
		assert.Equal(t, []string{"query 1", "query 2"}, result[0])
		assert.Equal(t, []string{"query 3"}, result[1])
	})

	t.Run("EmptyOutput", func(t *testing.T) {
		output := &pipelines.Seq2SeqOutput{
			GeneratedTexts:  [][]string{},
			GeneratedTokens: [][][]uint32{},
		}

		result := output.GetOutput()
		assert.Len(t, result, 0)
	})
}

func TestSeq2SeqPipelineOptionsORT(t *testing.T) {
	// Test pipeline option validation (doesn't require model)
	t.Run("MaxTokens", func(t *testing.T) {
		err := pipelines.WithSeq2SeqMaxTokens(0)(&pipelines.Seq2SeqPipeline{})
		assert.Error(t, err)

		err = pipelines.WithSeq2SeqMaxTokens(-1)(&pipelines.Seq2SeqPipeline{})
		assert.Error(t, err)

		pipeline := &pipelines.Seq2SeqPipeline{}
		err = pipelines.WithSeq2SeqMaxTokens(64)(pipeline)
		assert.NoError(t, err)
		assert.Equal(t, 64, pipeline.MaxNewTokens)
	})

	t.Run("NumReturnSequences", func(t *testing.T) {
		err := pipelines.WithNumReturnSequences(0)(&pipelines.Seq2SeqPipeline{})
		assert.Error(t, err)

		pipeline := &pipelines.Seq2SeqPipeline{}
		err = pipelines.WithNumReturnSequences(5)(pipeline)
		assert.NoError(t, err)
		assert.Equal(t, 5, pipeline.NumReturnSeqs)
	})

	t.Run("Sampling", func(t *testing.T) {
		err := pipelines.WithSampling(0.0, 1.0)(&pipelines.Seq2SeqPipeline{})
		assert.Error(t, err, "topP=0 should fail")

		err = pipelines.WithSampling(1.5, 1.0)(&pipelines.Seq2SeqPipeline{})
		assert.Error(t, err, "topP>1 should fail")

		err = pipelines.WithSampling(0.9, 0.0)(&pipelines.Seq2SeqPipeline{})
		assert.Error(t, err, "temperature=0 should fail")

		pipeline := &pipelines.Seq2SeqPipeline{}
		err = pipelines.WithSampling(0.95, 0.7)(pipeline)
		assert.NoError(t, err)
		assert.True(t, pipeline.DoSample)
		assert.InDelta(t, 0.95, pipeline.TopP, 0.001)
		assert.InDelta(t, 0.7, pipeline.Temperature, 0.001)
	})

	t.Run("RepetitionPenalty", func(t *testing.T) {
		err := pipelines.WithRepetitionPenalty(0.0)(&pipelines.Seq2SeqPipeline{})
		assert.Error(t, err)

		pipeline := &pipelines.Seq2SeqPipeline{}
		err = pipelines.WithRepetitionPenalty(1.2)(pipeline)
		assert.NoError(t, err)
		assert.InDelta(t, 1.2, pipeline.RepetitionPenalty, 0.001)
	})
}

func seq2seqPipeline(t *testing.T, session *Session) {
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
			Name:      "testSeq2SeqGreedy",
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

		// Output should be non-empty and reasonable (looks like a query)
		generatedText := output.GeneratedTexts[0][0]
		t.Logf("Generated query: %s", generatedText)
		assert.NotEmpty(t, generatedText, "Generated text should not be empty")

		// For doc2query, output should contain some keywords from input
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
			Name:      "testSeq2SeqBatch",
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

		t.Logf("Query 1: %s", output.GeneratedTexts[0][0])
		t.Logf("Query 2: %s", output.GeneratedTexts[1][0])
	})

	// Test 3: Sampling mode with top-p
	t.Run("SamplingMode", func(t *testing.T) {
		config := Seq2SeqConfig{
			ModelPath: modelPath,
			Name:      "testSeq2SeqSampling",
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
		t.Logf("Sampled query: %s", output.GeneratedTexts[0][0])
	})

	// Test 4: Generation config accessor
	t.Run("GenerationConfig", func(t *testing.T) {
		config := Seq2SeqConfig{
			ModelPath: modelPath,
			Name:      "testSeq2SeqConfig",
			Options: []backends.PipelineOption[*pipelines.Seq2SeqPipeline]{
				pipelines.WithSeq2SeqMaxTokens(64),
				pipelines.WithSampling(0.9, 0.8),
				pipelines.WithRepetitionPenalty(1.2),
			},
		}
		pipeline, err := NewPipeline(session, config)
		checkT(t, err)

		genConfig := pipeline.GetGenerationConfig()
		assert.Equal(t, 64, genConfig["max_new_tokens"])
		assert.True(t, genConfig["do_sample"].(bool))
		assert.InDelta(t, 0.9, genConfig["top_p"], 0.001)
		assert.InDelta(t, 0.8, genConfig["temperature"], 0.001)
		assert.InDelta(t, 1.2, genConfig["repetition_penalty"], 0.001)
	})

	// Test 5: Empty input handling
	t.Run("EmptyInput", func(t *testing.T) {
		config := Seq2SeqConfig{
			ModelPath: modelPath,
			Name:      "testSeq2SeqEmpty",
			Options: []backends.PipelineOption[*pipelines.Seq2SeqPipeline]{
				pipelines.WithSeq2SeqMaxTokens(16),
			},
		}
		pipeline, err := NewPipeline(session, config)
		checkT(t, err)

		// Empty batch should work (or return error gracefully)
		output, err := pipeline.RunPipeline([]string{""})
		// Either it works or fails gracefully
		if err == nil {
			assert.Len(t, output.GeneratedTexts, 1)
		}
	})

	// Test 6: Long input handling
	t.Run("LongInput", func(t *testing.T) {
		config := Seq2SeqConfig{
			ModelPath: modelPath,
			Name:      "testSeq2SeqLong",
			Options: []backends.PipelineOption[*pipelines.Seq2SeqPipeline]{
				pipelines.WithSeq2SeqMaxTokens(32),
			},
		}
		pipeline, err := NewPipeline(session, config)
		checkT(t, err)

		// Long input that would exceed typical context
		longInput := strings.Repeat("Python is a great programming language for data science. ", 50)
		output, err := pipeline.RunPipeline([]string{longInput})
		// Should either work or fail gracefully (truncation)
		if err == nil {
			assert.NotEmpty(t, output.GeneratedTexts[0][0])
			t.Logf("Long input query: %s", output.GeneratedTexts[0][0])
		}
	})

	// Test 7: LMQG/FLAN-T5 model support (if model is available)
	lmqgModelPath := "./models/flan-t5-small-qg"
	if _, err := os.Stat(lmqgModelPath); err == nil {
		t.Run("LMQGQuestionGeneration", func(t *testing.T) {
			config := Seq2SeqConfig{
				ModelPath: lmqgModelPath,
				Name:      "testLMQGPipeline",
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
			t.Logf("Generated question: %s", generatedQuestion)
			assert.NotEmpty(t, generatedQuestion)

			// Question should end with ? and likely mention Beyonce or related terms
			lowercaseQ := strings.ToLower(generatedQuestion)
			hasQuestionMark := strings.Contains(generatedQuestion, "?")
			hasRelevantTerm := strings.Contains(lowercaseQ, "beyonce") ||
				strings.Contains(lowercaseQ, "etta") ||
				strings.Contains(lowercaseQ, "cadillac") ||
				strings.Contains(lowercaseQ, "singer") ||
				strings.Contains(lowercaseQ, "actress") ||
				strings.Contains(lowercaseQ, "musical") ||
				strings.Contains(lowercaseQ, "who") ||
				strings.Contains(lowercaseQ, "what")
			assert.True(t, hasQuestionMark || hasRelevantTerm, "Generated output should look like a question")
		})

		t.Run("LMQGBatchQuestionGeneration", func(t *testing.T) {
			config := Seq2SeqConfig{
				ModelPath: lmqgModelPath,
				Name:      "testLMQGBatch",
				Options: []backends.PipelineOption[*pipelines.Seq2SeqPipeline]{
					pipelines.WithSeq2SeqMaxTokens(32),
				},
			}
			pipeline, err := NewPipeline(session, config)
			checkT(t, err)

			pairs := []pipelines.AnswerContextPair{
				{Answer: "Python", Context: "Python is a high-level programming language known for its readability."},
				{Answer: "Go", Context: "Go is a statically typed language developed by Google."},
			}
			output, err := pipeline.RunQuestionGeneration(pairs)
			checkT(t, err)

			assert.Len(t, output.GeneratedTexts, 2)
			t.Logf("Q1: %s", output.GeneratedTexts[0][0])
			t.Logf("Q2: %s", output.GeneratedTexts[1][0])
		})
	}

	// Test 8: Session management (GetPipeline, ClosePipeline)
	t.Run("SessionManagement", func(t *testing.T) {
		config := Seq2SeqConfig{
			ModelPath: modelPath,
			Name:      "testSessionMgmt",
			Options: []backends.PipelineOption[*pipelines.Seq2SeqPipeline]{
				pipelines.WithSeq2SeqMaxTokens(16),
			},
		}
		pipeline, err := NewPipeline(session, config)
		checkT(t, err)

		// Should be able to retrieve pipeline by name
		retrieved, err := GetPipeline[*pipelines.Seq2SeqPipeline](session, "testSessionMgmt")
		checkT(t, err)
		assert.Equal(t, pipeline, retrieved)

		// Should error for non-existent pipeline
		_, err = GetPipeline[*pipelines.Seq2SeqPipeline](session, "nonexistent")
		assert.Error(t, err)

		// Close pipeline should work
		err = ClosePipeline[*pipelines.Seq2SeqPipeline](session, "testSessionMgmt")
		checkT(t, err)

		// After closing, should not be retrievable
		_, err = GetPipeline[*pipelines.Seq2SeqPipeline](session, "testSessionMgmt")
		assert.Error(t, err)
	})

	// Test 9: Duplicate pipeline name should error
	t.Run("DuplicatePipelineName", func(t *testing.T) {
		config := Seq2SeqConfig{
			ModelPath: modelPath,
			Name:      "testDuplicate",
			Options: []backends.PipelineOption[*pipelines.Seq2SeqPipeline]{
				pipelines.WithSeq2SeqMaxTokens(16),
			},
		}
		_, err := NewPipeline(session, config)
		checkT(t, err)

		// Second pipeline with same name should fail
		_, err = NewPipeline(session, config)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "already been initialised")

		// Cleanup
		_ = ClosePipeline[*pipelines.Seq2SeqPipeline](session, "testDuplicate")
	})

	// Test 10: Run interface method
	t.Run("RunInterface", func(t *testing.T) {
		config := Seq2SeqConfig{
			ModelPath: modelPath,
			Name:      "testRunInterface",
			Options: []backends.PipelineOption[*pipelines.Seq2SeqPipeline]{
				pipelines.WithSeq2SeqMaxTokens(16),
			},
		}
		pipeline, err := NewPipeline(session, config)
		checkT(t, err)

		// Test Run() method (backends.Pipeline interface)
		output, err := pipeline.Run([]string{"Test input"})
		checkT(t, err)
		assert.NotNil(t, output)

		results := output.GetOutput()
		assert.Len(t, results, 1)
	})
}

func seq2seqPipelineValidation(t *testing.T, session *Session) {
	t.Helper()

	modelPath := "./models/doc2query-t5-small"

	// Check if model exists
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skipf("Model not found at %s. Run export_t5_onnx.py to create it.", modelPath)
	}

	config := Seq2SeqConfig{
		ModelPath: modelPath,
		Name:      "testSeq2SeqValidation",
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

// Benchmarks

func BenchmarkSeq2SeqGreedy(b *testing.B) {
	if os.Getenv("CI") != "" {
		b.SkipNow()
	}

	modelPath := "./models/doc2query-t5-small"
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		b.Skipf("Model not found at %s", modelPath)
	}

	opts := []options.WithOption{options.WithOnnxLibraryPath(onnxRuntimeSharedLibrary)}
	session, err := NewORTSession(opts...)
	if err != nil {
		b.Fatal(err)
	}
	defer session.Destroy()

	config := Seq2SeqConfig{
		ModelPath: modelPath,
		Name:      "benchSeq2SeqGreedy",
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

func BenchmarkSeq2SeqSampling(b *testing.B) {
	if os.Getenv("CI") != "" {
		b.SkipNow()
	}

	modelPath := "./models/doc2query-t5-small"
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		b.Skipf("Model not found at %s", modelPath)
	}

	opts := []options.WithOption{options.WithOnnxLibraryPath(onnxRuntimeSharedLibrary)}
	session, err := NewORTSession(opts...)
	if err != nil {
		b.Fatal(err)
	}
	defer session.Destroy()

	config := Seq2SeqConfig{
		ModelPath: modelPath,
		Name:      "benchSeq2SeqSampling",
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

func BenchmarkSeq2SeqBatch(b *testing.B) {
	if os.Getenv("CI") != "" {
		b.SkipNow()
	}

	modelPath := "./models/doc2query-t5-small"
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		b.Skipf("Model not found at %s", modelPath)
	}

	opts := []options.WithOption{options.WithOnnxLibraryPath(onnxRuntimeSharedLibrary)}
	session, err := NewORTSession(opts...)
	if err != nil {
		b.Fatal(err)
	}
	defer session.Destroy()

	config := Seq2SeqConfig{
		ModelPath: modelPath,
		Name:      "benchSeq2SeqBatch",
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

// BenchmarkSeq2SeqLMQG benchmarks the LMQG/FLAN-T5 model for question generation
func BenchmarkSeq2SeqLMQG(b *testing.B) {
	if os.Getenv("CI") != "" {
		b.SkipNow()
	}

	modelPath := "./models/flan-t5-small-qg"
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		b.Skipf("LMQG model not found at %s", modelPath)
	}

	opts := []options.WithOption{options.WithOnnxLibraryPath(onnxRuntimeSharedLibrary)}
	session, err := NewORTSession(opts...)
	if err != nil {
		b.Fatal(err)
	}
	defer session.Destroy()

	config := Seq2SeqConfig{
		ModelPath: modelPath,
		Name:      "benchLMQG",
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

// BenchmarkSeq2SeqLMQGBatch benchmarks batch question generation with LMQG
func BenchmarkSeq2SeqLMQGBatch(b *testing.B) {
	if os.Getenv("CI") != "" {
		b.SkipNow()
	}

	modelPath := "./models/flan-t5-small-qg"
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		b.Skipf("LMQG model not found at %s", modelPath)
	}

	opts := []options.WithOption{options.WithOnnxLibraryPath(onnxRuntimeSharedLibrary)}
	session, err := NewORTSession(opts...)
	if err != nil {
		b.Fatal(err)
	}
	defer session.Destroy()

	config := Seq2SeqConfig{
		ModelPath: modelPath,
		Name:      "benchLMQGBatch",
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
