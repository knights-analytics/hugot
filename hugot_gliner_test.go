//go:build (ORT || ALL) && !TRAINING

package hugot

import (
	"fmt"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/pipelines"
)

// =============================================================================
// GLiNER Unit Tests (No Model Required)
// =============================================================================

// TestGLiNERHelperFunctions tests the GLiNER helper functions
func TestGLiNERHelperFunctions(t *testing.T) {
	t.Run("BuildLabelPrefix", func(t *testing.T) {
		// Test with single label
		labels := []string{"person"}
		prefix := buildGLiNERLabelPrefixForTest(labels)
		assert.Contains(t, prefix, "<<ENT>>")
		assert.Contains(t, prefix, "person")
		assert.Contains(t, prefix, "<<SEP>>")

		// Test with multiple labels
		labels = []string{"person", "organization", "location"}
		prefix = buildGLiNERLabelPrefixForTest(labels)
		assert.Contains(t, prefix, "<<ENT>> person")
		assert.Contains(t, prefix, "<<ENT>> organization")
		assert.Contains(t, prefix, "<<ENT>> location")
		assert.True(t, strings.HasSuffix(prefix, "<<SEP>>"))
	})

	t.Run("CalculateLabelPrefixLength", func(t *testing.T) {
		labels := []string{"person"}
		length := calculateLabelPrefixLengthForTest(labels)
		// "<<ENT>> person <<SEP>> " = 7 + 1 + 6 + 1 + 7 + 1 = 23
		expectedLength := len("<<ENT>>") + 1 + len("person") + 1 + len("<<SEP>>") + 1
		assert.Equal(t, expectedLength, length)

		// Multiple labels
		labels = []string{"person", "org"}
		length = calculateLabelPrefixLengthForTest(labels)
		// "<<ENT>> person <<ENT>> org <<SEP>> "
		expectedLength = (len("<<ENT>>") + 1 + len("person") + 1) +
			(len("<<ENT>>") + 1 + len("org") + 1) +
			len("<<SEP>>") + 1
		assert.Equal(t, expectedLength, length)
	})

	t.Run("Sigmoid", func(t *testing.T) {
		// Test sigmoid function
		assert.InDelta(t, 0.5, sigmoidForTest(0), 0.001)
		assert.InDelta(t, 0.731, sigmoidForTest(1), 0.001)
		assert.InDelta(t, 0.269, sigmoidForTest(-1), 0.001)
		assert.InDelta(t, 0.9999, sigmoidForTest(10), 0.001)
		assert.InDelta(t, 0.0001, sigmoidForTest(-10), 0.001)
	})

	t.Run("RemoveNestedEntities", func(t *testing.T) {
		// Test that higher scoring entities are kept when overlapping
		entities := []pipelines.GLiNEREntity{
			{Text: "New York City", Start: 0, End: 13, Label: "location", Score: 0.9},
			{Text: "New York", Start: 0, End: 8, Label: "location", Score: 0.7},
			{Text: "City", Start: 9, End: 13, Label: "location", Score: 0.6},
		}
		result := removeNestedEntitiesForTest(entities)
		// Should only keep "New York City" as it has the highest score and overlaps with others
		assert.Len(t, result, 1)
		assert.Equal(t, "New York City", result[0].Text)

		// Test non-overlapping entities
		entities = []pipelines.GLiNEREntity{
			{Text: "John", Start: 0, End: 4, Label: "person", Score: 0.9},
			{Text: "Google", Start: 15, End: 21, Label: "organization", Score: 0.8},
		}
		result = removeNestedEntitiesForTest(entities)
		assert.Len(t, result, 2)
	})
}

// TestGLiNEROutputMethods tests the GLiNEROutput type methods
func TestGLiNEROutputMethods(t *testing.T) {
	t.Run("GetOutput", func(t *testing.T) {
		output := &pipelines.GLiNEROutput{
			Entities: [][]pipelines.GLiNEREntity{
				{{Text: "John", Label: "person"}},
				{{Text: "Google", Label: "organization"}},
			},
		}
		result := output.GetOutput()
		assert.Len(t, result, 2)
	})

	t.Run("HasRelations", func(t *testing.T) {
		output := &pipelines.GLiNEROutput{
			Entities: [][]pipelines.GLiNEREntity{{}},
		}
		assert.False(t, output.HasRelations())

		output.Relations = [][]pipelines.GLiNERRelation{
			{{Label: "works_at"}},
		}
		assert.True(t, output.HasRelations())
	})
}

// TestGLiNEROptionValidation tests option validation
func TestGLiNEROptionValidation(t *testing.T) {
	// Create a minimal pipeline struct for option testing
	p := &pipelines.GLiNERPipeline{}

	t.Run("ValidThreshold", func(t *testing.T) {
		opt := pipelines.WithGLiNERThreshold(0.5)
		err := opt(p)
		assert.NoError(t, err)
		assert.InDelta(t, 0.5, p.Threshold, 0.001)

		opt = pipelines.WithGLiNERThreshold(0.0)
		err = opt(p)
		assert.NoError(t, err)

		opt = pipelines.WithGLiNERThreshold(1.0)
		err = opt(p)
		assert.NoError(t, err)
	})

	t.Run("InvalidThreshold", func(t *testing.T) {
		opt := pipelines.WithGLiNERThreshold(-0.1)
		err := opt(p)
		assert.Error(t, err)

		opt = pipelines.WithGLiNERThreshold(1.1)
		err = opt(p)
		assert.Error(t, err)
	})

	t.Run("ValidMaxWidth", func(t *testing.T) {
		opt := pipelines.WithGLiNERMaxWidth(12)
		err := opt(p)
		assert.NoError(t, err)
		assert.Equal(t, 12, p.MaxWidth)
	})

	t.Run("InvalidMaxWidth", func(t *testing.T) {
		opt := pipelines.WithGLiNERMaxWidth(0)
		err := opt(p)
		assert.Error(t, err)

		opt = pipelines.WithGLiNERMaxWidth(-1)
		err = opt(p)
		assert.Error(t, err)
	})

	t.Run("Labels", func(t *testing.T) {
		opt := pipelines.WithGLiNERLabels([]string{"person", "org"})
		err := opt(p)
		assert.NoError(t, err)
		assert.Equal(t, []string{"person", "org"}, p.Labels)
	})

	t.Run("FlatNER", func(t *testing.T) {
		p.FlatNER = false
		opt := pipelines.WithGLiNERFlatNER()
		err := opt(p)
		assert.NoError(t, err)
		assert.True(t, p.FlatNER)
	})

	t.Run("MultiLabel", func(t *testing.T) {
		p.MultiLabel = false
		opt := pipelines.WithGLiNERMultiLabel()
		err := opt(p)
		assert.NoError(t, err)
		assert.True(t, p.MultiLabel)
	})

	t.Run("RelationThreshold", func(t *testing.T) {
		opt := pipelines.WithGLiNERRelationThreshold(0.7)
		err := opt(p)
		assert.NoError(t, err)
		assert.InDelta(t, 0.7, p.RelationThreshold, 0.001)

		opt = pipelines.WithGLiNERRelationThreshold(-0.1)
		err = opt(p)
		assert.Error(t, err)
	})

	t.Run("SequencePacking", func(t *testing.T) {
		opt := pipelines.WithGLiNERSequencePacking(512)
		err := opt(p)
		assert.NoError(t, err)
		assert.True(t, p.PackingEnabled)
		assert.Equal(t, 512, p.MaxPackedLen)

		// Test default value when 0 is passed
		opt = pipelines.WithGLiNERSequencePacking(0)
		err = opt(p)
		assert.NoError(t, err)
		assert.Equal(t, 512, p.MaxPackedLen)
	})
}

// TestGLiNERRelationHelpers tests relation extraction helper functions
func TestGLiNERRelationHelpers(t *testing.T) {
	relations := []pipelines.GLiNERRelation{
		{
			HeadEntity: pipelines.GLiNEREntity{Text: "John", Start: 0, End: 4},
			TailEntity: pipelines.GLiNEREntity{Text: "Google", Start: 15, End: 21},
			Label:      "works_at",
			Score:      0.9,
		},
		{
			HeadEntity: pipelines.GLiNEREntity{Text: "John", Start: 0, End: 4},
			TailEntity: pipelines.GLiNEREntity{Text: "NYC", Start: 30, End: 33},
			Label:      "lives_in",
			Score:      0.6,
		},
		{
			HeadEntity: pipelines.GLiNEREntity{Text: "Google", Start: 15, End: 21},
			TailEntity: pipelines.GLiNEREntity{Text: "California", Start: 45, End: 55},
			Label:      "located_in",
			Score:      0.8,
		},
	}

	t.Run("FilterRelationsByScore", func(t *testing.T) {
		filtered := pipelines.FilterRelationsByScore(relations, 0.7)
		assert.Len(t, filtered, 2)
		for _, rel := range filtered {
			assert.GreaterOrEqual(t, rel.Score, float32(0.7))
		}
	})

	t.Run("GroupRelationsByHead", func(t *testing.T) {
		grouped := pipelines.GroupRelationsByHead(relations)
		assert.Len(t, grouped, 2) // John and Google as heads
	})

	t.Run("GroupRelationsByType", func(t *testing.T) {
		grouped := pipelines.GroupRelationsByType(relations)
		assert.Len(t, grouped, 3) // works_at, lives_in, located_in
		assert.Len(t, grouped["works_at"], 1)
	})
}

// =============================================================================
// GLiNER Integration Tests (Require Model)
// =============================================================================

// Default GLiNER model path for tests
const glinerModelPath = "./models/gliner-small-v2.1"

// OnnxFilename is just the filename - the path walking finds it in subdirectories
const glinerOnnxFilename = "model.onnx"

// TestGLiNERPipelineORT runs the GLiNER integration tests with ORT backend
func TestGLiNERPipelineORT(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.Skip("Skipping GLiNER integration tests in CI - requires model download")
	}

	opts := []options.WithOption{options.WithOnnxLibraryPath(onnxRuntimeSharedLibrary)}
	session, err := NewORTSession(opts...)
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)

	glinerPipeline(t, session)
}

// TestGLiNERPipelineValidationORT tests validation errors
func TestGLiNERPipelineValidationORT(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.Skip("Skipping GLiNER validation tests in CI - requires model download")
	}

	opts := []options.WithOption{options.WithOnnxLibraryPath(onnxRuntimeSharedLibrary)}
	session, err := NewORTSession(opts...)
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)

	glinerPipelineValidation(t, session)
}

func glinerPipeline(t *testing.T, session *Session) {
	t.Helper()

	// Check if model exists
	if _, err := os.Stat(glinerModelPath); os.IsNotExist(err) {
		t.Skipf("GLiNER model not found at %s. Download with: python -c \"from gliner import GLiNER; m = GLiNER.from_pretrained('urchade/gliner_small-v2.1'); m.save_pretrained('%s')\"", glinerModelPath, glinerModelPath)
	}

	// Test 1: Basic entity extraction with default labels
	t.Run("BasicEntityExtraction", func(t *testing.T) {
		config := GLiNERConfig{
			ModelPath:    glinerModelPath,
			OnnxFilename: glinerOnnxFilename,
			Name:         "testGLiNERBasic",
			Options: []backends.PipelineOption[*pipelines.GLiNERPipeline]{
				pipelines.WithGLiNERLabels([]string{"person", "organization", "location"}),
				pipelines.WithGLiNERThreshold(0.5),
			},
		}
		pipeline, err := NewPipeline(session, config)
		checkT(t, err)

		inputs := []string{"Steve Jobs founded Apple Inc. in Cupertino, California."}
		output, err := pipeline.RunPipeline(inputs)
		checkT(t, err)

		assert.Len(t, output.Entities, 1, "Should have entities for 1 input")
		entities := output.Entities[0]
		t.Logf("Found %d entities", len(entities))
		for _, e := range entities {
			t.Logf("  Entity: %q (%s) [%d-%d] score=%.3f", e.Text, e.Label, e.Start, e.End, e.Score)
		}

		// We should find at least some entities
		assert.NotEmpty(t, entities, "Should extract at least one entity")

		// Check that entities have valid structure
		for _, e := range entities {
			assert.NotEmpty(t, e.Text, "Entity text should not be empty")
			assert.NotEmpty(t, e.Label, "Entity label should not be empty")
			assert.GreaterOrEqual(t, e.Score, float32(0.5), "Score should be >= threshold")
			assert.Less(t, e.Start, e.End, "Start should be < End")
		}
	})

	// Test 2: Zero-shot NER with custom labels
	t.Run("ZeroShotCustomLabels", func(t *testing.T) {
		config := GLiNERConfig{
			ModelPath:    glinerModelPath,
			OnnxFilename: glinerOnnxFilename,
			Name:         "testGLiNERZeroShot",
		}
		pipeline, err := NewPipeline(session, config)
		checkT(t, err)

		// Use custom labels for a specific domain
		customLabels := []string{"programming language", "technology company", "software"}
		inputs := []string{"Python is developed by the Python Software Foundation and is widely used at Google."}

		output, err := pipeline.RunPipelineWithLabels(inputs, customLabels)
		checkT(t, err)

		assert.Len(t, output.Entities, 1)
		entities := output.Entities[0]
		t.Logf("Zero-shot entities with custom labels:")
		for _, e := range entities {
			t.Logf("  Entity: %q (%s) score=%.3f", e.Text, e.Label, e.Score)
		}
	})

	// Test 3: Batched input
	t.Run("BatchedInput", func(t *testing.T) {
		config := GLiNERConfig{
			ModelPath:    glinerModelPath,
			OnnxFilename: glinerOnnxFilename,
			Name:         "testGLiNERBatch",
			Options: []backends.PipelineOption[*pipelines.GLiNERPipeline]{
				pipelines.WithGLiNERLabels([]string{"person", "organization"}),
			},
		}
		pipeline, err := NewPipeline(session, config)
		checkT(t, err)

		inputs := []string{
			"Elon Musk is the CEO of Tesla.",
			"Sundar Pichai leads Google.",
			"Tim Cook runs Apple.",
		}
		output, err := pipeline.RunPipeline(inputs)
		checkT(t, err)

		assert.Len(t, output.Entities, 3, "Should have entities for each input")

		for i, entities := range output.Entities {
			t.Logf("Input %d: %d entities found", i, len(entities))
			for _, e := range entities {
				t.Logf("  %q (%s)", e.Text, e.Label)
			}
		}
	})

	// Test 4: Different threshold values
	t.Run("ThresholdVariation", func(t *testing.T) {
		// High threshold
		configHigh := GLiNERConfig{
			ModelPath:    glinerModelPath,
			OnnxFilename: glinerOnnxFilename,
			Name:         "testGLiNERHighThreshold",
			Options: []backends.PipelineOption[*pipelines.GLiNERPipeline]{
				pipelines.WithGLiNERLabels([]string{"person", "organization"}),
				pipelines.WithGLiNERThreshold(0.9),
			},
		}
		pipelineHigh, err := NewPipeline(session, configHigh)
		checkT(t, err)

		// Low threshold
		configLow := GLiNERConfig{
			ModelPath:    glinerModelPath,
			OnnxFilename: glinerOnnxFilename,
			Name:         "testGLiNERLowThreshold",
			Options: []backends.PipelineOption[*pipelines.GLiNERPipeline]{
				pipelines.WithGLiNERLabels([]string{"person", "organization"}),
				pipelines.WithGLiNERThreshold(0.3),
			},
		}
		pipelineLow, err := NewPipeline(session, configLow)
		checkT(t, err)

		inputs := []string{"John Smith works at Acme Corporation."}

		outputHigh, err := pipelineHigh.RunPipeline(inputs)
		checkT(t, err)
		outputLow, err := pipelineLow.RunPipeline(inputs)
		checkT(t, err)

		// Low threshold should find at least as many entities as high threshold
		assert.GreaterOrEqual(t, len(outputLow.Entities[0]), len(outputHigh.Entities[0]),
			"Lower threshold should find >= entities than higher threshold")

		t.Logf("High threshold (0.9): %d entities", len(outputHigh.Entities[0]))
		t.Logf("Low threshold (0.3): %d entities", len(outputLow.Entities[0]))
	})

	// Test 5: Empty input handling
	t.Run("EmptyInput", func(t *testing.T) {
		config := GLiNERConfig{
			ModelPath:    glinerModelPath,
			OnnxFilename: glinerOnnxFilename,
			Name:         "testGLiNEREmpty",
		}
		pipeline, err := NewPipeline(session, config)
		checkT(t, err)

		// Empty slice
		output, err := pipeline.RunPipeline([]string{})
		checkT(t, err)
		assert.Empty(t, output.Entities)
	})

	// Test 6: Long text handling
	t.Run("LongText", func(t *testing.T) {
		config := GLiNERConfig{
			ModelPath:    glinerModelPath,
			OnnxFilename: glinerOnnxFilename,
			Name:         "testGLiNERLong",
			Options: []backends.PipelineOption[*pipelines.GLiNERPipeline]{
				pipelines.WithGLiNERLabels([]string{"person", "organization", "location"}),
			},
		}
		pipeline, err := NewPipeline(session, config)
		checkT(t, err)

		// Long text with multiple entities
		longText := `Steve Jobs, the co-founder of Apple Inc., was born in San Francisco, California.
He later moved to Cupertino where Apple's headquarters is located. Jobs worked closely
with Steve Wozniak and Ronald Wayne to establish the company in 1976. The company grew
to become one of the largest technology corporations in the world, headquartered in
Cupertino. Tim Cook succeeded Jobs as CEO after his passing in 2011.`

		output, err := pipeline.RunPipeline([]string{longText})
		checkT(t, err)

		entities := output.Entities[0]
		t.Logf("Long text: found %d entities", len(entities))

		// Group by label for analysis
		byLabel := make(map[string][]string)
		for _, e := range entities {
			byLabel[e.Label] = append(byLabel[e.Label], e.Text)
		}
		for label, texts := range byLabel {
			t.Logf("  %s: %v", label, texts)
		}
	})

	// Test 7: Session management
	t.Run("SessionManagement", func(t *testing.T) {
		config := GLiNERConfig{
			ModelPath:    glinerModelPath,
			OnnxFilename: glinerOnnxFilename,
			Name:         "testGLiNERSessionMgmt",
		}
		pipeline, err := NewPipeline(session, config)
		checkT(t, err)

		// Should be retrievable by name
		retrieved, err := GetPipeline[*pipelines.GLiNERPipeline](session, "testGLiNERSessionMgmt")
		checkT(t, err)
		assert.Equal(t, pipeline, retrieved)

		// Should error for non-existent pipeline
		_, err = GetPipeline[*pipelines.GLiNERPipeline](session, "nonexistent")
		assert.Error(t, err)

		// Close pipeline
		err = ClosePipeline[*pipelines.GLiNERPipeline](session, "testGLiNERSessionMgmt")
		checkT(t, err)

		// After closing, should not be retrievable
		_, err = GetPipeline[*pipelines.GLiNERPipeline](session, "testGLiNERSessionMgmt")
		assert.Error(t, err)
	})

	// Test 8: Statistics collection
	t.Run("Statistics", func(t *testing.T) {
		config := GLiNERConfig{
			ModelPath:    glinerModelPath,
			OnnxFilename: glinerOnnxFilename,
			Name:         "testGLiNERStats",
		}
		pipeline, err := NewPipeline(session, config)
		checkT(t, err)

		// Run a few inferences
		inputs := []string{"John works at Google."}
		for i := 0; i < 3; i++ {
			_, err := pipeline.RunPipeline(inputs)
			checkT(t, err)
		}

		stats := pipeline.GetStatistics()
		assert.Greater(t, stats.TokenizerExecutionCount, uint64(0), "Tokenizer should have been called")
		assert.Greater(t, stats.OnnxExecutionCount, uint64(0), "ONNX should have been called")

		t.Logf("Statistics: Tokenizer calls=%d, ONNX calls=%d",
			stats.TokenizerExecutionCount, stats.OnnxExecutionCount)
	})

	// Test 9: Max width option
	// Note: GLiNER ONNX models are exported with a fixed max_width (typically 12).
	// Changing max_width at runtime is only valid if it matches the export-time setting.
	// Using a different max_width will cause shape mismatches in the model.
	t.Run("MaxWidthOption", func(t *testing.T) {
		// Test with default max_width (12) - matches the model's export setting
		config := GLiNERConfig{
			ModelPath:    glinerModelPath,
			OnnxFilename: glinerOnnxFilename,
			Name:         "testGLiNERDefaultWidth",
			Options: []backends.PipelineOption[*pipelines.GLiNERPipeline]{
				pipelines.WithGLiNERLabels([]string{"person", "organization"}),
				// Default max_width=12 is used, matching the onnx-community model export
			},
		}
		pipeline, err := NewPipeline(session, config)
		checkT(t, err)

		inputs := []string{"The New York City Department of Transportation manages roads."}

		output, err := pipeline.RunPipeline(inputs)
		checkT(t, err)

		// Should be able to detect multi-word entities
		t.Logf("Default max width (12): %d entities", len(output.Entities[0]))
		for _, entity := range output.Entities[0] {
			t.Logf("  Entity: %q (%s) [%d-%d] score=%.3f",
				entity.Text, entity.Label, entity.Start, entity.End, entity.Score)
		}
	})
}

func glinerPipelineValidation(t *testing.T, session *Session) {
	t.Helper()

	// Check if model exists
	if _, err := os.Stat(glinerModelPath); os.IsNotExist(err) {
		t.Skipf("GLiNER model not found at %s", glinerModelPath)
	}

	// Test validation with empty labels
	t.Run("EmptyLabels", func(t *testing.T) {
		config := GLiNERConfig{
			ModelPath:    glinerModelPath,
			OnnxFilename: glinerOnnxFilename,
			Name:         "testGLiNEREmptyLabels",
			Options: []backends.PipelineOption[*pipelines.GLiNERPipeline]{
				pipelines.WithGLiNERLabels([]string{}),
			},
		}
		_, err := NewPipeline(session, config)
		assert.Error(t, err, "Should fail with empty labels")
	})
}

// TestGLiNERRuntimeValidation tests that GLiNER rejects non-ORT runtimes
func TestGLiNERRuntimeValidation(t *testing.T) {
	// This test checks that the validation correctly rejects non-ORT runtimes
	// We create a minimal pipeline struct with just enough to test runtime validation

	// Create a mock model with the required input metadata
	mockModel := &backends.Model{
		InputsMeta: []backends.InputOutputInfo{
			{Name: "input_ids"},
			{Name: "attention_mask"},
			{Name: "words_mask"},
			{Name: "text_lengths"},
			{Name: "span_idx"},
			{Name: "span_mask"},
		},
		OutputsMeta: []backends.InputOutputInfo{
			{Name: "output"},
		},
	}

	p := &pipelines.GLiNERPipeline{
		BasePipeline: &backends.BasePipeline{
			Runtime: "GO",
			Model:   mockModel,
		},
		Labels: []string{"person"},
	}

	err := p.Validate()
	assert.Error(t, err, "Should fail with GO runtime")
	assert.Contains(t, err.Error(), "ORT backend")
}

// =============================================================================
// GLiNER Benchmarks
// =============================================================================

func BenchmarkGLiNERSingleInput(b *testing.B) {
	if os.Getenv("CI") != "" {
		b.Skip("Skipping benchmark in CI")
	}

	if _, err := os.Stat(glinerModelPath); os.IsNotExist(err) {
		b.Skipf("GLiNER model not found at %s", glinerModelPath)
	}

	opts := []options.WithOption{options.WithOnnxLibraryPath(onnxRuntimeSharedLibrary)}
	session, err := NewORTSession(opts...)
	if err != nil {
		b.Fatal(err)
	}
	defer session.Destroy()

	config := GLiNERConfig{
		ModelPath:    glinerModelPath,
		OnnxFilename: glinerOnnxFilename,
		Name:         "benchGLiNERSingle",
		Options: []backends.PipelineOption[*pipelines.GLiNERPipeline]{
			pipelines.WithGLiNERLabels([]string{"person", "organization", "location"}),
		},
	}
	pipeline, err := NewPipeline(session, config)
	if err != nil {
		b.Fatal(err)
	}

	inputs := []string{"Steve Jobs founded Apple Inc. in Cupertino, California."}

	b.ResetTimer()
	for b.Loop() {
		_, err := pipeline.RunPipeline(inputs)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkGLiNERBatchInput(b *testing.B) {
	if os.Getenv("CI") != "" {
		b.Skip("Skipping benchmark in CI")
	}

	if _, err := os.Stat(glinerModelPath); os.IsNotExist(err) {
		b.Skipf("GLiNER model not found at %s", glinerModelPath)
	}

	opts := []options.WithOption{options.WithOnnxLibraryPath(onnxRuntimeSharedLibrary)}
	session, err := NewORTSession(opts...)
	if err != nil {
		b.Fatal(err)
	}
	defer session.Destroy()

	config := GLiNERConfig{
		ModelPath:    glinerModelPath,
		OnnxFilename: glinerOnnxFilename,
		Name:         "benchGLiNERBatch",
		Options: []backends.PipelineOption[*pipelines.GLiNERPipeline]{
			pipelines.WithGLiNERLabels([]string{"person", "organization", "location"}),
		},
	}
	pipeline, err := NewPipeline(session, config)
	if err != nil {
		b.Fatal(err)
	}

	// Batch of 5 inputs
	inputs := []string{
		"Steve Jobs founded Apple Inc. in Cupertino.",
		"Elon Musk is the CEO of Tesla.",
		"Sundar Pichai leads Google in Mountain View.",
		"Tim Cook runs Apple from California.",
		"Satya Nadella is CEO of Microsoft in Redmond.",
	}

	b.ResetTimer()
	for b.Loop() {
		_, err := pipeline.RunPipeline(inputs)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkGLiNERLabelCount(b *testing.B) {
	if os.Getenv("CI") != "" {
		b.Skip("Skipping benchmark in CI")
	}

	if _, err := os.Stat(glinerModelPath); os.IsNotExist(err) {
		b.Skipf("GLiNER model not found at %s", glinerModelPath)
	}

	opts := []options.WithOption{options.WithOnnxLibraryPath(onnxRuntimeSharedLibrary)}
	session, err := NewORTSession(opts...)
	if err != nil {
		b.Fatal(err)
	}
	defer session.Destroy()

	inputs := []string{"Steve Jobs founded Apple Inc. in Cupertino, California."}

	// Test with different numbers of labels
	labelCounts := []int{3, 5, 10}
	allLabels := []string{"person", "organization", "location", "product", "event",
		"date", "money", "time", "quantity", "technology"}

	for _, count := range labelCounts {
		labels := allLabels[:count]
		b.Run(fmt.Sprintf("Labels_%d", count), func(b *testing.B) {
			config := GLiNERConfig{
				ModelPath:    glinerModelPath,
				OnnxFilename: glinerOnnxFilename,
				Name:         fmt.Sprintf("benchGLiNER_%d_labels", count),
				Options: []backends.PipelineOption[*pipelines.GLiNERPipeline]{
					pipelines.WithGLiNERLabels(labels),
				},
			}
			pipeline, err := NewPipeline(session, config)
			if err != nil {
				b.Fatal(err)
			}

			b.ResetTimer()
			for b.Loop() {
				_, err := pipeline.RunPipeline(inputs)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// =============================================================================
// Helper function wrappers (to access internal functions for testing)
// =============================================================================

// These wrappers expose internal GLiNER functions for unit testing.
// They call the actual implementation through test-accessible means.

func buildGLiNERLabelPrefixForTest(labels []string) string {
	var sb strings.Builder
	for _, label := range labels {
		sb.WriteString("<<ENT>>")
		sb.WriteString(" ")
		sb.WriteString(label)
		sb.WriteString(" ")
	}
	sb.WriteString("<<SEP>>")
	return sb.String()
}

func calculateLabelPrefixLengthForTest(labels []string) int {
	length := 0
	for _, label := range labels {
		length += len("<<ENT>>") + 1 + len(label) + 1
	}
	length += len("<<SEP>>") + 1
	return length
}

func sigmoidForTest(x float32) float32 {
	return 1.0 / (1.0 + float32(math.Exp(-float64(x))))
}

func removeNestedEntitiesForTest(entities []pipelines.GLiNEREntity) []pipelines.GLiNEREntity {
	if len(entities) <= 1 {
		return entities
	}

	// Sort by score descending
	sorted := make([]pipelines.GLiNEREntity, len(entities))
	copy(sorted, entities)
	for i := 0; i < len(sorted)-1; i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[j].Score > sorted[i].Score {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}

	var result []pipelines.GLiNEREntity
	for _, entity := range sorted {
		overlaps := false
		for _, kept := range result {
			if entity.Start < kept.End && entity.End > kept.Start {
				overlaps = true
				break
			}
		}
		if !overlaps {
			result = append(result, entity)
		}
	}

	return result
}
