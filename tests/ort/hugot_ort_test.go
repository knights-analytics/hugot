//go:build cgo && (ORT || ALL) && !TRAINING

package ort_test

import (
	"context"
	"fmt"
	"os"
	"testing"

	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/options"
	testutil "github.com/knights-analytics/hugot/tests"
)

// FEATURE EXTRACTION

func TestFeatureExtractionPipelineORT(t *testing.T) {
	session, err := hugot.NewORTSession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.FeatureExtractionPipeline(t, session)
}

func TestFeatureExtractionPipelineORTCuda(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	session, err := hugot.NewORTSession(t.Context(), options.WithCuda(map[string]string{
		"device_id": "0",
	}))
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.FeatureExtractionPipeline(t, session)
}

func TestFeatureExtractionPipelineValidationORT(t *testing.T) {
	session, err := hugot.NewORTSession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.FeatureExtractionPipelineValidation(t, session)
}

// Text classification

func TestTextClassificationPipelineORT(t *testing.T) {
	opts := []options.WithOption{
		options.WithOnnxLibraryPath("/usr/lib"),
		options.WithTelemetry(),
		options.WithCPUMemArena(true),
		options.WithMemPattern(true),
		options.WithIntraOpNumThreads(1),
		options.WithInterOpNumThreads(1),
	}
	session, err := hugot.NewORTSession(t.Context(), opts...)
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.TextClassificationPipeline(t, session)
}

func TestTextClassificationPipelineORTCuda(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	session, err := hugot.NewORTSession(t.Context(), options.WithCuda(map[string]string{
		"device_id": "0",
	}))
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.TextClassificationPipeline(t, session)
}

func TestTextClassificationPipelineMultiORT(t *testing.T) {
	opts := []options.WithOption{
		options.WithOnnxLibraryPath("/usr/lib"),
		options.WithTelemetry(),
		options.WithCPUMemArena(true),
		options.WithMemPattern(true),
		options.WithIntraOpNumThreads(1),
		options.WithInterOpNumThreads(1),
	}
	session, err := hugot.NewORTSession(t.Context(), opts...)
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.TextClassificationPipelineMulti(t, session)
}

func TestTextClassificationPipelineORTMultiCuda(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	session, err := hugot.NewORTSession(t.Context(), options.WithCuda(map[string]string{
		"device_id": "0",
	}))
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.TextClassificationPipelineMulti(t, session)
}

func TestTextClassificationPipelineValidationORT(t *testing.T) {
	session, err := hugot.NewORTSession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.TextClassificationPipelineValidation(t, session)
}

// Token classification

func TestTokenClassificationPipelineORT(t *testing.T) {
	session, err := hugot.NewORTSession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.TokenClassificationPipeline(t, session)
}

func TestTokenClassificationPipelineORTCuda(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	session, err := hugot.NewORTSession(t.Context(), options.WithCuda(map[string]string{
		"device_id": "0",
	}))
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.TokenClassificationPipeline(t, session)
}

func TestTokenClassificationPipelineValidationORT(t *testing.T) {
	session, err := hugot.NewORTSession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.TokenClassificationPipelineValidation(t, session)
}

// Zero shot

func TestZeroShotClassificationPipelineORT(t *testing.T) {
	session, err := hugot.NewORTSession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.ZeroShotClassificationPipeline(t, session)
}

func TestZeroShotClassificationPipelineORTCuda(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	session, err := hugot.NewORTSession(t.Context(), options.WithCuda(map[string]string{
		"device_id": "0",
	}))
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.ZeroShotClassificationPipeline(t, session)
}

func TestZeroShotClassificationPipelineValidationORT(t *testing.T) {
	session, err := hugot.NewORTSession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.ZeroShotClassificationPipelineValidation(t, session)
}

// Cross Encoder

func TestCrossEncoderPipelineORT(t *testing.T) {
	session, err := hugot.NewORTSession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.CrossEncoderPipeline(t, session)
}

func TestCrossEncoderPipelineORTCuda(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	session, err := hugot.NewORTSession(t.Context(), options.WithCuda(map[string]string{
		"device_id": "0",
	}))
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.CrossEncoderPipeline(t, session)
}

func TestCrossEncoderPipelineValidationORT(t *testing.T) {
	session, err := hugot.NewORTSession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.CrossEncoderPipelineValidation(t, session)
}

// Image classification

func TestImageClassificationPipelineORT(t *testing.T) {
	session, err := hugot.NewORTSession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.ImageClassificationPipeline(t, session)
}

func TestImageClassificationPipelineORTCuda(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	session, err := hugot.NewORTSession(t.Context(), options.WithCuda(map[string]string{
		"device_id": "0",
	}))
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.ImageClassificationPipeline(t, session)
}

func TestImageClassificationPipelineValidationORT(t *testing.T) {
	session, err := hugot.NewORTSession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.ImageClassificationPipelineValidation(t, session)
}

// Object detection

func TestObjectDetectionPipelineORT(t *testing.T) {
	session, err := hugot.NewORTSession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.ObjectDetectionPipeline(t, session)
}

func TestObjectDetectionPipelineORTCuda(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	session, err := hugot.NewORTSession(t.Context(), options.WithCuda(map[string]string{
		"device_id": "0",
	}))
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.ObjectDetectionPipeline(t, session)
}

func TestObjectDetectionPipelineValidationORT(t *testing.T) {
	session, err := hugot.NewORTSession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.ObjectDetectionPipelineValidation(t, session)
}

// Text generation
// These currently only run locally due to resource constraints in CI/CD

func TestTextGenerationPipelineORT(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	session, err := hugot.NewORTSession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.TextGenerationPipeline(t, session)
}

func TestTextGenerationPipelineORTCuda(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	session, err := hugot.NewORTSession(t.Context(),
		options.WithCuda(map[string]string{
			"device_id": "0",
		}),
		options.WithGenerativeEngine())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.TextGenerationPipeline(t, session)
}

func TestTextGenerationPipelineValidationORT(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	session, err := hugot.NewORTSession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.TextGenerationPipelineValidation(t, session)
}

// Question answering

func TestQAPipelineORT(t *testing.T) {
	session, err := hugot.NewORTSession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.QuestionAnsweringPipeline(t, session)
}

func TestQAPipelineORTCuda(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	session, err := hugot.NewORTSession(t.Context(), options.WithCuda(map[string]string{
		"device_id": "0",
	}))
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.QuestionAnsweringPipeline(t, session)
}

// Tabular pipeline

func TestTabularPipelineORT(t *testing.T) {
	session, err := hugot.NewORTSession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.TabularPipeline(t, session)
}

func TestTabularPipelineORTCuda(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	session, err := hugot.NewORTSession(t.Context(), options.WithCuda(map[string]string{
		"device_id": "0",
	}))
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.TabularPipeline(t, session)
}

// No same name

func TestNoSameNamePipelineORT(t *testing.T) {
	session, err := hugot.NewORTSession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.NoSameNamePipeline(t, session)
}

func TestClosePipelineORT(t *testing.T) {
	session, err := hugot.NewORTSession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.DestroyPipelines(t, session)
}

// Thread safety

func TestThreadSafetyORT(t *testing.T) {
	session, err := hugot.NewORTSession(t.Context())
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.ThreadSafety(t, session, 250)
}

func TestThreadSafetyORTCuda(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	session, err := hugot.NewORTSession(t.Context(), options.WithCuda(map[string]string{
		"device_id": "0",
	}))
	testutil.CheckT(t, err)
	defer func(session *hugot.Session) {
		destroyErr := session.Destroy()
		testutil.CheckT(t, destroyErr)
	}(session)
	testutil.ThreadSafety(t, session, 1000)
}

// Benchmarks

func runBenchmarkEmbedding(ctx context.Context, strings *[]string, cuda bool) {
	var opts []options.WithOption
	switch cuda {
	case true:
		opts = []options.WithOption{
			options.WithCuda(map[string]string{
				"device_id": "0",
			}),
		}
	default:
		opts = []options.WithOption{}
	}
	session, err := hugot.NewORTSession(ctx, opts...)
	if err != nil {
		panic(err)
	}

	defer func(session *hugot.Session) {
		errDestroy := session.Destroy()
		if errDestroy != nil {
			panic(errDestroy)
		}
	}(session)

	modelPath := testutil.ModelsFolder + "KnightsAnalytics_all-MiniLM-L6-v2"
	config := hugot.FeatureExtractionConfig{
		ModelPath: modelPath,
		Name:      "benchmarkEmbedding",
	}
	pipelineEmbedder, err2 := hugot.NewPipeline(session, config)
	if err2 != nil {
		panic(err2)
	}
	res, err := pipelineEmbedder.Run(ctx, *strings)
	if err != nil {
		panic(err)
	}
	fmt.Println(len(res.GetOutput()))
}

func BenchmarkORTCudaEmbedding(b *testing.B) {
	if os.Getenv("CI") != "" {
		b.SkipNow()
	}
	p := make([]string, 30000)
	for i := range 30000 {
		p[i] = "The goal of this library is to provide an easy, scalable, and hassle-free way to run huggingface transformer pipelines in golang applications."
	}
	for b.Loop() {
		runBenchmarkEmbedding(b.Context(), &p, true)
	}
}

func BenchmarkORTCPUEmbedding(b *testing.B) {
	if os.Getenv("CI") != "" {
		b.SkipNow()
	}
	p := make([]string, 5000)
	for i := range 5000 {
		p[i] = "The goal of this library is to provide an easy, scalable, and hassle-free way to run huggingface transformer pipelines in golang applications."
	}
	for b.Loop() {
		runBenchmarkEmbedding(b.Context(), &p, false)
	}
}
