package hugot

import (
	"fmt"
	"os"
	"testing"

	"github.com/knights-analytics/hugot/options"
)

// FEATURE EXTRACTION

func TestFeatureExtractionPipelineORT(t *testing.T) {
	opts := []options.WithOption{options.WithOnnxLibraryPath(onnxRuntimeSharedLibrary)}
	session, err := NewORTSession(opts...)
	check(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		check(t, destroyErr)
	}(session)
	featureExtractionPipeline(t, session)
}

func TestFeatureExtractionPipelineORTCuda(t *testing.T) {
	opts := []options.WithOption{
		options.WithOnnxLibraryPath("/usr/lib64/onnxruntime-gpu/libonnxruntime.so"),
		options.WithCuda(map[string]string{
			"device_id": "0",
		}),
	}
	session, err := NewORTSession(opts...)
	check(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		check(t, destroyErr)
	}(session)
	featureExtractionPipeline(t, session)
}

func TestFeatureExtractionPipelineValidationORT(t *testing.T) {
	opts := []options.WithOption{options.WithOnnxLibraryPath(onnxRuntimeSharedLibrary)}
	session, err := NewORTSession(opts...)
	check(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		check(t, destroyErr)
	}(session)
	featureExtractionPipelineValidation(t, session)
}

// Text classification

func TestTextClassificationPipelineORT(t *testing.T) {
	opts := []options.WithOption{
		options.WithOnnxLibraryPath(onnxRuntimeSharedLibrary),
		options.WithTelemetry(),
		options.WithCpuMemArena(true),
		options.WithMemPattern(true),
		options.WithIntraOpNumThreads(1),
		options.WithInterOpNumThreads(1),
	}
	session, err := NewORTSession(opts...)
	check(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		check(t, destroyErr)
	}(session)
	textClassificationPipeline(t, session)

}

func TestTextClassificationPipelineORTCuda(t *testing.T) {
	opts := []options.WithOption{
		options.WithOnnxLibraryPath("/usr/lib64/onnxruntime-gpu/libonnxruntime.so"),
		options.WithCuda(map[string]string{
			"device_id": "0",
		}),
	}
	session, err := NewORTSession(opts...)
	check(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		check(t, destroyErr)
	}(session)
	textClassificationPipeline(t, session)
}

func TestTextClassificationPipelineValidationORT(t *testing.T) {
	opts := []options.WithOption{options.WithOnnxLibraryPath(onnxRuntimeSharedLibrary)}
	session, err := NewORTSession(opts...)
	check(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		check(t, destroyErr)
	}(session)
	textClassificationPipelineValidation(t, session)
}

// Zero shot

func TestZeroShotClassificationPipelineORT(t *testing.T) {
	session, err := NewORTSession()
	check(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		check(t, destroyErr)
	}(session)
	zeroShotClassificationPipeline(t, session)
}

func TestZeroShotClassificationPipelineORTCuda(t *testing.T) {
	opts := []options.WithOption{
		options.WithOnnxLibraryPath("/usr/lib64/onnxruntime-gpu/libonnxruntime.so"),
		options.WithCuda(map[string]string{
			"device_id": "0",
		}),
	}
	session, err := NewORTSession(opts...)
	check(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		check(t, destroyErr)
	}(session)
	zeroShotClassificationPipeline(t, session)
}

func TestZeroShotClassificationPipelineValidationORT(t *testing.T) {
	opts := []options.WithOption{options.WithOnnxLibraryPath(onnxRuntimeSharedLibrary)}
	session, err := NewORTSession(opts...)
	check(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		check(t, destroyErr)
	}(session)
	zeroShotClassificationPipelineValidation(t, session)
}

// Token classification

func TestTokenClassificationPipelineORT(t *testing.T) {
	opts := []options.WithOption{options.WithOnnxLibraryPath(onnxRuntimeSharedLibrary)}
	session, err := NewORTSession(opts...)
	check(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		check(t, destroyErr)
	}(session)
	tokenClassificationPipeline(t, session)
}

func TestTokenClassificationPipelineORTCuda(t *testing.T) {
	opts := []options.WithOption{
		options.WithOnnxLibraryPath("/usr/lib64/onnxruntime-gpu/libonnxruntime.so"),
		options.WithCuda(map[string]string{
			"device_id": "0",
		}),
	}
	session, err := NewORTSession(opts...)
	check(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		check(t, destroyErr)
	}(session)
	tokenClassificationPipeline(t, session)
}

func TestTokenClassificationPipelineValidationORT(t *testing.T) {
	opts := []options.WithOption{options.WithOnnxLibraryPath(onnxRuntimeSharedLibrary)}
	session, err := NewORTSession(opts...)
	check(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		check(t, destroyErr)
	}(session)
	tokenClassificationPipelineValidation(t, session)
}

// No Same Name

func TestNoSameNamePipelineORT(t *testing.T) {
	opts := []options.WithOption{options.WithOnnxLibraryPath(onnxRuntimeSharedLibrary)}
	session, err := NewORTSession(opts...)
	check(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		check(t, destroyErr)
	}(session)
	noSameNamePipeline(t, session)
}

// Benchmarks

func runBenchmarkEmbedding(strings *[]string, cuda bool) {
	var opts []options.WithOption
	switch cuda {
	case true:
		opts = []options.WithOption{
			options.WithOnnxLibraryPath("/usr/lib64/onnxruntime-gpu/libonnxruntime.so"),
			options.WithCuda(map[string]string{
				"device_id": "0",
			}),
		}
	default:
		opts = []options.WithOption{options.WithOnnxLibraryPath("/usr/lib64/onnxruntime.so")}
	}
	session, err := NewORTSession(opts...)
	if err != nil {
		panic(err)
	}

	defer func(session *Session) {
		errDestroy := session.Destroy()
		if errDestroy != nil {
			panic(errDestroy)
		}
	}(session)

	modelPath := "./models/KnightsAnalytics_all-MiniLM-L6-v2"
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
