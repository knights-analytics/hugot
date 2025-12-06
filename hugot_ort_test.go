//go:build (ORT || ALL) && !TRAINING

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
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	featureExtractionPipeline(t, session)
}

func TestFeatureExtractionPipelineORTCuda(t *testing.T) {
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
	featureExtractionPipeline(t, session)
}

func TestFeatureExtractionPipelineValidationORT(t *testing.T) {
	opts := []options.WithOption{options.WithOnnxLibraryPath(onnxRuntimeSharedLibrary)}
	session, err := NewORTSession(opts...)
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	featureExtractionPipelineValidation(t, session)
}

// Text classification

func TestTextClassificationPipelineORT(t *testing.T) {
	opts := []options.WithOption{
		options.WithOnnxLibraryPath(onnxRuntimeSharedLibrary),
		options.WithTelemetry(),
		options.WithCPUMemArena(true),
		options.WithMemPattern(true),
		options.WithIntraOpNumThreads(1),
		options.WithInterOpNumThreads(1),
	}
	session, err := NewORTSession(opts...)
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	textClassificationPipeline(t, session)
}

func TestTextClassificationPipelineORTCuda(t *testing.T) {
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
	textClassificationPipeline(t, session)
}

func TestTextClassificationPipelineMultiORT(t *testing.T) {
	opts := []options.WithOption{
		options.WithOnnxLibraryPath(onnxRuntimeSharedLibrary),
		options.WithTelemetry(),
		options.WithCPUMemArena(true),
		options.WithMemPattern(true),
		options.WithIntraOpNumThreads(1),
		options.WithInterOpNumThreads(1),
	}
	session, err := NewORTSession(opts...)
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	textClassificationPipelineMulti(t, session)
}

func TestTextClassificationPipelineORTMultiCuda(t *testing.T) {
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
	textClassificationPipelineMulti(t, session)
}

func TestTextClassificationPipelineValidationORT(t *testing.T) {
	opts := []options.WithOption{options.WithOnnxLibraryPath(onnxRuntimeSharedLibrary)}
	session, err := NewORTSession(opts...)
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	textClassificationPipelineValidation(t, session)
}

// Token classification

func TestTokenClassificationPipelineORT(t *testing.T) {
	opts := []options.WithOption{options.WithOnnxLibraryPath(onnxRuntimeSharedLibrary)}
	session, err := NewORTSession(opts...)
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	tokenClassificationPipeline(t, session)
}

func TestTokenClassificationPipelineORTCuda(t *testing.T) {
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
	tokenClassificationPipeline(t, session)
}

func TestTokenClassificationPipelineValidationORT(t *testing.T) {
	opts := []options.WithOption{options.WithOnnxLibraryPath(onnxRuntimeSharedLibrary)}
	session, err := NewORTSession(opts...)
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	tokenClassificationPipelineValidation(t, session)
}

// Zero shot

func TestZeroShotClassificationPipelineORT(t *testing.T) {
	opts := []options.WithOption{options.WithOnnxLibraryPath(onnxRuntimeSharedLibrary)}
	session, err := NewORTSession(opts...)
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	zeroShotClassificationPipeline(t, session)
}

func TestZeroShotClassificationPipelineORTCuda(t *testing.T) {
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
	zeroShotClassificationPipeline(t, session)
}

func TestZeroShotClassificationPipelineValidationORT(t *testing.T) {
	opts := []options.WithOption{options.WithOnnxLibraryPath(onnxRuntimeSharedLibrary)}
	session, err := NewORTSession(opts...)
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	zeroShotClassificationPipelineValidation(t, session)
}

// Cross Encoder.
func TestCrossEncoderPipelineORT(t *testing.T) {
	opts := []options.WithOption{options.WithOnnxLibraryPath(onnxRuntimeSharedLibrary)}
	session, err := NewORTSession(opts...)
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	crossEncoderPipeline(t, session)
}

func TestCrossEncoderPipelineORTCuda(t *testing.T) {
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
	crossEncoderPipeline(t, session)
}

func TestCrossEncoderPipelineValidationORT(t *testing.T) {
	opts := []options.WithOption{options.WithOnnxLibraryPath(onnxRuntimeSharedLibrary)}
	session, err := NewORTSession(opts...)
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	crossEncoderPipelineValidation(t, session)
}

// Image classification.
func TestImageClassificationPipelineORT(t *testing.T) {
	opts := []options.WithOption{options.WithOnnxLibraryPath(onnxRuntimeSharedLibrary)}
	session, err := NewORTSession(opts...)
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	imageClassificationPipeline(t, session)
}

func TestObjectDetectionPipelineORT(t *testing.T) {
	opts := []options.WithOption{options.WithOnnxLibraryPath(onnxRuntimeSharedLibrary)}
	session, err := NewORTSession(opts...)
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	objectDetectionPipeline(t, session)
}

func TestImageClassificationPipelineORTCuda(t *testing.T) {
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
	imageClassificationPipeline(t, session)
}

func TestImageClassificationPipelineValidationORT(t *testing.T) {
	opts := []options.WithOption{options.WithOnnxLibraryPath(onnxRuntimeSharedLibrary)}
	session, err := NewORTSession(opts...)
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	imageClassificationPipelineValidation(t, session)
}

// Text generation

func TestTextGenerationPipelineORT(t *testing.T) {
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
	textGenerationPipeline(t, session)
}

func TestTextGenerationPipelineORTCuda(t *testing.T) {
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
	textGenerationPipeline(t, session)
}

func TestTextGenerationPipelineValidationORT(t *testing.T) {
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
	textGenerationPipelineValidation(t, session)
}

// No Same Name

func TestNoSameNamePipelineORT(t *testing.T) {
	opts := []options.WithOption{options.WithOnnxLibraryPath(onnxRuntimeSharedLibrary)}
	session, err := NewORTSession(opts...)
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	noSameNamePipeline(t, session)
}

func TestClosePipelineORT(t *testing.T) {
	opts := []options.WithOption{options.WithOnnxLibraryPath(onnxRuntimeSharedLibrary)}
	session, err := NewORTSession(opts...)
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	destroyPipelines(t, session)
}

// Thread safety

func TestThreadSafetyORT(t *testing.T) {
	opts := []options.WithOption{options.WithOnnxLibraryPath(onnxRuntimeSharedLibrary)}
	session, err := NewORTSession(opts...)
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	threadSafety(t, session, 250)
}

func TestThreadSafetyORTCuda(t *testing.T) {
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
	threadSafety(t, session, 1000)
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

func BenchmarkORTCudaEmbedding(b *testing.B) {
	if os.Getenv("CI") != "" {
		b.SkipNow()
	}
	p := make([]string, 30000)
	for i := range 30000 {
		p[i] = "The goal of this library is to provide an easy, scalable, and hassle-free way to run huggingface transformer pipelines in golang applications."
	}
	for b.Loop() {
		runBenchmarkEmbedding(&p, true)
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
		runBenchmarkEmbedding(&p, false)
	}
}
