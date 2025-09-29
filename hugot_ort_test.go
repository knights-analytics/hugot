//go:build (ORT || ALL) && !TRAINING

package hugot

import (
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/pipelineBackends"
	"github.com/knights-analytics/hugot/pipelines"
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
		options.WithCpuMemArena(true),
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
		options.WithCpuMemArena(true),
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

// Cross Encoder
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

// Image classification
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
	// if os.Getenv("CI") != "" {
	// 	t.SkipNow()
	// }
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

// TestTextGenerationLongPromptCUDA tests a long prompt with the Phi-3.5-mini-instruct-onnx model on GPU.
func TestTextGenerationLongPromptCUDA(t *testing.T) {
	// if os.Getenv("CI") != "" {
	// 	t.SkipNow()
	// }
	opts := []options.WithOption{
		options.WithOnnxLibraryPath("/usr/lib64/onnxruntime-gpu/libonnxruntime.so"),
		options.WithCuda(map[string]string{"device_id": "0"}),
	}
	session, err := NewORTSession(opts...)
	if err != nil {
		t.Fatalf("session init failed: %v", err)
	}
	defer func() { _ = session.Destroy() }()

	system := `You are an assistant that helps summarise json documents. For a list of json documents, provide a concise summary of the key details and differences.`

	// config := TextGenerationConfig{
	// 	ModelPath:    "./models/onnx-community_Qwen2.5-1.5B",
	// 	Name:         "qwen-long-prompt-gpu",
	// 	OnnxFilename: "model_quantized.onnx",
	// 	Options: []pipelineBackends.PipelineOption[*pipelines.TextGenerationPipeline]{
	// 		pipelines.WithMaxTokens(256),
	// 		pipelines.WithQwenTemplate(),
	// 		// Stop on <|endoftext|> (151643) or <|im_end|> (151645)
	// 		pipelines.WithCustomStopTokens([]int64{151643, 151645}),
	// 	},
	// }

	config := TextGenerationConfig{
		ModelPath:    "./models/KnightsAnalytics_Phi-3.5-mini-instruct-onnx",
		Name:         "long-prompt-test-gpu",
		OnnxFilename: "model.onnx",
		Options: []pipelineBackends.PipelineOption[*pipelines.TextGenerationPipeline]{
			pipelines.WithMaxTokens(500),
			pipelines.WithPhiTemplate(),
			pipelines.WithCustomStopTokens([]int64{32007}),
		},
	}
	textGenPipeline, err := NewPipeline(session, config)
	if err != nil {
		t.Fatalf("pipeline init failed: %v", err)
	}

	msg := [][]pipelines.Message{{
		{Role: "system", Content: system},
		{Role: "user", Content: `{"a":"hi"},{"b":"hi"},{"c":"hi"}`},
	}}

	fmt.Println("Warming generation...")
	_, err = textGenPipeline.RunWithTemplate(msg)
	if err != nil {
		t.Fatalf("generation failed: %v", err)
	}
	fmt.Println("Generation warmed...")

	start := time.Now()
	fmt.Println("Starting generation...")
	msg = [][]pipelines.Message{{
		{Role: "system", Content: system},
		{Role: "user", Content: longTestPrompt},
	}}
	out, err := textGenPipeline.RunWithTemplate(msg)
	if err != nil {
		t.Fatalf("generation failed: %v", err)
	}

	fmt.Println("Generation completed.")
	wall := time.Since(start)
	genOut, ok := out.(*pipelines.TextGenerationOutput)
	if !ok || len(genOut.GeneratedTokens) == 0 {
		t.Fatalf("unexpected output type or empty tokens")
	}
	stats := textGenPipeline.GetStats()
	fmt.Printf("WallTime=%s\n", wall)
	fmt.Printf("generated output: %s\n", genOut.GetOutput()...)
	for _, s := range stats {
		fmt.Println(s)
	}
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

const longTestPrompt = `
	Summarise this list of jsons:

	{
    "_id": "68c19f301b7efa6a20ad1184",
    "index": 0,
    "guid": "ecee3e65-b8b3-4377-9948-b52d861455b0",
    "isActive": false,
    "balance": "$1,257.94",
    "picture": "http://placehold.it/32x32",
    "age": 37,
    "eyeColor": "brown",
    "name": "Mcfarland Coleman",
    "gender": "male",
    "company": "KOFFEE",
    "email": "mcfarlandcoleman@koffee.com",
    "phone": "+1 (949) 541-2357",
    "address": "209 Livonia Avenue, Faxon, Ohio, 8074",
    "about": "Aute est fugiat quis officia cillum tempor duis amet tempor sunt ad duis ut ea. Ex tempor aliqua aute quis labore labore dolore duis consequat deserunt. Eiusmod tempor culpa cillum nulla consectetur duis deserunt quis voluptate dolore incididunt eiusmod.\r\n",
    "registered": "2016-04-09T09:01:23 -02:00",
    "latitude": 23.201234,
    "longitude": -120.665901,
    "tags": [
      "ex",
      "et",
      "exercitation",
      "irure",
      "nisi",
      "minim",
      "minim"
    ],
    "friends": [
      {
        "id": 0,
        "name": "Dorothea Kelly"
      },
      {
        "id": 1,
        "name": "Sally Espinoza"
      },
      {
        "id": 2,
        "name": "Whitney Wolfe"
      }
    ],
    "greeting": "Hello, Mcfarland Coleman! You have 1 unread messages.",
    "favoriteFruit": "banana"
  },
  {
    "_id": "68c19f30bb8a142738db1b4a",
    "index": 1,
    "guid": "aa666967-b69f-4a40-bb13-c1c1140036dc",
    "isActive": true,
    "balance": "$2,321.80",
    "picture": "http://placehold.it/32x32",
    "age": 20,
    "eyeColor": "brown",
    "name": "Daniels Webster",
    "gender": "male",
    "company": "CALLFLEX",
    "email": "danielswebster@callflex.com",
    "phone": "+1 (918) 488-3620",
    "address": "995 Lawrence Avenue, Harold, Illinois, 3159",
    "about": "Qui non proident minim do ad cillum eu mollit excepteur est laboris in incididunt. Incididunt adipisicing eu Lorem minim irure fugiat exercitation ullamco proident occaecat. Fugiat anim reprehenderit irure ex officia ad.\r\n",
    "registered": "2017-11-26T05:46:21 -01:00",
    "latitude": 2.345942,
    "longitude": 153.228303,
    "tags": [
      "non",
      "sint",
      "nulla",
      "aliqua",
      "laborum",
      "in",
      "esse"
    ],
    "friends": [
      {
        "id": 0,
        "name": "Juliet Leonard"
      },
      {
        "id": 1,
        "name": "Melva Waters"
      },
      {
        "id": 2,
        "name": "Margarita Clark"
      }
    ],
    "greeting": "Hello, Daniels Webster! You have 5 unread messages.",
    "favoriteFruit": "banana"
  },
  {
    "_id": "68c19f30397bc7dadecfa155",
    "index": 2,
    "guid": "2d1d915c-327a-4c16-a4b3-84474dbbd391",
    "isActive": false,
    "balance": "$1,714.61",
    "picture": "http://placehold.it/32x32",
    "age": 31,
    "eyeColor": "green",
    "name": "Laverne Bean",
    "gender": "female",
    "company": "FISHLAND",
    "email": "lavernebean@fishland.com",
    "phone": "+1 (816) 444-2065",
    "address": "227 Holt Court, Russellville, Federated States Of Micronesia, 8054",
    "about": "Ea cupidatat occaecat consectetur quis Lorem quis sint duis. Do veniam aute cillum elit sit culpa amet sint ut magna incididunt eiusmod eiusmod minim. Cillum esse nulla nulla nisi laboris magna dolor.\r\n",
    "registered": "2019-05-24T05:10:44 -02:00",
    "latitude": 63.682722,
    "longitude": 167.857033,
    "tags": [
      "voluptate",
      "velit",
      "ipsum",
      "ipsum",
      "do",
      "consectetur",
      "cupidatat"
    ],
    "friends": [
      {
        "id": 0,
        "name": "Leila Marquez"
      },
      {
        "id": 1,
        "name": "Margery Valdez"
      },
      {
        "id": 2,
        "name": "Liz Salazar"
      }
    ],
    "greeting": "Hello, Laverne Bean! You have 2 unread messages.",
    "favoriteFruit": "banana"
  },
  {
    "_id": "68c19f307fd1b9d20a4660c3",
    "index": 3,
    "guid": "aa4c6e09-21b8-4a99-a8fe-7650dd16576c",
    "isActive": false,
    "balance": "$2,378.44",
    "picture": "http://placehold.it/32x32",
    "age": 40,
    "eyeColor": "green",
    "name": "Elisabeth Galloway",
    "gender": "female",
    "company": "JIMBIES",
    "email": "elisabethgalloway@jimbies.com",
    "phone": "+1 (893) 576-2684",
    "address": "226 Seigel Court, Greenock, Washington, 2699",
    "about": "Et cupidatat nostrud veniam culpa nostrud aliquip consequat qui enim. Et reprehenderit sit est duis elit. Occaecat aute consectetur tempor reprehenderit incididunt id nisi commodo quis exercitation consequat aliquip reprehenderit incididunt. Esse magna aliquip nulla in magna.\r\n",
    "registered": "2016-03-12T11:51:43 -01:00",
    "latitude": -9.0195,
    "longitude": 23.194577,
    "tags": [
      "do",
      "ad",
      "veniam",
      "consequat",
      "ipsum",
      "tempor",
      "occaecat"
    ],
    "friends": [
      {
        "id": 0,
        "name": "Trudy Bray"
      },
      {
        "id": 1,
        "name": "Crane Spence"
      },
      {
        "id": 2,
        "name": "Mullen Solis"
      }
    ],
    "greeting": "Hello, Elisabeth Galloway! You have 8 unread messages.",
    "favoriteFruit": "apple"
  },
  {
    "_id": "68c19f30c03d508648be1d07",
    "index": 4,
    "guid": "c6f6c4cc-8fce-4cca-96c5-483cdfd2deec",
    "isActive": false,
    "balance": "$3,110.34",
    "picture": "http://placehold.it/32x32",
    "age": 33,
    "eyeColor": "blue",
    "name": "Lee Hall",
    "gender": "male",
    "company": "LOVEPAD",
    "email": "leehall@lovepad.com",
    "phone": "+1 (876) 422-3809",
    "address": "905 Broadway , Albrightsville, Puerto Rico, 8907",
    "about": "Nostrud aute ea pariatur labore aliqua dolor enim aliqua nulla ullamco enim. Qui eu aliqua anim sunt non ea nisi enim aliquip eu aliquip duis consequat quis. Commodo ullamco sit aute officia laborum esse cillum ex consequat nostrud. Ex commodo exercitation minim aliquip quis fugiat Lorem ullamco commodo. Consectetur in culpa ut ex amet mollit ut dolor cupidatat. Esse irure tempor qui qui eiusmod.\r\n",
    "registered": "2024-09-08T07:18:31 -02:00",
    "latitude": 74.955239,
    "longitude": -173.268385,
    "tags": [
      "magna",
      "non",
      "pariatur",
      "nulla",
      "adipisicing",
      "commodo",
      "velit"
    ],
    "friends": [
      {
        "id": 0,
        "name": "Frost Schroeder"
      },
      {
        "id": 1,
        "name": "Josefa Buck"
      },
      {
        "id": 2,
        "name": "Mann Hill"
      }
    ],
    "greeting": "Hello, Lee Hall! You have 10 unread messages.",
    "favoriteFruit": "strawberry"
  }
	`
