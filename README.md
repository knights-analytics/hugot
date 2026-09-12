# <span>Hugot: ONNX Transformer Pipelines for Go

[![Go Reference](https://pkg.go.dev/badge/github.com/knights-analytics/hugot.svg)](https://pkg.go.dev/github.com/knights-analytics/hugot)
[![Go Report Card](https://goreportcard.com/badge/github.com/knights-analytics/hugot)](https://goreportcard.com/report/github.com/knights-analytics/hugot)
[![Coverage Status](https://coveralls.io/repos/github/knights-analytics/hugot/badge.svg?branch=main)](https://coveralls.io/github/knights-analytics/hugot?branch=main)

<div style="text-align:center">
<img src="./hugot.png" width="300" alt="Go Gopher Transformer">
</div>

## What

**TL;DR:** Hugot brings Hugging Face-style transformer pipelines to Go, letting you run and fine-tune ONNX models for text, vision, audio, and multimodal workloads directly inside your Go applications, with pluggable pure Go, ONNX Runtime, and OpenXLA backends.

The goal of this library is to provide an easy, scalable, and hassle-free way to run transformer pipelines inference and training in golang applications, such as Hugging Face 🤗 transformers pipelines. It is built on the following principles:

1. Hugging Face compatibility: models trained and tested using the python Hugging Face transformer library can be exported to onnx and used with the Hugot pipelines to obtain identical predictions as in the python version.
2. Hassle-free and performant production use: we exclusively support onnx models. Pytorch transformer models that don't have an onnx version can be easily exported to onnx via [Hugging Face Optimum](https://huggingface.co/docs/optimum/index), and used with the library.
3. Run on your hardware: this library is for those who want to run transformer models tightly coupled with their go applications, without the performance drawbacks of having to hit a rest API or the hassle of setting up and maintaining e.g. a python RPC service that talks to go.
4. Simplicity: the Hugot API allows you to easily deploy pipelines without having to write your own inference or training code. It also now includes a pure Go backend for minimal dependencies!

We support inference on CPU and on all accelerators supported by ONNX Runtime/OpenXLA. Note, however, that currently only CPU, TPU, and GPU inference on Nvidia GPUs via CUDA, are tested (see below).

IMPORTANT: The Go backend is designed for simpler workloads, environments that disallow cgo, and for smaller models such as [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It works best with small batches of roughly 32 inputs per call. If you have performance requirements, please move to a C backend such as XLA or ORT (detailed below).

Hugot loads and saves models in the ONNX format.

## Why

Developing and fine-tuning transformer models with the Hugging Face python library is great, but if your production stack is golang-based being able to reliably deploy and scale the resulting pytorch models can be challenging. This library aims to allow you to just lift-and-shift your python model and use the same Hugging Face pipelines you use for development for inference in a go application.

## For whom

For the golang developer or ML engineer who wants to run or fine-tune transformer pipelines on their own hardware and tightly coupled with their own application, without having to deal with writing their own inference or training code.

## By whom

Hugot is brought to you by the friendly folks at [Knights Analytics](https://knightsanalytics.com), who use Hugot in production to automate ai-powered data curation.

## Implemented pipelines

Currently, we have implementations for the following transformer pipelines:

- [audioClassification](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.AudioClassificationPipeline)
- [audioToAudio](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.AudioToAudioPipeline)
- [automaticSpeechRecognition](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.AutomaticSpeechRecognitionPipeline) (currently ORT only)
- [backgroundRemoval](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.BackgroundRemovalPipeline)
- [crossEncoder](https://huggingface.co/cross-encoder)
- [depthEstimation](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.DepthEstimationPipeline) (currently ORT only)
- [documentQuestionAnswering](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.DocumentQuestionAnsweringPipeline) (currently ORT only)
- [fillMask](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.FillMaskPipeline)
- [featureExtraction](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.FeatureExtractionPipeline)
- [imageClassification](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.ImageClassificationPipeline)
- [imageFeatureExtraction](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.ImageFeatureExtractionPipeline)
- [imageSegmentation](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.ImageSegmentationPipeline)
- [imageTextToText](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.ImageTextToTextPipeline) (currently ORT only)
- [imageToText](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.ImageToTextPipeline) (currently ORT only)
- [maskGeneration](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.MaskGenerationPipeline)
- [objectDetection](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.ObjectDetectionPipeline)
- [questionAnswering](https://huggingface.co/docs/transformers/tasks/question_answering)
- [tableQuestionAnswering](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.TableQuestionAnsweringPipeline) (currently ORT only)
- tabular (classic ML models such as decision trees, random forests etc) (currently ORT only)
- [textClassification](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.TextClassificationPipeline)
- [textGeneration](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.TextGenerationPipeline) (currently ORT only)
- [textToAudio](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.TextToAudioPipeline) (currently ORT only)
- [textToSpeech](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.TextToSpeechPipeline) (currently ORT only)
- [tokenClassification](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.TokenClassificationPipeline)
- [visualQuestionAnswering](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.VisualQuestionAnsweringPipeline) (currently ORT only)
- [zeroShotAudioClassification](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.ZeroShotAudioClassificationPipeline)
- [zeroShotClassification](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.ZeroShotClassificationPipeline)
- [zeroShotImageClassification](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.ZeroShotImageClassificationPipeline)
- [zeroShotObjectDetection](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.ZeroShotObjectDetectionPipeline) (currently ORT only)

## Installation and usage

### Choosing a backend

Hugot supports pluggable backends to perform the tokenization and run the ONNX models. Currently, we support the following backends:

- Go
- [Onnx Runtime](https://onnxruntime.ai/)
- [OpenXLA](https://openxla.org/)

The native Go backend is always available, as it does not require any cgo dependencies. It is provided by the [GoMLX](https://github.com/gomlx/gomlx) project.

Onnx Runtime can be included at compile time via the build tag "-tags ORT".  It is currently the fastest backend for inference and supports all pipelines, including generative pipelines such as text generation.

OpenXLA can be included at compile time via the build tag "-tags XLA". This is the only backend that supports TPUs. Note that it does not yet support generative pipelines or dynamic shapes.

CUDA requires a C backend, either OpenXLA or Onnx Runtime.

Once compiled, Hugot can be instantiated with your backend of choice via calling `NewGoSession()`, `NewXLASession()` or `NewORTSession()` respectively.

You may combine build tags "-tags XLA,ORT" or use "-tags ALL" to be able to use all available backends interchangeably.

### Usage

To use Hugot as a library in your application, you can directly import it and follow the example below.

#### Backends

- if using Onnx Runtime, the libonnxruntime.so file should be obtained from the releases section of this page. If you want to use other architectures than `linux/amd64` you will have to download it from [the ONNX Runtime releases page](https://github.com/microsoft/onnxruntime/releases/), see the [dockerfile](./Dockerfile) as an example. Hugot looks for this file at /usr/lib/libonnxruntime.so by default. A different location can be specified by passing the `WithOnnxLibraryPath()` option to `NewORTSession()`, e.g:

```go
session, err := NewORTSession(
    ctx,
    options.WithOnnxLibraryPath("/path/to/my/lib/directory"),
)
```

- if using XLA, the easiest way is to run "GOPROXY=direct go run github.com/gomlx/go-xla/cmd/pjrt_installer@latest -plugin=linux -version=v${GOPJRT_VERSION} -path=/usr/local/lib/go-xla", which will install the XLA backend provided by the [goMLX](https://github.com/gomlx/gomlx) project.

Alternatively, you can also use the [docker image](https://github.com/knights-analytics/hugot/pkgs/container/hugot) which has all the above dependencies already baked in.

- the latest versions of the onnxruntime and gopjrt libraries that are used in our testing can be found in the [bakefile](./docker-bake.hcl).

The library can be used as follows:

```go
package main

import (
    "github.com/knights-analytics/hugot"
    "context"
	"encoding/json"
    "fmt"
)

func check(err error) {
    if err != nil {
        panic(err.Error())
    }
}

func main() {
    // all sessions require context.Context
    ctx := context.Background()
    // start a new session
    session, err := hugot.NewGoSession(ctx)
	// For XLA (requires go build tags "XLA" or "ALL"):
	// session, err := hugot.NewXLASession(ctx)
	// For ORT (requires go build tags "ORT" or "ALL"):
	// session, err := hugot.NewORTSession(ctx)
	// This looks for the libonnxruntime.so library in its default path, e.g. /usr/lib
    // If your libonnxruntime.so is somewhere else, you can explicitly set it by using WithOnnxLibraryPath
    // session, err := hugot.NewORTSession(ctx, WithOnnxLibraryPath("/path/to/my/lib/directory"))
	check(err)
	
    // A successfully created hugot session needs to be destroyed when you're done
    defer func (session *hugot.Session) {
    err = session.Destroy()
    check(err)
    }(session)

    // Let's download an onnx sentiment test classification model in the current directory
    // note: if you compile your library with build flag NODOWNLOAD, this will exclude the downloader.
    // Useful in case you just want the core engine (because you already have the models)
    modelPath, err := hugot.DownloadModel("KnightsAnalytics/distilbert-base-uncased-finetuned-sst-2-english", "./models/", hugot.NewDownloadOptions())
    check(err)

    // We now create the configuration for the text classification pipeline we want to create.
    // Options to the pipeline can be set here using the Options field
    config := hugot.TextClassificationConfig{
        ModelPath: modelPath,
        Name:      "testPipeline",
    }
    // then we create out pipeline.
    // Note: the pipeline will also be added to the session object, so all pipelines can be destroyed at once
    sentimentPipeline, err := session.NewPipeline(config)
    check(err)

    // we can now use the pipeline for prediction on a batch of strings
    batch := []string{"This movie is disgustingly good !", "The director tried too much"}
    batchResult, err := sentimentPipeline.RunPipeline(ctx, batch)
    check(err)

    // and do whatever we want with it :)
    s, err := json.Marshal(batchResult)
    check(err)
    fmt.Println(string(s))
}
// OUTPUT: {"ClassificationOutputs":[[{"Label":"POSITIVE","Score":0.99031734}],[{"Label":"NEGATIVE","Score":0.963696}]]}
```

See also hugot_test.go for further examples for all pipelines.

## Generative models

Hugot uses the [Onnx Runtime Generative AI](https://onnxruntime.ai/generative-ai) backend to run generative models.

Generative models are used in a variety of text and multimodal pipelines. Please look at the [ORT text tests](tests/ort/hugot_ort_text_test.go) and [ORT multimodal tests](tests/ort/hugot_ort_multimodal_test.go) for examples of their usage.

To use the ORT Engine support for concurrent requests and inference batching (text-only messages), use the `WithGenerativeEngine()` option when creating a session.

## Hardware acceleration 🚀

Hugot now also supports the following accelerator backends for your inference:
 - CUDA (tested on Onnx Runtime and XLA). See below for setup instructions.
 - TPU (XLA only)
 - TensorRT (available in Onnx Runtime only)
 - DirectML (available in Onnx Runtime only)
 - CoreML (available in Onnx Runtime only)
 - OpenVINO (available in Onnx Runtime only)

Please provide feedback if encountering any issues with the accelerators above!

To use Hugot with Nvidia gpu acceleration, you need to have the following:

- The Nvidia driver for your graphics card (if running in Docker and WSL2, starting with --gpus all should inherit the drivers from the host OS)
- ONNX Runtime:
    - The cuda gpu version of ONNX Runtime on the machine/docker container. You can see how we get that by looking at the [Dockerfile](./Dockerfile). You can also get the ONNX Runtime libraries that we use for testing from the release. Just download the gpu .so libraries and put them in /usr/lib.
    - The required CUDA libraries installed on your system that are compatible with the ONNX Runtime gpu version you use. See [here](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html). For instance, for onnxruntime 1.28.0, we need CUDA 13.x (any minor version should be compatible) and cuDNN 9.x.
    - Start a session with the following:
      ```go
      ctx := context.Background()
      opts := []options.WithOption{
        options.WithCuda(map[string]string{
          "device_id": "0",
        }),
      }
      session, err := NewORTSession(ctx, opts...)
      ```
- OpenXLA
    - Install CUDA support via the command `GOPROXY=direct go run github.com/gomlx/go-xla/cmd/pjrt_installer@latest -plugin=cuda13 -version=${JAX_CUDA_VERSION} -path=/usr/local/lib/go-xla`
    - Start a session with the following:
      ```go
      ctx := context.Background()
      opts := []options.WithOption{
        options.WithCuda(map[string]string{
          "device_id": "0",
        }),
      }
      session, err := NewXLASession(ctx, opts...)
      ```

For the ONNX Runtime Cuda libraries, you can install CUDA 13.x by installing the full cuda toolkit, but that's quite a big package. In our testing on awslinux/fedora, we have been able to limit the libraries needed to run Hugot with Nvidia gpu acceleration to just these:

- cuda-cudart-13-3 libcublas-13-3 libcurand-13-3 libcufft-13-3 libcudnn9-cuda-13

libcufft and libcudnn9 are lazy loaded when needed, so may be skippable depending on the models you load.

On different distros (e.g. Ubuntu), you should be able to install the equivalent packages.

## Training and fine-tuning pipelines 

Hugot now also supports the training and fine-tuning of transformer pipelines! Training always runs through [goMLX](https://github.com/gomlx/gomlx), on any backend:
the onnx model is loaded into goMLX, fine-tuned, and serialized back to onnx format.

Go and XLA sessions (`hugot.NewGoSession`, `hugot.NewXLASession`) already run on goMLX, so they can train as they are, with no
extra options. Only an ORT session needs something more: it has to be created with `options.WithGoMLX()` to train. Note that this
changes the whole session, not just the trainer: every pipeline in a `WithGoMLX` session runs through goMLX rather than native
ONNX Runtime. If you also want native ORT inference, use a separate session for training.

A trainer always loads its own copy of the model, so training never changes the weights of a pipeline you are serving, even
in the same session. To serve the fine-tuned model, save it with `trainer.Save` and load the saved path like any other model.

This is currently supported only for the **FeatureExtractionPipeline**. This can be used to fine-tune the vector embeddings for e.g. semantic textual similarity (for applications like RAG and semantic search). In order to fine-tune the feature extraction pipeline for semantic search you will need to collect a training dataset in the following format:

```js
{"sentence1": "The quick brown fox jumps over the lazy dog", "sentence2": "A quick brown fox jumps over a lazy dog", "score": 1}
{"sentence1": "The quick brown fox jumps over the lazy dog", "sentence2": "A quick brown cow jumps over a lazy caterpillar", "score": 0.5}
```

See the [example](testcases/semanticSimilarityTest.jsonl) for a sample dataset.

The score is assumed to be a float between 0 and 1 that encodes the semantic similarity between the sentences, and by default a cosine similarity loss is used (see [sentence transformers](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss)). You can specify a different optimizer or loss function from `goMLX` using the `GOMLXOptions` field of `TrainerConfig`.

A trainer is created from a session, in the same way as a pipeline:

```go
// Training runs through goMLX on every backend. A Go or XLA session needs nothing extra;
// an ORT session must be created with options.WithGoMLX().
// NewXLASession requires the go build tags "XLA" or "ALL".
session, err := hugot.NewXLASession(ctx)
// To train on an Nvidia GPU, enable CUDA on the session:
// session, err := hugot.NewXLASession(ctx, options.WithCuda(nil))
check(err)
defer func() { _ = session.Destroy() }()

dataset, err := datasets.NewSemanticSimilarityDataset(ctx, "dataset.jsonl", 32, nil, nil)
check(err)

trainer, err := session.NewTrainer(
    hugot.TrainerConfig[*pipelines.FeatureExtractionPipeline]{
        ModelPath:    modelPath,
        TrainDataset: dataset,
        Verbose:      true,
    },
    hugot.WithEpochs(2),
)
check(err)
defer func() { _ = trainer.Close() }()

check(trainer.Train(ctx))
check(trainer.Save(ctx, outputPath))
```

Besides `WithEpochs` (default 100), the following trainer options are available:

- `WithEarlyStopping()` / `WithEarlyStoppingParams(patience, tolerance)`: stop when the loss on `TrainerConfig.EvalDataset` stops improving (defaults: patience 3, tolerance 1e-4). Requires `EvalDataset` to be set.
- `WithFreezeLayers(layers)`: freeze the given transformer layers (0 is the first); `[]int{-1}` freezes every layer except the last.
- `WithFreezeEmbeddings()`: freeze the embedding layers.

Set `TrainerConfig.TrainEvalDataset` to record a per-epoch training loss, available from `trainer.Statistics()` and written to `statistics.txt` on save.

`trainer.Close()` releases the trainer's copy of the model once you have saved it; a trainer that is never closed is released by `session.Destroy()`, so in a long-lived session that trains repeatedly, close each trainer when you are done with it. The fine-tuned model is written back as onnx, together with the tokenizer files and a `statistics.txt` holding the per-epoch losses, so the output directory can be loaded straight back into a pipeline.

Note that training on GPU is currently much faster and memory efficient than training on CPU, although optimizations are underway. On CPU, we recommend smaller batch sizes.

See [the training tests](tests/training/hugot_training_test.go) for an example on how to fine-tune semantic similarity starting with an open source sentence transformers model and a few examples.

## Performance Tuning

Firstly, the throughput depends largely on the size of the input requests. The best batch size is affected by the number of tokens per input, but we find batches of roughly 32 inputs per call to be a good starting point.

### ONNX Runtime
The library defaults to ONNX Runtime's default tuning settings. These are optimised for latency over throughput, and will attempt to parallelize single threaded calls to ONNX Runtime over multiple cores.

For maximum throughput, it is best to call a single shared Hugot pipeline from multiple goroutines (1 per core), using a channel to pass the input data. In this scenario, the following settings will greatly increase inference throughput.

```go
session, err := hugot.NewORTSession(
	context.Background(),
	hugot.WithInterOpNumThreads(1),
	hugot.WithIntraOpNumThreads(1),
	hugot.WithCpuMemArena(false),
	hugot.WithMemPattern(false),
)
```

InterOpNumThreads and IntraOpNumThreads constricts each goroutine's call to a single core, greatly reducing locking and cache penalties. Disabling CpuMemArena and MemPattern skips pre-allocation of some memory structures, increasing latency, but also throughput efficiency.

## File Systems
Hugot uses the standard operating system filesystem by default. File operations are defined by `fileutil.FileSystem`, which can be replaced with an adapter for an object store or another filesystem. Pass the adapter with `options.WithFileSystem` when creating a session; the adapter is scoped to that session and can be used safely by concurrent sessions:
```go
session, err := hugot.NewGoSession(
    context.Background(),
    options.WithFileSystem(myFileSystemAdapter),
)
if err != nil {
    return err
}
defer session.Destroy()
```

The adapter must implement the operations in `util/fileutil/file.go` (`OpenFile`, `CopyFile`, `Walk`, `DeleteFile`, `FileExists`, `FileStats`, and `NewFileWriter`). This keeps Hugot independent of storage providers while allowing integrations such as `afs` or `gocloud` to translate those operations to their own APIs.

## Limitations

Apart from the fact that only the aforementioned pipelines are currently implemented, the current limitations are:

- the library is only built/tested on amd64-linux currently.

## Contributing

If you would like to contribute to Hugot, please see the [contribution guidelines](./contrib.md).
