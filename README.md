# <span>Hugot: ONNX Transformer Pipelines for Go

[![Go Reference](https://pkg.go.dev/badge/github.com/knights-analytics/hugot.svg)](https://pkg.go.dev/github.com/knights-analytics/hugot)
[![Go Report Card](https://goreportcard.com/badge/github.com/knights-analytics/hugot)](https://goreportcard.com/report/github.com/knights-analytics/hugot)
[![Coverage Status](https://coveralls.io/repos/github/knights-analytics/hugot/badge.svg?branch=main)](https://coveralls.io/github/knights-analytics/hugot?branch=main)

<div style="text-align:center">
<img src="./hugot.png" width="300" alt="Go Gopher Transformer">
</div>

## What

TL;DR: AI use-cases such as embeddings, text generation, image classification, entity recognition, fine-tuning, and more, natively running in Go!

The goal of this library is to provide an easy, scalable, and hassle-free way to run transformer pipelines inference and training in golang applications, such as Hugging Face 🤗 transformers pipelines. It is built on the following principles:

1. Hugging Face compatibility: models trained and tested using the python Hugging Face transformer library can be exported to onnx and used with the hugot pipelines to obtain identical predictions as in the python version.
2. Hassle-free and performant production use: we exclusively support onnx models. Pytorch transformer models that don't have an onnx version can be easily exported to onnx via [Hugging Face Optimum](https://huggingface.co/docs/optimum/index), and used with the library.
3. Run on your hardware: this library is for those who want to run transformer models tightly coupled with their go applications, without the performance drawbacks of having to hit a rest API or the hassle of setting up and maintaining e.g. a python RPC service that talks to go.
4. Simplicity: the hugot api allows you to easily deploy pipelines without having to write your own inference or training code. It also now includes a pure Go backend for minimal dependencies!

We support inference on CPU and on all accelerators supported by ONNX Runtime/OpenXLA. Note, however, that currently only CPU, and GPU inference on Nvidia GPUs via CUDA, are tested (see below).

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

- [featureExtraction](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.FeatureExtractionPipeline)
- [textClassification](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.TextClassificationPipeline)
- [tokenClassification](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.TokenClassificationPipeline)
- [zeroShotClassification](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.ZeroShotClassificationPipeline)
- [textGeneration](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.TextGenerationPipeline)
- [crossEncoder](https://huggingface.co/cross-encoder)
- [imageClassification](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.ImageClassificationPipeline)

Implementations for additional pipelines will follow. We also very gladly accept PRs to expand the set of pipelines! See [here](https://huggingface.co/docs/transformers/en/main_classes/pipelines) for the missing pipelines that can be implemented, and the contributing section below if you want to lend a hand.

Hugot can be used both as a library and as a command-line application. See below for usage instructions.

## Installation and usage

Hugot can be used in two ways: as a library in your go application, or as a command-line binary.

### Choosing a backend

Hugot supports pluggable backends to perform the tokenization and run the ONNX models. Currently, we support the following backends:

- (default) native go (provided by [GoMLX](https://github.com/gomlx/gomlx))
- [Onnx Runtime](https://onnxruntime.ai/)
- [OpenXLA](https://openxla.org/)

Onnx Runtime can also be selected as a backend via the build tag "-tags ORT". It does not support training, but it is currently the fastest backend for CPU inference and supports
all pipelines.

OpenXLA can be included at compile time via the build tag "-tags XLA". This is required for fine-tuning of e.g. embedding models. Note that it does not yet support generative pipelines.

CUDA requires a C backend, either OpenXLA or Onnx Runtime.

Once compiled, Hugot can be instantiated with your backend of choice via calling `NewGoSession()`, `NewXLASession()` or `NewORTSession()` respectively.

You may combine build tags "-tags XLA,ORT" or use "-tags ALL" to be able to use all available backends interchangeably.

### Use it as a library

To use Hugot as a library in your application, you can directly import it and follow the example below.

#### Backends

- if using Onnx Runtime, the onnxruntime.so file should be obtained from the releases section of this page. If you want to use alternative architectures from `linux/amd64` you will have to download it from [the ONNX Runtime releases page](https://github.com/microsoft/onnxruntime/releases/), see the [dockerfile](./Dockerfile) as an example. Hugot looks for this file at /usr/lib/onnxruntime.so or /usr/lib64/onnxruntime.so by default. A different location can be specified by passing the `WithOnnxLibraryPath()` option to `NewORTSession()`, e.g:

```
session, err := NewORTSession(
    options.WithOnnxLibraryPath("/path/to/onnxruntime.so"),
)
```

- if using XLA, the easiest way is to run "curl -sSf https://raw.githubusercontent.com/gomlx/gopjrt/main/cmd/install_linux_amd64.sh | bash", which will install the XLA backend provided by the [goMLX](https://github.com/gomlx/gomlx) project.

- if using XLA or ORT, you will also need to use the rust-based tokenizer. The tokenizers.a file can be obtained from the releases section of this page (if you want to use alternative architecture from `linux/amd64` you will have to build the tokenizers.a yourself, see [here](https://github.com/daulet/tokenizers)). This file should be at /usr/lib/tokenizers.a so that Hugot can load it. Alternatively, you can explicitly specify the path to the folder with the `libtokenizers.a` file using the `CGO_LDFLAGS` env variable, see the [dockerfile](./Dockerfile). The tokenizer is statically linked at build time.

Alternatively, you can also use the [docker image](https://github.com/knights-analytics/hugot/pkgs/container/hugot) which has all the above dependencies already baked in.

The library can be used as follows:

```go
package main

import (
    "github.com/knights-analytics/hugot"
    "encoding/json"
    "fmt"
)

func check(err error) {
    if err != nil {
        panic(err.Error())
    }
}

func main() {
    // start a new session
    session, err := hugot.NewGoSession()
	// For XLA (requires go build tags "XLA" or "ALL"):
	// session, err := hugot.NewXLASession()
	// For ORT (requires go build tags "ORT" or "ALL"):
	// session, err := hugot.NewORTSession()
	// This looks for the onnxruntime.so library in its default path, e.g. /usr/lib/onnxruntime.so
    // If your onnxruntime.so is somewhere else, you can explicitly set it by using WithOnnxLibraryPath
    // session, err := hugot.NewORTSession(WithOnnxLibraryPath("/path/to/onnxruntime.so"))
	check(err)
	
    // A successfully created hugot session needs to be destroyed when you're done
    defer func (session *hugot.Session) {
    err := session.Destroy()
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
    sentimentPipeline, err := hugot.NewPipeline(session, config)
    check(err)

    // we can now use the pipeline for prediction on a batch of strings
    batch := []string{"This movie is disgustingly good !", "The director tried too much"}
    batchResult, err := sentimentPipeline.RunPipeline(batch)
    check(err)

    // and do whatever we want with it :)
    s, err := json.Marshal(batchResult)
    check(err)
    fmt.Println(string(s))
}
// OUTPUT: {"ClassificationOutputs":[[{"Label":"POSITIVE","Score":0.99031734}],[{"Label":"NEGATIVE","Score":0.963696}]]}
```

See also hugot_test.go for further examples for all pipelines.

### Use it as a cli: Hugging Face 🤗 pipelines from the command line

Note: the cli is currently only built and tested on amd64-linux using the ONNX Runtime backend.

With Hugot you don't need python, pytorch, or even go to run Hugging Face transformers. Simply install the Hugot cli (alpha):

```
curl https://raw.githubusercontent.com/knights-analytics/hugot/main/scripts/install-hugot-cli.sh | bash
```

This will install the Hugot binary at $HOME/.local/bin/hugot, and the corresponding onnxruntime.so library at $HOME/lib/hugot/onnxruntime.so.
The if $HOME/.local/bin is on your $PATH, you can do:

```
hugot run --model=KnightsAnalytics/distilbert-base-uncased-finetuned-sst-2-english --input=/path/to/input.jsonl --output=/path/to/folder/output --type=textClassification
```

Hugot will load the model, process the input, and write the results in the output folder.
Note that the Hugot cli currently expects the input in a specific format: json lines with an "input" key containing the string to process.
Example:

```
{"input": "The director tried too much"}
{"input": "The film was excellent"}
```

Will produce a file called result_0.jsonl in the output folder with contents:

```
{"input":"The director tried too much","output":[{"Label":"NEGATIVE","Score":0.99752176}]}
{"input":"The film was excellent","output":[{"Label":"POSITIVE","Score":0.99986285}]}
```

Note that if --input is not provided, Hugot will read from stdin, and if --output is not provided, it will write to stdout.
This allows to chain things like:

```
echo '{"input":"The director tried too much"}' | hugot run --model=/path/to/model --type=textClassification | jq
```

To be able to run transformers fully from the command line.

Note that the --model parameter can be:
    1. the full path to a model to load
    2. the name of a Hugging Face model. Hugot will first try to look for the model at $HOME/hugot, or will try to download the model from Hugging Face.

## Generative models

The TextGenerationPipeline provides generative text inference using ONNX models. It is currently only supported with the ORT backend. We tested the pipeline with the
following models:
 
- **Gemma Family**: `onnx-community/gemma-3-1b-it-ONNX`, `onnx-community/gemma-3-270m-it-ONNX`
- **Phi Family**: `microsoft/Phi-3-mini-4k-instruct-onnx`, `microsoft/Phi-3.5-mini-instruct-onnx`

Generative models typically use external weights, so use the downloadOptions.ExternalDataPath option when downloading the model. See the [example](./testData/downloadModels.go ) here.
 
### Example Usage
````go
session, err := NewORTSession()
check(err)
 
defer func(session *Session) {
    err := session.Destroy()
    check(err)
}(session)
 
config := TextGenerationConfig{
    ModelPath:    "./models/KnightsAnalytics_Phi-3.5-mini-instruct-onnx",
    Name:         "testPipeline",
    OnnxFilename: "model.onnx",
    Options: []pipelineBackends.PipelineOption[*pipelines.TextGenerationPipeline]{
        pipelines.WithMaxTokens(200),
        pipelines.WithPhiTemplate(),
    },
}
 
genPipeline, err := NewPipeline(session, config)
check(err)

messages := [][]pipelines.Message{
    {
       {Role: "system", Content: "you are a helpful assistant."},
       {Role: "user", Content: "what is the capital of the Netherlands?"},
    },
    {
       {Role: "system", Content: "you are a helpful assistant."},
       {Role: "user", Content: "who was the first president of the United States?"},
    },
},
 
batchResult, err := genPipeline.RunWithTemplate(messages)
if err == nil {
    fmt.Println(batchResult.GetOutput())
}
````

## Hardware acceleration 🚀

Hugot now also supports the following accelerator backends for your inference:
 - CUDA (tested on Onnx Runtime and OpenXLA). See below for setup instructions.
 - TensorRT (untested, available in Onnx Runtime only)
 - DirectML (untested, available in Onnx Runtime only)
 - CoreML (untested, available in Onnx Runtime only)
 - OpenVINO (untested, available in Onnx Runtime only)

Please help us out by testing the untested options above and providing feedback, good or bad!

To use Hugot with Nvidia gpu acceleration, you need to have the following:

- The Nvidia driver for your graphics card (if running in Docker and WSL2, starting with --gpus all should inherit the drivers from the host OS)
- ONNX Runtime:
    - The cuda gpu version of ONNX Runtime on the machine/docker container. You can see how we get that by looking at the [Dockerfile](./Dockerfile). You can also get the ONNX Runtime libraries that we use for testing from the release. Just download the gpu .so libraries and put them in /usr/lib64.
    - The required CUDA libraries installed on your system that are compatible with the ONNX Runtime gpu version you use. See [here](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html). For instance, for onnxruntime-gpu 19.0, we need CUDA 12.x (any minor version should be compatible) and cuDNN 9.x.
    - Start a session with the following:
      ```
      opts := []options.WithOption{
        options.WithOnnxLibraryPath("/usr/lib64/onnxruntime-gpu/libonnxruntime.so"),
        options.WithCuda(map[string]string{
          "device_id": "0",
        }),
      }
      session, err := NewORTSession(opts...)
      ```
- OpenXLA
    - Install CUDA support via the command `curl -sSf https://raw.githubusercontent.com/gomlx/gopjrt/main/cmd/install_cuda.sh | bash`
    - Start a session with the following:
      ```
      opts := []options.WithOption{
        options.WithCuda(map[string]string{
          "device_id": "0",
        }),
      }
      session, err := NewXLASession(opts...)
      ```

For the ONNX Runtime Cuda libraries, you can install CUDA 12.x by installing the full cuda toolkit, but that's quite a big package. In our testing on awslinux/fedora, we have been able to limit the libraries needed to run Hugot with Nvidia gpu acceleration to just these:

- cuda-cudart-12-9 cuda-nvrtc-12-9 libcublas-12-9 libcurand-12-9 libcufft-12-9 libcudnn9-cuda-12

On different distros (e.g. Ubuntu), you should be able to install the equivalent packages.

## Training and fine-tuning pipelines 

Hugot now also supports the training and fine-tuning of transformer pipelines (beta)! This functionality requires that you build with XLA enabled as we use gomlx behind the
scenes for training/fine-tuning: the onnx model will be loaded, converted to xla and trained using [goMLX](https://github.com/gomlx/gomlx), and serialized back to onnx format.

We is currently supported only for the **FeatureExtractionPipeline**. This can be used to fine-tune the vector embeddings for e.g. semantic textual similarity (for applications like RAG and semantic search). In order to fine-tune the feature extraction pipeline for semantic search you will need to collect a training dataset in the following format:

```
{"sentence1": "The quick brown fox jumps over the lazy dog", "sentence2": "A quick brown fox jumps over a lazy dog", "score": 1}
{"sentence1": "The quick brown fox jumps over the lazy dog", "sentence2": "A quick brown cow jumps over a lazy caterpillar", "score": 0.5}
```

See the [example](./testData/semanticSimilarityTest.jsonl) for a sample dataset.

The score is assumed to be a float between 0 and 1 that encodes the semantic similarity between the sentences, and by default a cosine similarity loss is used (see [sentence transformers](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss)). However, you can also specify a different loss function from `goMLX` using the `XLATrainingOptions` field in the `TrainingConfig` struct. See [the training tests](./hugot_training_test.go) for examples on how to train or fine-tune feature extraction pipelines.

Note that training on GPU is currently much faster and memory efficient than training on CPU, although optimizations are underway. On CPU, we recommend smaller batch sizes.

See [the tests](hugot_training_test.go) for an example on how to fine-tune semantic similarity starting with an open source sentence transformers model and a few examples.

## Performance Tuning

Firstly, the throughput depends largely on the size of the input requests. The best batch size is affected by the number of tokens per input, but we find batches of roughly 32 inputs per call to be a good starting point.

### ONNX Runtime
The library defaults to ONNX Runtime's default tuning settings. These are optimised for latency over throughput, and will attempt to parallelize single threaded calls to ONNX Runtime over multiple cores.

For maximum throughput, it is best to call a single shared Hugot pipeline from multiple goroutines (1 per core), using a channel to pass the input data. In this scenario, the following settings will greatly increase inference throughput.

```go
session, err := hugot.NewORTSession(
	hugot.WithInterOpNumThreads(1),
	hugot.WithIntraOpNumThreads(1),
	hugot.WithCpuMemArena(false),
	hugot.WithMemPattern(false),
)
```

InterOpNumThreads and IntraOpNumThreads constricts each goroutine's call to a single core, greatly reducing locking and cache penalties. Disabling CpuMemArena and MemPattern skips pre-allocation of some memory structures, increasing latency, but also throughput efficiency.

## File Systems
We use an [abstract file system](https://github.com/viant/afs) within Hugot. It works out of the box with various OS filesystems, to use object stores such as S3 please import the appropriate plugin from the afsc library, e.g.
```go
import _ "github.com/viant/afsc/s3"
```

## Limitations

Apart from the fact that only the aforementioned pipelines are currently implemented, the current limitations are:

- the library and cli are only built/tested on amd64-linux currently.

## Contributing

If you would like to contribute to Hugot, please see the [contribution guidelines](./contrib.md).
