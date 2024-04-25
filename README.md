# <span>Hugot: Huggingface 🤗 pipelines for golang

[![Go Reference](https://pkg.go.dev/badge/github.com/knights-analytics/hugot.svg)](https://pkg.go.dev/github.com/knights-analytics/hugot)
[![Go Report Card](https://goreportcard.com/badge/github.com/knights-analytics/hugot)](https://goreportcard.com/report/github.com/knights-analytics/hugot)
[![Coverage Status](https://coveralls.io/repos/github/knights-analytics/hugot/badge.svg?branch=main)](https://coveralls.io/github/knights-analytics/hugot?branch=main)

## What

The goal of this library is to provide an easy, scalable, and hassle-free way to run huggingface transformer pipelines in golang applications. It is built on the following principles:

1. Fidelity to the original Huggingface python implementations: the aim is to accurately replicate huggingface inference implementations for the implemented pipelines, so that models trained and tested in python can be seamlessly deployed in a golang application
2. Hassle-free and performant production use: we exclusively support onnx exports of huggingface models. Pytorch transformer models that don't have an onnx version can be easily exported to onnx via [huggingface optimum](https://huggingface.co/docs/optimum/index), and used with the library
3. Run on your hardware: this library is for those who want to run transformer models tightly coupled with their go applications, without the performance drawbacks of having to hit a rest API, or the hassle of setting up and maintaining e.g. a python RPC service that talks to go.

We support inference on CPU and on all accelerators supported by ONNXRuntime. Note, however, that currently only CPU and GPU inference on nvidia GPU (with cuda) are tested (see below).

## Why

Developing and fine-tuning transformer models with the huggingface python library is a great experience, but if your production stack is golang-based being able to reliably deploy and scale the resulting pytorch models can be challenging and require quite some setup. This library aims to allow you to just lift-and-shift your python model and use the same huggingface pipelines you use for development for inference in a go application.

## For whom

For the golang developer or ML engineer who wants to run transformer piplines on their own hardware, tightly coupled with their own application.

## Implemented pipelines

Currently, we have implementations for the following transfomer pipelines:

- [featureExtraction](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.FeatureExtractionPipeline)
- [textClassification](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.TextClassificationPipeline)
- [tokenClassification](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.TokenClassificationPipeline)

Implementations for additional pipelines will follow. We also very gladly accept PRs to expand the set of pipelines! See [here](https://huggingface.co/docs/transformers/en/main_classes/pipelines) for the missing pipelines that can be implemented, and the contributing section below if you want to lend a hand.

Hugot can be used both as a library and as a command-line application. See below for usage instructions.

## Hardware acceleration 🚀

Hugot now also supports the following accelerator backends for your inference:
 - CUDA (tested). See below for setup instructions.
 - TensorRT (untested)
 - DirectML (untested)
 - CoreML (untested)
 - OpenVINO (untested)

Please help us out by testing the untested options above and providing feedback, good or bad!

To use Hugot with nvidia gpu acceleration, you need to have the following:

- The cuda gpu version of onnxruntime on the machine/docker container. You can see how we get that by looking at the [Dockerfile](./Dockerfile). You can also get the onnxruntime libraries that we use for testing from the release. Just download the gpu .so libraries and put them in /usr/lib64.
- the nvidia driver for your graphics card
- the required cuda libraries installed on your system that are compatible with the onnxruntime gpu version you use. See [here](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html). For instance, for onnxruntime-gpu 17.3, we need CUDA 12.x (any minor version should be compatible) and cuDNN 8.9.2.26.

On the last point above, you can install CUDA 12.x by installing the full cuda toolkit, but that's quite a big package. In our testing on awslinux/fedora, we have been able to limit the libraries needed to run hugot with nvidia gpu acceleration to just these:

- cuda-cudart-12-4 libcublas-12-4 libcurand-12-4 libcufft-12-4 (from fedora repo)
- libcudnn8 (from RHEL repo, for cuDNN)

On different distros (e.g. Ubuntu), you should be able to install the equivalent packages and gpu inference should work.

## Limitations

Apart from the fact that only the aforementioned pipelines are currently implemented, the current limitations are:
    - the library and cli are only built/tested on amd64-linux

Pipelines are also tested on specifically NLP use cases. In particular, we use the following models for testing:
- feature extraction: all-MiniLM-L6-v2
- text classification: distilbert-base-uncased-finetuned-sst-2-english
- token classification: distilbert-NER and Roberta-base-go_emotions

If you encounter any further issues or want further features, please open an issue.

## Installation and usage

Hugot can be used in two ways: as a library in your go application, or as a command-line binary.

### Use it as a library

To use Hugot as a library in your application, you will need the following dependencies on your system:

- the tokenizers.a file obtained from building the [tokenizer](https://github.com/Knights-Analytics/tokenizers) go library (which is itself a fork of https://github.com/daulet/tokenizers). This file should be at /usr/lib/tokenizers.a so that hugot can load it.
- the onnxruntime.go file obtained from the onnxruntime project. This is dynamically linked by hugot and used by the onnxruntime inference library[onnxruntime_go](https://github.com/yalue/onnxruntime_go). This file should be at /usr/lib/onnxruntime.so or /usr/lib64/onnxruntime.so

You can get the libtokenizers.a in two ways. Assuming you have rust installed, you can compile the tokenizers library and get the required libtokenizers.a:

```
git clone https://github.com/Knights-Analytics/tokenizers -b main && \
    cd tokenizers && \
    cargo build --release
mv target/release/libtokenizers.a /usr/lib/libtokenizers.a
```

Alternatively, you can just download libtokenizers.a from the release section of the repo.

For onnxruntime, it suffices to download it, untar it, and place it in the right location:

```
curl -LO https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz && \
   tar -xzf onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz && \
   mv ./onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}/lib/libonnxruntime.so.${ONNXRUNTIME_VERSION} /usr/lib/onnxruntime.so
```

See also the [dockerfile](./Dockerfile) used for building & testing.

Once these pieces are in place, the library can be used as follows:

```go
import (
	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/pipelines"
)

func check(err error) {
    if err != nil {
        panic(err.Error())
    }
}
// start a new session. This looks for the onnxruntime.so library in its default path, e.g. /usr/lib/onnxruntime.so
session, err := hugot.NewSession()
// if your onnxruntime.so is somewhere else, you can explicitly set it by using WithOnnxLibraryPath
// session, err := hugot.NewSession(WithOnnxLibraryPath("/path/to/onnxruntime.so"))
check(err)
// A successfully created hugot session needs to be destroyed when you're done
defer func(session *hugot.Session) {
    err := session.Destroy()
    check(err)
}(session)

// Let's download an onnx sentiment test classification model in the current directory
// note: if you compile your library with build flag NODOWNLOAD, this will exclude the downloader.
// Useful in case you just want the core engine (because you already have the models) and want to
// drop the dependency on huggingfaceModelDownloader.
modelPath, err := session.DownloadModel("KnightsAnalytics/distilbert-base-uncased-finetuned-sst-2-english", "./", hugot.NewDownloadOptions())
check(err)

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
```

See also hugot_test.go for further examples.

### Use it as a cli: Huggingface 🤗 pipelines from the command line

With hugot you don't need python, pytorch, or even go to run huggingface transformers. Simply install the hugot cli (alpha):

```
curl https://raw.githubusercontent.com/knights-analytics/hugot/main/scripts/install-hugot-cli.sh | bash
```

This will install the hugot binary at $HOME/.local/bin/hugot, and the corresponding onnxruntime.so library at $HOME/lib/hugot/onnxruntime.so.
The if $HOME/.local/bin is on your $PATH, you can do:

```
hugot run --model=KnightsAnalytics/distilbert-base-uncased-finetuned-sst-2-english --input=/path/to/input.jsonl --output=/path/to/folder/output --type=textClassification
```

Hugot will load the model, process the input, and write the results in the output folder.
Note that the hugot cli currently expects the input in a specific format: json lines with an "input" key containing the string to process.
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

Note that if --input is not provided, hugot will read from stdin, and if --output is not provided, it will write to stdout.
This allows to chain things like:

```
echo '{"input":"The director tried too much"}' | hugot run --model=/path/to/model --type=textClassification | jq
```

To be able to run transformers fully from the command line.

Note that the --model parameter can be:
    1. the full path to a model to load
    2. the name of a huggingface model. Hugot will first try to look for the model at $HOME/hugot, or will try to download the model from huggingface.

## Performance Tuning

Firstly, the throughput of onnxruntime depends largely on the size of the input requests. The best batch size is affected by the number of tokens per input, but we find batches of roughly 32 inputs per call to be optimal.

The library defaults to onnxruntime's default tuning settings. These are optimised for latency over throughput, and will attempt to parallelize single threaded calls to onnxruntime over multiple cores.

For maximum throughput, it is best to call a single shared hugot pipeline from multiple goroutines (1 per core), using a channel to pass the input data. In this scenario, the following settings will greatly increase inference throughput.

```go
session, err := hugot.NewSession(
	hugot.WithInterOpNumThreads(1),
	hugot.WithIntraOpNumThreads(1),
	hugot.WithCpuMemArena(false),
	hugot.WithMemPattern(false),
)
```

InterOpNumThreads and IntraOpNumThreads constricts each goroutine's call to a single core, greatly reducing locking and cache penalties. Disabling CpuMemArena and MemPattern skips pre-allocation of some memory structures, increasing latency, but also throughput efficiency.

For GPU the config above also applies. We are still testing the optimum GPU configuration, whether it is better to run in parallel or with a single thread, and what size of input batch is fastest.

## Contributing

### Development environment

The easiest way to contribute to hugot is by developing inside a docker container that has the tokenizer and onnxruntime libraries.
From the source folder, it should be as easy as:

```bash
make start-dev-container
```

which will download the test models, build the test container, and launch it (see [compose-dev](./compose-dev.yaml)), mounting the source code at /home/testuser/repositories/hugot. Then you can attach to the container with e.g. vscode remote extension as testuser. The vscode attached container configuration file can be set to:

```
{
    "remoteUser": "testuser",
    "workspaceFolder": "/home/testuser/repositories/hugot",
    "extensions": [
		"bierner.markdown-preview-github-styles",
		"golang.go",
		"ms-azuretools.vscode-docker"
	],
    "remoteEnv": {"GOPATH": "/home/testuser/go"}
}
```

Once you're done, you can tear the container down with:

```bash
make stop-dev-container
```

Alternatively, you can use your IDE devcontainer support, and point it to the [Dockerfile](./Dockerfile).

If you prefer to develop on bare metal, you will need to download the tokenizers.a to /usr/lib/tokenizers.a and onnxruntime.so to /usr/lib/onnxruntime.so.

### Run the tests

The full test suite can be run as follows. From the source folder:

```bash
make clean run-tests
```

This will build a test image and run all tests in a container. A testTarget folder will appear in the source directory with the test results.

### Contribution process

1. create or find an issue for your contribution
2. fork and develop
3. add tests and make sure the full test suite passes and test coverage does not dip below 80%
4. create a MR linking to the relevant issue

Thank you for contributing to hugot!
