# <span>Hugot: Huggingface 🤗 pipelines for golang

[![Go Reference](https://pkg.go.dev/badge/github.com/knights-analytics/hugot.svg)](https://pkg.go.dev/github.com/knights-analytics/hugot)
[![Go Report Card](https://goreportcard.com/badge/github.com/knights-analytics/hugot)](https://goreportcard.com/report/github.com/knights-analytics/hugot)
[![Coverage Status](https://coveralls.io/repos/github/knights-analytics/hugot/badge.svg?branch=main)](https://coveralls.io/github/knights-analytics/hugot?branch=main)

## What

The goal of this library is to provide an easy, scalable, and hassle-free way to run huggingface transformer pipelines in golang applications. It is built on the following principles:

1. Fidelity to the original Huggingface python implementations: the aim is to accurately replicate huggingface inference implementations for the implemented pipelines, so that models trained and tested in python can be seamlessly deployed in a golang application
2. Hassle-free and performant production use: we exclusively support onnx exports of huggingface models. Pytorch transformer models that don't have an onnx version can be easily exported to onnx via [huggingface optimum](https://huggingface.co/docs/optimum/index), and used with the library
3. Run on your hardware: this library is for those who want to run transformer models tightly coupled with their go applications, without the performance drawbacks of having to hit a rest API, or the hassle of setting up and maintaining e.g. a python RPC service that talks to go.

## Why

Developing and fine-tuning transformer models with the huggingface python library is a great experience, but if your production stack is golang-based being able to reliably deploy and scale the resulting pytorch models can be challenging and require quite some setup. This library aims to allow you to just lift-and-shift your python model and use the same huggingface pipelines you use for development for inference in a go application.

## For whom

For the golang developer or ML engineer who wants to run transformer piplines on their own hardware, tightly coupled with their own application.

## What is already there

Currently we have implementations for the following transfomer pipelines:

- [featureExtraction](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.FeatureExtractionPipeline)
- [textClassification](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.TextClassificationPipeline) (single label classification only)
- [tokenClassification](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.TokenClassificationPipeline)

Implementations for additional pipelines will follow. We also very gladly accept PRs to expand the set of pipelines! See [here](https://huggingface.co/docs/transformers/en/main_classes/pipelines) for the missing pipelines that can be implemented, and the contributing section below if you want to lend a hand.

Hugot can be used both as a library and as a command-line application. See below for usage instructions.

## Limitations

Apart from the fact that only the aforementioned pipelines are currently implemented, the current limitations are:
    - the library and cli are only tested on amd64-linux
    - only CPU inference is supported

Pipelines are also tested on specifically NLP use cases. In particular, we use the following models for testing:
- feature extraction: all-MiniLM-L6-v2
- text classification: distilbert-base-uncased-finetuned-sst-2-english
- token classification: distilbert-NER

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
// we now create a text classification pipeline. It requires the path to the onnx model folder,
// and a pipeline name
sentimentPipeline, err := session.NewTextClassificationPipeline(modelPath, "testPipeline")
check(err)
// we can now use the pipeline for prediction on a batch of strings
batch := []string{"This movie is disgustingly good !", "The director tried too much"}
batchResult, err := sentimentPipeline.Run(batch)
check(err)
// batchResult is an interface so that we can treat pipelines uniformly.
// we can cast it to the concrete result type of this pipeline
result, ok := batchResult.(*pipelines.TextClassificationOutput)
// and do whatever we want with it :)
s, err := json.Marshal(result)
check(err)
fmt.Println(string(s))
// {"ClassificationOutputs":[[{"Label":"POSITIVE","Score":0.9998536}],[{"Label":"NEGATIVE","Score":0.99752176}]]}
```

See also hugot_test.go for further examples.

### Use it as a cli: Huggingface 🤗 pipelines from the command line

With hugot you don't need python, pytorch, or even go to run huggingface transformers. Simply install the hugot cli (alpha):

```
bash <(curl -s https://github.com/knights-analytics/hugot/blob/main/scripts/install-hugot-cli.sh)
```

This will install the hugot binary at $HOME/.local/bin/hugot, and the corresponding onnxruntime.so library at $HOME/lib/hugot/onnxruntime.so.
The if $HOME/.local/bin is on your $PATH, you can do:

```
hugot run --model=/path/to/onnx/model --input=/path/to/input.jsonl --output=/path/to/folder/output --type=textClassification
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
echo '{"input":"The director tried too much","output":[{"Label":"NEGATIVE","Score":0.99752176}]}' | hugot run --model=/path/to/model --type=textClassification | jq
```

To be able to run transformers fully from the command line.

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

The full test suite can be ran as follows. From the source folder:

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