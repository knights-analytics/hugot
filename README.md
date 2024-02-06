# <span>HuGo: Huggingface 🤗 pipelines for golang

## What

This library aims to provide an easy, scalable, and hassle-free way to run huggingface transformer pipelines in golang applications. It is built on the following principles:

1. Fidelity to the original Huggingface python implementations: we aim to accurately replicate huggingface inference implementations for the implemented pipelines, so that models trained and tested in python can be seamlessly deployed in golang
2. Hassle-free and performant production use: we exclusively support onnx exports of huggingface models. Huggingface transformer models can be easily exported to onnx via huggingface optimum and used with the library (see instructions below)
3. Run on your hardware: the aim is to be able to run onnx-exported huggingface transformer models on local hardware rather than relying on the http huggingface API

## Why

While developing and fine-tuning transformer models with the huggingface python library is a great experience, if your production stack is golang-based being able to reliably deploy and scale the resulting pytorch models can be challenging. This library aims to make the process easy.

## For whom

For the golang developer or ML engineer who wants to run transformer piplines at scale on their own hardware for their application

## What is already there

We currently have implementations for the following three transfomer pipelines:

- [featureExtraction](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.FeatureExtractionPipeline)
- [textClassification](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.TextClassificationPipeline) (single label classification only)
- [tokenClassification](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.TokenClassificationPipeline)

Implementations for additional pipelines will follow. We also very gladly accept PRs to expand the set of pipelines! See [here](https://huggingface.co/docs/transformers/en/main_classes/pipelines) for the missing pipelines that can be implemented.

## Installation and usage

HuGo has two main dependencies:

- the [tokenizer](https://github.com/Knights-Analytics/tokenizers) library with bindings to huggingface's rust tokenizer, which is itself a fork of https://github.com/daulet/tokenizers. In particular, you will need to make available to HuGo the compiled libtokenizers.a file, which resides by default at /usr/lib/libtokenizers.a.
- the [onnxruntime_go](https://github.com/yalue/onnxruntime_go) library, with go bindings to onnxruntime. You will need to make available to HuGo the onnxruntime.so file, which resides by default at /usr/lib/onnxruntime.so

Assuming you have rust installed, you can compile the tokenizers library and get the required libtokenizers.a as simply as follows:

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

See also the dev/test [dockerfile](./Dockerfile).

Once these pieces are in place, the library can be used as follows:

```
TODO
```
