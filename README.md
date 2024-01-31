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

HuGo relies on the [tokenizer](https://github.com/Knights-Analytics/tokenizers) library with bindings to huggingface's rust tokenizer, which is itself a fork of https://github.com/daulet/tokenizers. In particular, you will need to make available to HuGo the compiled libtokenizers.a file, which resides by default at /usr/lib/libtokenizers.a.

Assuming you have rust installed, you can compile the tokenizers library and get the required libtokenizers.a as simply as follows:

```
git clone https://github.com/Knights-Analytics/tokenizers -b main && \
    cd tokenizers && \
    cargo build --release
mv target/release/libtokenizers.a /usr/lib/libtokenizers.a
```

Alternatively, you can just download libtokenizers.a from the release section of the repo.

Once libtokenizers.a is in place, the library can be used as follows:

```
import (
	"github.com/phuslu/log"
	ort "github.com/yalue/onnxruntime_go"
)

log.Info().Msg("Initialising Onnx Runtime Environment")
checks.Check(ort.InitializeEnvironment())
checks.Check(ort.DisableTelemetry())
defer func() {
    log.Info().Msg("Destroying Onnx Runtime")
    checks.Check(ort.DestroyEnvironment())
}()

log.Info().Msg("Creating new token classification pipeline")
// we use an onnx exported version of distilbert-NER which recognizes
// Individuals, organisations, locations, and miscellaneous
// see https://huggingface.co/dslim/distilbert-NER
modelPath := "./distilbert-NER"
pipelineSimple := NewTokenClassificationPipeline(modelPath, "testPipeline", WithSimpleAggregation())

log.Info().Msg("Running the pipeline on a batch of strings")
results := pipelineSimple.run(["Microsoft incorporated.", "Yesterday I went to Berlin and met with Jack Brown."])
PrintTokenEntities(results)
```

Which yields the following:

```
Input 0
{Entity:LABEL_3 Score:0.9953674 Scores:[] Index:0 Word:Microsoft TokenId:0 Start:0 End:9 IsSubword:false}
{Entity:LABEL_0 Score:0.9985231 Scores:[] Index:0 Word:incorporated. TokenId:0 Start:10 End:23 IsSubword:false}
Input 1
{Entity:LABEL_0 Score:0.9994594 Scores:[] Index:0 Word:Yesterday I went to TokenId:0 Start:0 End:19 IsSubword:false}
{Entity:LABEL_5 Score:0.99794966 Scores:[] Index:0 Word:Berlin TokenId:0 Start:20 End:26 IsSubword:false}
{Entity:LABEL_0 Score:0.9997991 Scores:[] Index:0 Word:and met with TokenId:0 Start:27 End:39 IsSubword:false}
{Entity:LABEL_1 Score:0.9973983 Scores:[] Index:0 Word:Jack TokenId:0 Start:40 End:44 IsSubword:false}
{Entity:LABEL_2 Score:0.998172 Scores:[] Index:0 Word:Brown TokenId:0 Start:45 End:50 IsSubword:false}
{Entity:LABEL_0 Score:0.9996651 Scores:[] Index:0 Word:. TokenId:0 Start:50 End:51 IsSubword:false}
```
