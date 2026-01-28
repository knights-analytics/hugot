# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [0.6.2] - 2026-01-28

First release of 2026!

###  Changed

- Added TabularPipeline to run onnx exports of classical ML models (regression or classification) from, e.g., sklearn
- Added multimodal support for text generation pipeline (kudos to @ajroetker)
- Memory mapped loading of onnx models on native fs to reduce memory usage

## [0.6.1] 🌲❄️🎄🎁 - 2025-12-23

###  Changed

- Explicit XLA compilation cache control via WithGoMLXBatchBuckets and WithGoMLXSequenceBuckets.
- Finalize device memory earlier in XLA sessions, reducing memory pressure when using TPU/GPU.
- Tokenizer can handle/ignore additional image tensor types.
- Allow 4D dimensions for multimodal featureExtraction pipelines.
- Allow multiple ONNX files/models to load from the same model directory.
- Restore support for GLIBC 2.34 in XLA
- Also disable XLA dependency autoinstall in training mode (will implement global C dependency autoinstall in near future!)

Thanks @ajroetker for your contributions!

## [0.6.0] 🌲❄️🎄🎁 - 2025-12-18

### Changed

- Integrated [ONNX Runtime GenAI](https://github.com/microsoft/onnxruntime-genai) backend for significantly faster generative inference and broad model support.
- Added ObjectDetection pipeline
- Added WithTPU option to NewXLASession
- FeatureExtractionPipeline now supports image inputs, enabling vision models like CLIP
- Updated Onnx Runtime to 1.23.2, and GoMLX to 0.26.0

### Breaking changes

- ORT Gen AI has strong requirements on the name of the base ORT library. It should not be renamed from the release zip (e.g. libonnxruntime.so)
- WithOnnxLibraryPath should now be the folder contining the ORT library. The library name is now inferred from the current operating system.
- XLA now uses go-xla to manage PJRT dependencies, see our [Dockerfile](./Dockerfile) for details 

### Fixed

- Model loading path could potentially duplicate paths (thanks @ajroetker)

## [0.5.10] - 2025-12-08

### Breaking changes

- breaking: GetStatistics on a session returns a map of pipeline name to statistics object

## [0.5.9] - 2025-12-08

### Breaking changes

- breaking: GetStatistics now returns a Statistics struct for the pipelines rather than a list of strings
- breaking: pipelineBackends has been renamed to backends

### Changed

- update of onnxruntime_go, goMLX, gopjrt

### Improvements

- support splitIntoWords for tokenClassificationPipeline

## [0.5.8] - 2025-11-22

### Changed

- Support models that do not utilize attention masks in FeatureExtractionPipeline
- Bump onnx-gomlx to v0.3.2 for expanded model support in Go sessions.

## [0.5.7] - 2025-11-11

### Changed

- Update Go, Tokenizers, OnnxRuntime and GoMLX dependencies
- Compatibility with NHWC and NCHW formats in Image Classification Pipeline

## [0.5.6] - 2025-10-22

### Changed

- Update to new goMLX project structure (0.24.0+)
- remove the dependency on python when installing goMLX libraries

## [0.5.5] - 2025-09-30

### Changed

- Update Go, Tokenizers, and GoMLX dependencies
- XLA now uses [StableHLO](https://openxla.org/stablehlo)
- XLA CUDA backend is now CUDA 13

## [0.5.4] - 2025-09-10

### Changed

- Fix: use right tokenization and token type IDs for Bert-style sentence pair in cross encoder

## [0.5.3] - 2025-09-01

### Changed

- Performance improvement: do not pad inputs to POW2 when using simplego backend

## [0.5.2] - 2025-08-30

### Changed

- Apply small input performance fix to goMLX backend 

## [0.5.1] - 2025-08-29

### Changed

- Fix performance regression for small inputs

## [0.5.0] - 2025-08-29

### 🚀 Generative pipeline in hugot!

- The new TextGenerationPipeline allows you to run generative models such as Gemma and Phi in golang. Kudos to [Riley Oh](https://github.com/riley-oh6) for getting this one
over the line!
- Currently only implemented for the ORT backend. Implementations for XLA and GO backend coming soon!
- See the documentation for how to get started

### 🚀 New pipelines: cross encoder and image classification

- The CrossEncoderPipeline implements the equivalent of sentence transformers' [Cross Encoder](https://sbert.net/docs/package_reference/cross_encoder/cross_encoder.html). Kudos to
[Fábio Correia](https://github.com/fabiodcorreia) for providing the initial implementation
- The ImageClassificationPipeline implements the equivalent of [Hugging Face's Image Classification](https://huggingface.co/tasks/image-classification) pipeline

### ✨ Training improvements

- The training session to fine-tune embeddings now accept TrainEval and Eval datasets to compute in-sample and test statistics
- The training session now implements early stopping based on the loss on the Eval dataset. Early stopping is evaluated at the end of each training epoch.
- The training session now accepts a layer freezing configuration to specify which layers of the transformer will be frozen during fine-tuning

### 📝 Tokenization

- The go tokenizer now supports unigram tokenization

### Changed

- Updated go to 1.25.0
- Upgraded GoMLX to 0.22.1

## [0.4.3] - 2025-07-18

### Changed

- Upgraded to latest Rust and Go tokenizers
- Generalised output tensor types, added int64 support (preparation for text generation pipeline) 
- Dependency updates and go 1.24.5

## [0.4.2] - 2025-06-26

### Changed

- Tokenizers obey max_position_embeddings, closing [issue #73](https://github.com/knights-analytics/hugot/issues/73)
- Provided default ORT path in darwin/mac
- Dependency updates and go 1.24.4

## [0.4.1] - 2025-06-03

### 🚀 New Features

#### ✨ Pure Go Backend
- Run inference and fine-tuning, all from native Go!
- Added `NewGoSession()` backend, enabling Hugot to run in pure Go environments without C dependencies
- Implements our most requested community feature using SimpleGo from [GoMLX](https://github.com/gomlx/gomlx)
- Optimized for simpler workloads and environments where cgo is restricted
- Works best with smaller models like [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- Performance sweet spot: batches of approximately 32 inputs per call

### 📊 Performance Notes
- The Go backend currently performs 5-20x slower than C backends
- For performance-critical applications, we recommend using C backends (XLA or ORT)
- Significant speed improvements expected with the introduction of SIMD in Go 1.25

### 🐞 Known Limitations
- The Go backend and tokenizer currently lack support for certain operators
- If you encounter compatibility issues, please [open an issue](https://github.com/knights-analytics/hugot/issues/new)
- As a temporary workaround, use a C backend until the issue is resolved

### 🚨 Breaking Changes
- **Build Tag Requirements Updated**:
    - For `NewORTSession()`: Add `-tags "ORT"` or `-tags "ALL"` to your build arguments
    - For `NewXLASession()`: Add `-tags "XLA"` or `-tags "ALL"` to your build arguments

### Added

- SimpleGo backend and SugarMe tokenizers for running Hugot without C dependencies

### Changed

- Upgraded CUDA libraries to 12.9
- Upgraded GoMLX for performance improvements and bugfixes.
