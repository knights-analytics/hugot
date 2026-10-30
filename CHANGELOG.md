# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [0.8.5]

### Changed
- The Go backend now attempts to use dynamic axes to reduce graph compilations, falling back to bucketed compilations.

## [0.8.4]

- This release brings the list of supported pipelines to parity with Huggingface transformers, adding the remaining pipelines and removing others that have been deprecated.

### Added

- Added `AnyToAnyPipeline` for supported multimodal models, tested with Phi-4 multimodal, Janus, and LFM bundles.
- Added `VideoClassificationPipeline` for VideoMAE ONNX models.
- Added `KeypointMatchingPipeline` for SuperGlue exports.
- Added `DownloadOptions.CacheDir` and `DownloadGraphBundle` to configure the Hugging Face cache and download complete ONNX graph bundles, including related weights and child graphs.
- Added `TextGenerationPipeline.GetContextLength()` to report the model's configured context length.
- Fine-tuning for `TextClassificationPipeline`, single-label and multi-label, with `datasets.TextClassificationDataset` (JSONL or in memory). The defaults are AdamW (learning rate 5e-5, weight decay 0.01) with softmax or sigmoid cross-entropy.
- Fine-tuning for `TokenClassificationPipeline` (e.g. NER) with `datasets.TokenClassificationDataset`. Each word's tag goes on its first sub-word token; other sub-word, special and padding tokens are masked out of the loss.
- `TrainingStatistics.EpochTrainAccuracies` and `EpochEvalAccuracies`, recorded for classifiers.
- Contrastive fine-tuning of embeddings with in-batch negatives (sentence-transformers' `MultipleNegativesRankingLoss`): `datasets.ContrastiveDataset` takes anchor/positive pairs with optional hard negatives, and needs no similarity scores.
- The new `evaluation` package: entity-level precision, recall and F1 for token classification (`ScoreEntities`, `EvaluateTokenClassification`), and per-class precision, recall and F1 with macro, weighted and micro averages for text classification (`ScoreClassification`, `ScoreMultiLabel`, `EvaluateTextClassification`).
- `TrainingStatistics.EpochTrainF1` / `EpochEvalF1`: entity-level F1 for token classification and macro-F1 for text classification, per epoch.
- `pipelines.WithWordSplitter` for token classification: one function splits raw text into the words the model labels, at inference (`Run` labels the splitter's words, as `RunWords` would) and in training, where `TokenClassificationDataset` examples may be given as raw `text` and are split by the trainer's pipeline. New `TrainerConfig.PipelineOptions` passes pipeline options such as the splitter to the trainer.
- Span-annotated training data for token classification, in the format Prodigy's `ner_manual` exports: `TokenClassificationExample.Spans` holds entities as character offsets, converted to IOB word tags with the pipeline's word splitter or the annotation tool's tokens as the words. Word-level and span lines can share a file, and examples not answered `accept` are skipped. `EvaluateTokenClassification` scores span examples the same way.
- Prodigy text classification exports for `TextClassificationDataset`: choice annotations (`accept` lists) for single- and multi-label datasets, and binary annotations (`label` + `answer`). A binary rejection becomes the other class of a two-class model and is skipped otherwise; binary annotations train a multi-label model only when it has that single label. `ignore` answers are skipped. `EvaluateTextClassification` resolves them the same way.

### Breaking changes

- Removed `QuestionAnsweringPipeline`.
- Removed `VisualQuestionAnsweringPipeline`.
- Removed `ImageToTextPipeline` and its configuration types. Use `AnyToAnyPipeline` with a supported multimodal model for image understanding and captioning.

### Changed

- `WithGenerativeEngine` now admits requests fairly. Free batch slots go round-robin across jobs, with new jobs served first in arrival order. Before, slots went to whichever job Go's randomized map iteration picked first, so a request could be passed over repeatedly under load.
- Embedding training reads the number of model inputs and the attention mask's position from the model, instead of assuming three inputs with the mask second.
- `Trainer.Save` also copies the model's `config.json`, so a saved model keeps its labels and configuration.
- The last epoch is now evaluated too. Before, the epoch statistics and early stopping skipped it.
- `TrainerConfig.EvalDataset` is evaluated every epoch whenever it is set. Before, it was evaluated only with early stopping and otherwise silently ignored.

### Fixed

- On backends that embed variables as constants (the Go backend), every evaluation during training reused the weights of the first one, so the epoch statistics and early stopping never saw training progress. Evaluation graphs are now rebuilt before each evaluation on those backends.

## [0.8.3] - 2026-10-14

### Added

- `GroundedObjectDetectionPipeline` implements support for Grounding DINO models, which allow for object detection using a set of textual labels (zero shot)
- Support for SAM2 models in `maskGeneration` with both box and point based segmentation prompts

### Changed

- Do not pad batches when using the Go backend (performance improvement).
- Fixed #149 Hugot/GoMLX panics with index out of range when the batch size is padded.
- Upgraded to ONNX Runtime v1.31.0.

## [0.8.2] - 2026-10-06

### Changed

- Python parity tests: embedded Python ONNX reference vectors with pinned tolerances to verify numeric parity across ORT, GoMLX, and XLA backends.
- Raw feature access and dimensions for text and image feature-extraction pipelines.

## [0.8.1] - 2026-10-02

### Added

- **New pipelines**:
  - **Image**: `FillMaskPipeline`, `MaskGenerationPipeline`, `ImageSegmentationPipeline`, `DepthEstimationPipeline`, `BackgroundRemovalPipeline`, `ZeroShotImageClassificationPipeline`.
  - **Audio**: `AudioClassificationPipeline`, `ZeroShotAudioClassificationPipeline`, `AutomaticSpeechRecognitionPipeline`, `TextToAudioPipeline`, `TextToSpeechPipeline`.
  - **Multimodal**: `VisualQuestionAnsweringPipeline`, `DocumentQuestionAnsweringPipeline`, `TableQuestionAnsweringPipeline`, `ImageToTextPipeline`, `ImageTextToTextPipeline`.
- Image pipelines expose `RunWithImages(ctx, []image.Image)` so callers can feed in-memory `image.Image` values without a disk round-trip, in addition to `Run(ctx, []string)` which loads from paths or data URIs.
- Audio input plumbing: `backends.AudioWaveform`, `backends.AudioURLs`, `Message.AudioURLs`, and `ConversationHasAudio` / `FlattenAudioURLs` helpers used by the ORT session pipeline.
- **Per-conversation system turns** are expressed as `History[0] = {Role: "system", ...}` on all five input types. The pipeline-level `SystemPrompt` on `multimodalGeneration` continues to be applied globally by the ORT backend via `createORTMessages`.
- Upgraded to ONNX Runtime GenAI 0.17.0.

### Changed

- Improved several new pipelines to better match the Hugging Face reference behavior and numerics: pre/post-processing, output naming, and result shapes for feature extraction, image captioning, mask generation, table/document question answering, and zero-shot audio.
- Pipeline operations are now methods on `*hugot.Session` (`session.NewPipeline`, `session.GetPipeline`, `session.GetPipelines`, `session.ClosePipeline`); the old free-function variants are deprecated thin wrappers kept for compatibility.
- BREAKING: Training is now configured with `session.NewTrainer(TrainerConfig, ...TrainerOption)` and run with `trainer.Train(ctx)`.

### Removed

- Removed the Rust tokenizer dependency; the Go tokenizer is fast enough, supports all of our test pipelines, and avoids the extra cgo dependency/project complexity.

## [0.8.0] - 2026-09-22

### Fixed

- Fix compilation on non-linux platforms.

## [0.7.9] - 2026-09-22

### Added

- Upgraded to ONNX Runtime v1.30.0.
- Migrated to Go bindings from the official repository.
- Fixed a missing filesystem error if using downloader before starting a session.
- Made file copies atomic to avoid partial model downloads on abort.
- Check for a valid download destination before downloading.

## [0.7.8] - 2026-09-02

### Added

- Added `NewORTTrainingSession` for training with the ORT backend.
- Added `options.WithGoMLX` to use GoMLX for model execution with an ORT session.
- Added pluggable, session-scoped filesystem adapters through `options.WithFileSystem`, see README.MD for details.
- Added concurrency protection for model and pipeline registries and made model closing idempotent.

### Changed

- Upgraded to Go 1.27.0.
- Upgraded GoMLX dependencies, ONNX Runtime to 1.29.0, and ORT GenAI to 0.15.2.
- Refactored backend and pipeline handling around typed backend and model abstractions.

### Breaking changes

- Filesystem integrations must implement `fileutil.FileSystem` and be supplied with `options.WithFileSystem`; the previous global `viant/afs` integration is no longer used.
- `fileutil.WalkDir` now takes a context and handler directly, and `fileutil.FileStats` returns `os.FileInfo`; update callers and adapters accordingly.
- Exported pipeline constructors no longer accept `*options.Options`; backend configuration is now supplied when creating the session.

## [0.7.7] - 2026-08-03

###  Changed

- Upgraded to CUDA 13
- Upgraded to ORT v1.28.0
- Upgraded to ORT GenAI v0.15.0
- Upgraded to GoMLX v0.28.2
- Fixes go tokenizer FIRST/MAX/AVERAGE aggregation (thanks @luanlorenzo)

## [0.7.6] - 2026-07-24

###  Changed

- Upgraded to GoMLX [v0.28.0](https://github.com/gomlx/gomlx/releases/tag/v0.28.0)
  - Includes experimental SIMD for significant performance improvements for the Go backend
- Refactor/simplification of pipeline management and lifecycle
- Token Classification pipeline correctly handles unknown tokens (thanks @ohhmkar)
- Fixes tokenizer timings reporting per pipeline

## [0.7.5] - 2026-06-05

###  Changed

- Upgraded to ORT v1.26.0
- Upgraded to ORT GenAI v0.14.0

## [0.7.4] - 2026-05-24

###  Changed

- Do not specify go patch version in go.mod
- Fixes for release docker image build and push

## [0.7.3] - 2026-05-20

###  Changed

- Fix panic on double-close (thanks @sananguliyev) 
- Upgraded to ORT v1.25.1
- Test structure improvements
- Documentation and examples improvements (thanks @kepae)

## [0.7.2] - 2026-04-22

###  Changed

- Upgraded to ORT v1.25.0

## [0.7.1] - 2026-04-19

###  Changed

- Breaking: Added context.Context to many functions for graceful cancellation
- Experimental: Added WithGenerativeEngine to ORT, to enable concurrent inferencing and intelligent batching 
- Added locks to protect against concurrent model and pipeline initialisations
- Added 2-bit bucketing to XLA backend for reduced padding overhead
- Upgraded to GoMLX v0.27.3
- Upgraded to ORT v1.24.4
- Upgraded to ORT GenAI v0.13.1

## [0.7.0] - 2026-03-30

###  Changed

- Added QuestionAnsweringPipeline
- Added ConstrainedGeneration, Tools and Guidance to TextGenerationPipeline
- Upgraded to GoMLX v0.27.2
  - Fixes multiple quantization issues in simple Go backend
- Upgraded to ORT v1.24.4
- Upgraded to ORT GenAI v0.12.2
- Replaced Go Tokenizers with GoMLX version
  - Supports Pair Tokenization
  - Further reduces dependencies
- Fix: Allow use of SOFTMAX in TextClassificationPipeline
- Breaking: Remove CLI to simplify dependencies (can be reintroduced as a separate library on request)

## [0.6.5] - 2026-03-17

###  Changed

- Upgraded to GoMLX v0.27.1
  - Includes performance improvements, new onnx ops and quantization types for Go and XLA backends
  - See [here](https://github.com/gomlx/gomlx/releases/tag/v0.27.1) for details
- Upgraded to ORT v1.24.3
- Upgraded to ORT GenAI v0.12.1
- Upgraded to Tokenizers v1.26.0
- Added WithProfiling and WithOptimizedModelFilePath options

## [0.6.4] - 2026-03-02

- bugfix: correct wrong input token creation in feature extraction for models with eos tokens.

## [0.6.3] - 2026-02-26

###  Changed

- Upgraded to ORT GenAI v0.12.0
- Upgraded to ORT v1.24.2
- Upgraded to Go v1.26.0
- Upgraded tokenizers to v1.25.0
- Added WithLogSeverityLevel, WithEnvLoggingLevel, WithGraphOptimizationLevel and WithExtraExecutionProvider options
- Separated TensorRT and NvTensorRTRTX
- Memory usage and bugfixes to generative pipelines

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
