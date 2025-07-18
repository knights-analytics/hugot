# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

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
