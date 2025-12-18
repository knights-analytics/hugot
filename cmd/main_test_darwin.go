//go:build (ORT || ALL) && darwin

package main

// onnxRuntimeSharedLibrary is the default ONNX Runtime library path for macOS.
const onnxRuntimeSharedLibrary = "/opt/homebrew/lib/libonnxruntime.dylib"
