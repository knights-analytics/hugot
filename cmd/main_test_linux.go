//go:build (ORT || ALL) && linux

package main

// onnxRuntimeSharedLibrary is the default ONNX Runtime library path for Linux.
const onnxRuntimeSharedLibrary = "/usr/lib64/onnxruntime.so"
