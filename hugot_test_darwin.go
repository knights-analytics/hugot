//go:build darwin

package hugot

// onnxRuntimeSharedLibrary is the default ONNX Runtime library path for macOS.
// This assumes ONNX Runtime was installed via Homebrew (Apple Silicon default location).
// For Intel Macs, this may be at /usr/local/lib/libonnxruntime.dylib
const onnxRuntimeSharedLibrary = "/opt/homebrew/lib/libonnxruntime.dylib"
