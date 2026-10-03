//go:build cgo && (ORT || ALL) && (linux || darwin)

package hugot

import (
	_ "github.com/gomlx/compute-onnx"
)
