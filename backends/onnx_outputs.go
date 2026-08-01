package backends

import (
	"fmt"
	"strconv"
	"strings"
)

// ModelIdentity returns the canonical model cache identity for a filesystem model path,
// ONNX filename, and ordered ONNX output selection.
//
// Empty or nil onnxOutputNames preserve the legacy path:filename identity so existing
// all-output sessions continue to share models. A non-empty selection is encoded with a
// count and length-prefixed names so names containing ':' or ',' cannot collide with a
// different ordered selection. Order is preserved exactly; names are never sorted.
//
// Encoding for a non-empty selection:
//
//	<path>:<onnxFilename>#o<count>/<len>:<name>/...
//
// path: absolute or relative model directory path used as the first identity component.
// onnxFilename: optional ONNX filename within the model directory; may be empty when the
// directory contains a single .onnx file.
// onnxOutputNames: ordered subset of graph outputs to request; nil or empty means all outputs.
//
// Returns the identity string used for the session model map key, model lock key, and Model.ID.
func ModelIdentity(path, onnxFilename string, onnxOutputNames []string) string {
	var b strings.Builder
	b.Grow(len(path) + len(onnxFilename) + 2)
	b.WriteString(path)
	b.WriteByte(':')
	b.WriteString(onnxFilename)
	if len(onnxOutputNames) == 0 {
		return b.String()
	}
	b.WriteString("#o")
	b.WriteString(strconv.Itoa(len(onnxOutputNames)))
	for _, name := range onnxOutputNames {
		b.WriteByte('/')
		b.WriteString(strconv.Itoa(len(name)))
		b.WriteByte(':')
		b.WriteString(name)
	}
	return b.String()
}

// CopyOnnxOutputNames returns a defensive copy of onnxOutputNames so later caller mutations
// cannot change a loaded model's selection or identity.
//
// onnxOutputNames: requested ONNX output names from pipeline configuration; may be nil.
//
// Returns nil when onnxOutputNames is nil, otherwise an independent slice with the same
// elements and order.
func CopyOnnxOutputNames(onnxOutputNames []string) []string {
	if onnxOutputNames == nil {
		return nil
	}
	copied := make([]string, len(onnxOutputNames))
	copy(copied, onnxOutputNames)
	return copied
}

// ValidateOnnxOutputNameList checks that a non-empty requested ONNX output selection contains
// no blank names and no duplicates. Empty and nil selections are valid and mean "all outputs".
//
// onnxOutputNames: ordered ONNX output names to validate before graph metadata lookup.
//
// Returns an actionable error for blank or duplicate names, otherwise nil.
func ValidateOnnxOutputNameList(onnxOutputNames []string) error {
	if len(onnxOutputNames) == 0 {
		return nil
	}
	seen := make(map[string]struct{}, len(onnxOutputNames))
	for _, name := range onnxOutputNames {
		if strings.TrimSpace(name) == "" {
			return fmt.Errorf("ONNX output name must not be blank")
		}
		if _, exists := seen[name]; exists {
			return fmt.Errorf("duplicate ONNX output name %q", name)
		}
		seen[name] = struct{}{}
	}
	return nil
}

// SelectOnnxOutputs validates a requested ONNX output selection against the complete graph
// output metadata and returns the selected metadata and names in the exact requested order.
//
// Nil or empty requested names preserve all graph outputs and their discovery order.
// Non-empty selections reject blank names, duplicates, and names absent from available.
// Unknown-name errors list every available graph output.
//
// available: complete ONNX graph output metadata discovered before session creation.
// requested: ordered subset of output names to request from ORT; nil or empty means all.
//
// Returns the selected metadata, the ordered session output names, and a validation error.
func SelectOnnxOutputs(available []InputOutputInfo, requested []string) ([]InputOutputInfo, []string, error) {
	if len(requested) == 0 {
		return available, GetNames(available), nil
	}
	if err := ValidateOnnxOutputNameList(requested); err != nil {
		return nil, nil, err
	}

	byName := make(map[string]InputOutputInfo, len(available))
	for _, output := range available {
		byName[output.Name] = output
	}

	selected := make([]InputOutputInfo, 0, len(requested))
	for _, name := range requested {
		output, ok := byName[name]
		if !ok {
			return nil, nil, fmt.Errorf(
				"unknown ONNX output %q; available outputs: %s",
				name,
				strings.Join(GetNames(available), ", "),
			)
		}
		selected = append(selected, output)
	}
	return selected, append([]string(nil), requested...), nil
}
