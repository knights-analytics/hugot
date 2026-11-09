//go:build cgo && (ORT || ALL)

package backends

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"path"
	"path/filepath"
	"sort"
	"strings"

	"github.com/gomlx/compute-onnx/support/protos"
	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/util/fileutil"
	"github.com/knights-analytics/ortgenai"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"
)

type genaiMemoryGraphFile struct {
	name       string
	bundlePath string
}

// createORTGenerativeMemorySession explicitly uses the context-bound adapter for
// every model artifact. The native constructor copies graph and weight bytes;
// the temporary Go buffers can be released as soon as it returns.
func createORTGenerativeMemorySession(ctx context.Context, model *Model, opts *options.Options) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := validateMemoryGenAIOptions(opts.ORTOptions); err != nil {
		return err
	}

	bundleCache := map[string][]byte{}
	read := func(name string) ([]byte, error) {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		bundlePath, err := normalizeBundleReference(name)
		if err != nil {
			return nil, fmt.Errorf("invalid model bundle file reference %q: %w", name, err)
		}
		if cached, ok := bundleCache[bundlePath]; ok {
			return cached, nil
		}
		filename := fileutil.PathJoinSafe(model.Path, bundlePath)
		data, err := fileutil.ReadFileBytes(ctx, filename)
		if err != nil {
			return nil, fmt.Errorf("reading model bundle file %q: %w", name, err)
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		bundleCache[bundlePath] = data
		return data, nil
	}

	config, err := read("genai_config.json")
	if err != nil {
		return err
	}
	tokenizerConfig, err := read("tokenizer_config.json")
	if err != nil {
		return err
	}
	tokenizerJSON, err := read("tokenizer.json")
	if err != nil {
		return err
	}
	graphFiles, err := memoryConfigGraphFiles(config)
	if err != nil {
		return err
	}
	modelData := make(map[string][]byte, len(graphFiles))
	externalData := map[string][]byte{}
	externalLocations := map[string]string{}
	for _, graphFile := range graphFiles {
		graphData, readErr := read(graphFile.bundlePath)
		if readErr != nil {
			return readErr
		}
		modelData[graphFile.name] = graphData
		references, parseErr := onnxExternalDataLocations(graphData)
		if parseErr != nil {
			return fmt.Errorf("inspecting external data references in graph %q: %w", graphFile.name, parseErr)
		}
		for _, location := range references {
			bundleName, resolveErr := resolveGraphRelativeBundlePath(graphFile.bundlePath, location)
			if resolveErr != nil {
				return fmt.Errorf("invalid external data reference %q in graph %q: %w", location, graphFile.name, resolveErr)
			}
			if prior, exists := externalLocations[location]; exists {
				if prior != bundleName {
					return fmt.Errorf("external data filename %q is referenced by multiple bundle files (%q and %q)", location, prior, bundleName)
				}
				continue
			}
			data, readErr := read(bundleName)
			if readErr != nil {
				return readErr
			}
			externalLocations[location] = bundleName
			externalData[location] = data
		}
	}

	// Hugging Face may store the Jinja template as a standalone file. GenAI's
	// memory API accepts tokenizer_config.json and tokenizer.json only, so carry
	// the template in the tokenizer config when that config does not already have
	// one (the directory loader's embedded setting remains authoritative).
	templatePath := fileutil.PathJoinSafe(model.Path, "chat_template.jinja")
	hasTemplate, err := fileutil.FileExists(ctx, templatePath)
	if err != nil {
		return fmt.Errorf("checking model bundle file %q: %w", "chat_template.jinja", err)
	}
	if hasTemplate {
		if err := ctx.Err(); err != nil {
			return err
		}
		template, readErr := fileutil.ReadFileBytes(ctx, templatePath)
		if readErr != nil {
			return fmt.Errorf("reading model bundle file %q: %w", "chat_template.jinja", readErr)
		}
		tokenizerConfig, err = mergeStandaloneChatTemplate(tokenizerConfig, template)
		if err != nil {
			return fmt.Errorf("merging standalone chat template: %w", err)
		}
	}

	if err := ctx.Err(); err != nil {
		return err
	}
	if err := initialiseORTGenAI(ctx, opts); err != nil {
		return err
	}
	session, err := ortgenai.CreateSessionFromMemory(config, modelData, externalData, tokenizerConfig, tokenizerJSON)
	if err != nil {
		return fmt.Errorf("error creating ortgenai in-memory session: %w", err)
	}
	model.ORTModel = &ORTModel{
		Generative: &generativeORTProvider{session: session},
		Options:    opts.ORTOptions,
	}
	return nil
}

func mergeStandaloneChatTemplate(tokenizerConfig, template []byte) ([]byte, error) {
	var configObject map[string]json.RawMessage
	if err := json.Unmarshal(tokenizerConfig, &configObject); err != nil {
		return nil, fmt.Errorf("decoding tokenizer_config.json: %w", err)
	}
	if existing, present := configObject["chat_template"]; present {
		var value string
		if err := json.Unmarshal(existing, &value); err != nil || value != "" {
			return tokenizerConfig, nil
		}
	}
	configObject["chat_template"], _ = json.Marshal(string(template))
	encoded, err := json.Marshal(configObject)
	if err != nil {
		return nil, fmt.Errorf("encoding tokenizer_config.json: %w", err)
	}
	return encoded, nil
}

func validateMemoryGenAIOptions(opts *options.OrtOptions) error {
	if opts == nil {
		return errors.New("ORT options must be provided to create a generative model")
	}
	if opts.UseEngine {
		return errors.New("in-memory GenAI loading does not support WithGenerativeEngine")
	}
	if opts.UseMTP != nil && *opts.UseMTP {
		return errors.New("in-memory GenAI loading does not support MTP because MTP requires a model directory")
	}
	if opts.CudaOptions != nil || opts.CoreMLOptions != nil || opts.DirectMLOptions != nil || opts.OpenVINOOptions != nil ||
		opts.TensorRTOptions != nil || opts.NvTensorRTRTXOptions != nil || len(opts.ExtraExecutionProviders) > 0 || opts.GenAIGPUDeviceID != nil {
		return errors.New("in-memory GenAI loading supports CPU only; GPU and custom execution provider options are not supported")
	}
	if len(opts.GenAIAdapters) > 0 || opts.GenAIActiveAdapter != nil {
		return errors.New("in-memory GenAI loading does not support adapters because adapter files are path-based")
	}
	if opts.Telemetry != nil || opts.IntraOpNumThreads != nil || opts.InterOpNumThreads != nil || opts.CPUMemArena != nil ||
		opts.MemPattern != nil || opts.ParallelExecutionMode != nil || opts.IntraOpSpinning != nil || opts.InterOpSpinning != nil ||
		opts.LogSeverityLevel != nil || opts.GraphOptimizationLevel != nil || opts.OptimizedModelFilePath != nil ||
		opts.ProfilingEnabled != nil || opts.ProfilingFilePrefix != nil {
		return errors.New("in-memory GenAI loading does not support session-specific ORT options")
	}
	return nil
}

func memoryConfigGraphFiles(configJSON []byte) ([]genaiMemoryGraphFile, error) {
	var config map[string]any
	if err := json.Unmarshal(configJSON, &config); err != nil {
		return nil, fmt.Errorf("decoding generative model config: %w", err)
	}
	if err := rejectUnsupportedMemoryConfig(config, "genai_config.json"); err != nil {
		return nil, err
	}
	modelConfig, ok := config["model"].(map[string]any)
	if !ok {
		return nil, errors.New("in-memory GenAI loading requires an object-valued model config")
	}
	files := map[string]string{}
	var collect func(any) error
	collect = func(value any) error {
		switch value := value.(type) {
		case map[string]any:
			for key, child := range value {
				if strings.EqualFold(key, "filename") {
					filename, ok := child.(string)
					if !ok || filename == "" {
						return errors.New("model graph filename must be a non-empty string")
					}
					normalized, err := normalizeBundleReference(filename)
					if err != nil {
						return fmt.Errorf("invalid model graph reference %q: %w", filename, err)
					}
					files[filename] = normalized
					continue
				}
				if err := collect(child); err != nil {
					return err
				}
			}
		case []any:
			for _, child := range value {
				if err := collect(child); err != nil {
					return err
				}
			}
		}
		return nil
	}
	if err := collect(modelConfig); err != nil {
		return nil, err
	}
	if len(files) == 0 {
		return nil, errors.New("in-memory GenAI loading found no model graph filenames")
	}
	result := make([]genaiMemoryGraphFile, 0, len(files))
	for filename, bundlePath := range files {
		result = append(result, genaiMemoryGraphFile{name: filename, bundlePath: bundlePath})
	}
	// Keep order stable for repeatable reads and diagnostics.
	sort.Slice(result, func(i, j int) bool { return result[i].name < result[j].name })
	return result, nil
}

func rejectUnsupportedMemoryConfig(value any, location string) error {
	switch value := value.(type) {
	case map[string]any:
		for key, child := range value {
			lower := strings.ToLower(key)
			if containsAny(lower, "vision", "image", "audio", "speech", "video", "multimodal", "embedding", "mtp") {
				return fmt.Errorf("in-memory GenAI loading does not support multimodal or MTP config field %q", key)
			}
			if strings.Contains(lower, "provider") || lower == "execution_provider" || lower == "execution_providers" {
				if !isCPUProviderConfig(child) {
					return fmt.Errorf("in-memory GenAI loading supports CPU provider settings only; config field %q is incompatible", key)
				}
			}
			if strings.Contains(lower, "gpu") || strings.Contains(lower, "device_id") {
				return fmt.Errorf("in-memory GenAI loading does not support GPU/device config field %q", key)
			}
			if lower == "device_type" {
				device, _ := child.(string)
				if !strings.EqualFold(device, "cpu") {
					return fmt.Errorf("in-memory GenAI loading supports CPU device_type only, got %q", device)
				}
			}
			if err := rejectUnsupportedMemoryConfig(child, location+"."+key); err != nil {
				return err
			}
		}
	case []any:
		for i, child := range value {
			if err := rejectUnsupportedMemoryConfig(child, fmt.Sprintf("%s[%d]", location, i)); err != nil {
				return err
			}
		}
	}
	return nil
}

func isCPUProviderConfig(value any) bool {
	switch value := value.(type) {
	case string:
		return strings.EqualFold(value, "cpu") || strings.EqualFold(value, "cpuexecutionprovider")
	case []any:
		for _, item := range value {
			if !isCPUProviderConfig(item) {
				return false
			}
		}
		return len(value) > 0
	case map[string]any:
		for key, item := range value {
			if strings.EqualFold(key, "cpu") || strings.EqualFold(key, "cpuexecutionprovider") {
				continue
			}
			if key == "provider" || key == "name" || key == "execution_provider" {
				if !isCPUProviderConfig(item) {
					return false
				}
				continue
			}
			if !isCPUProviderConfig(item) {
				return false
			}
		}
		return len(value) > 0
	default:
		return false
	}
}

func normalizeBundleReference(reference string) (string, error) {
	if reference == "" || strings.Contains(reference, "\\") || strings.Contains(reference, ":") || path.IsAbs(reference) || filepath.IsAbs(reference) {
		return "", errors.New("reference must be a relative bundle path")
	}
	clean := path.Clean(reference)
	if clean == "." || clean == ".." || strings.HasPrefix(clean, "../") {
		return "", errors.New("reference escapes the model bundle")
	}
	return clean, nil
}

func resolveGraphRelativeBundlePath(graphName, location string) (string, error) {
	if location == "" || strings.Contains(location, "\\") || strings.Contains(location, ":") || path.IsAbs(location) || filepath.IsAbs(location) {
		return "", errors.New("reference must be a relative bundle path")
	}
	resolved := path.Clean(path.Join(path.Dir(graphName), location))
	if resolved == "." || resolved == ".." || strings.HasPrefix(resolved, "../") {
		return "", errors.New("reference escapes the model bundle")
	}
	return resolved, nil
}

func onnxExternalDataLocations(graph []byte) ([]string, error) {
	var model protos.ModelProto
	if err := proto.Unmarshal(graph, &model); err != nil {
		return nil, err
	}
	locations := map[string]struct{}{}
	var visit func(protoreflect.Message)
	visit = func(message protoreflect.Message) {
		if !message.IsValid() {
			return
		}
		if string(message.Descriptor().Name()) == "TensorProto" {
			var location string
			externalDataField := message.Descriptor().Fields().ByName("external_data")
			entries := message.Get(externalDataField).List()
			for i := 0; i < entries.Len(); i++ {
				entry := entries.Get(i).Message()
				keyField := entry.Descriptor().Fields().ByName("key")
				valueField := entry.Descriptor().Fields().ByName("value")
				if entry.Get(keyField).String() == "location" {
					location = entry.Get(valueField).String()
					break
				}
			}
			if location != "" {
				locations[location] = struct{}{}
			} else {
				dataLocationField := message.Descriptor().Fields().ByName("data_location")
				if message.Get(dataLocationField).Enum() == protoreflect.EnumNumber(protos.TensorProto_EXTERNAL) {
					// Validate malformed external tensors rather than allowing a
					// silent fallthrough that omits their weight file.
					locations[""] = struct{}{}
				}
			}
		}
		fields := message.Descriptor().Fields()
		for i := 0; i < fields.Len(); i++ {
			field := fields.Get(i)
			if field.Kind() != protoreflect.MessageKind && field.Kind() != protoreflect.GroupKind {
				continue
			}
			value := message.Get(field)
			if field.IsList() {
				list := value.List()
				for j := 0; j < list.Len(); j++ {
					visit(list.Get(j).Message())
				}
			} else if field.IsMap() {
				value.Map().Range(func(_ protoreflect.MapKey, mapValue protoreflect.Value) bool {
					if field.MapValue().Kind() == protoreflect.MessageKind {
						visit(mapValue.Message())
					}
					return true
				})
			} else if value.Message().IsValid() {
				visit(value.Message())
			}
		}
	}
	visit(model.ProtoReflect())
	result := make([]string, 0, len(locations))
	for location := range locations {
		if location == "" {
			return nil, errors.New("external tensor is missing its location")
		}
		result = append(result, location)
	}
	sort.Strings(result)
	return result, nil
}

func containsAny(value string, needles ...string) bool {
	for _, needle := range needles {
		if strings.Contains(value, needle) {
			return true
		}
	}
	return false
}
