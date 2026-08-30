package backends

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"sync"

	"github.com/knights-analytics/hugot/backends/modelconfig"
	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/util/fileutil"
)

// ModelMetadata contains model identity and inference metadata.
type ModelMetadata struct {
	ID                    string
	Path                  string
	OnnxFilename          string
	OnnxPath              string
	UnknownToken          string
	InputsMeta            []InputOutputInfo
	OutputsMeta           []InputOutputInfo
	IDLabelMap            map[int]string
	SeparatorToken        string
	MaxPositionEmbeddings int
	IsGenerative          bool
}

// ModelResources owns the backend and tokenizer resources for a model.
type ModelResources struct {
	Backend    Backend
	ORTModel   *ORTModel
	GoMLXModel *GoMLXModel
	Tokenizer  *Tokenizer
	OnnxReader io.ReadCloser
	Pipelines  map[string]Pipeline
}

type Model struct {
	ModelMetadata
	ModelResources
	closeMu sync.Mutex
	closed  bool
}

func LoadModel(ctx context.Context, path string, onnxFilename string, opts *options.Options, isGenerative bool) (*Model, error) {
	model := &Model{
		ID:           path + ":" + onnxFilename,
		Path:         path,
		OnnxFilename: onnxFilename,
		IsGenerative: isGenerative,
		Pipelines:    map[string]Pipeline{},
	}
	backend, backendErr := newBackend(opts)
	if backendErr != nil {
		return nil, backendErr
	}
	model.Backend = backend

	if isGenerative {
		// creation of the session. Only one output (either token or sentence embedding).
		if opts.Backend != options.BackendORT {
			return nil, fmt.Errorf("generative models are only supported with ORT backend currently")
		}
		if onnxFilename != "" {
			return nil, fmt.Errorf("onnx filename should not be provided for generative models as we currently rely on genai_config for the onnx backend")
		}

		err := createORTGenerativeSession(ctx, model, opts)
		if err != nil {
			return nil, errors.Join(err, model.Close())
		}
	} else {
		err := loadModelConfig(ctx, model)
		if err != nil {
			return nil, errors.Join(err, model.Close())
		}
		err = CreateModelBackend(ctx, model, opts)
		if err != nil {
			return nil, errors.Join(err, model.Close())
		}
		tkErr := LoadTokenizer(ctx, model, opts)
		if tkErr != nil {
			return nil, errors.Join(tkErr, model.Close())
		}
	}

	return model, nil
}

// Close releases all resources owned by the model. It is safe to call more
// than once, which makes session and pipeline cleanup composable.
func (model *Model) Close() error {
	model.closeMu.Lock()
	defer model.closeMu.Unlock()
	if model.closed {
		return nil
	}
	model.closed = true

	var closeErr error
	if model.Tokenizer != nil {
		closeErr = errors.Join(closeErr, model.Tokenizer.Close())
		model.Tokenizer = nil
	}
	if model.OnnxReader != nil {
		closeErr = errors.Join(closeErr, model.OnnxReader.Close())
		model.OnnxReader = nil
	}
	if model.ORTModel != nil {
		closeErr = errors.Join(closeErr, model.ORTModel.Close())
		model.ORTModel = nil
	}
	if model.GoMLXModel != nil {
		model.GoMLXModel.Close()
		model.GoMLXModel = nil
	}
	return closeErr
}

func GetOnnxModelPath(ctx context.Context, model *Model) error {
	onnxFiles, err := getOnnxFiles(ctx, model.Path)
	if err != nil {
		return err
	}
	if len(onnxFiles) == 0 {
		return fmt.Errorf("no .onnx file detected at %s. There should be exactly .onnx file", model.Path)
	}
	if len(onnxFiles) > 1 {
		if model.OnnxFilename == "" {
			return fmt.Errorf("multiple .onnx file detected at %s and no OnnxFilename specified", model.Path)
		}
		for i := range onnxFiles {
			if onnxFiles[i][1] == model.OnnxFilename {
				model.OnnxPath = fileutil.PathJoinSafe(onnxFiles[i]...)
				return nil
			}
		}
		return fmt.Errorf("file %s not found at %s", model.OnnxFilename, model.Path)
	}
	model.OnnxPath = fileutil.PathJoinSafe(onnxFiles[0]...)
	return nil
}

func getOnnxFiles(ctx context.Context, path string) ([][]string, error) {
	var onnxFiles [][]string
	walker := func(ctx context.Context, _ string, parent string, info os.FileInfo, _ io.Reader) (toContinue bool, err error) {
		if ctx.Err() != nil {
			return false, ctx.Err()
		}
		if strings.HasSuffix(info.Name(), ".onnx") {
			onnxFiles = append(onnxFiles, []string{parent, info.Name()})
		}
		return true, nil
	}
	err := fileutil.WalkDir(ctx, path, walker)
	return onnxFiles, err
}

func loadModelConfig(ctx context.Context, model *Model) error {
	// load config.json if it exists, to determine max_position_embeddings
	configPath := fileutil.PathJoinSafe(model.Path, "config.json")
	exists, err := fileutil.FileExists(ctx, configPath)
	if err != nil {
		return err
	}
	if exists {
		configBytes, readErr := fileutil.ReadFileBytes(ctx, configPath)
		if readErr != nil {
			return readErr
		}
		var configMap modelconfig.Config
		readErr = json.Unmarshal(configBytes, &configMap)
		if readErr != nil {
			return readErr
		}
		// Some multimodal models store text model config under text_config, so standardise that now
		if configMap.TextConfig != nil {
			configMap.MaxPositionEmbeddings = configMap.TextConfig.MaxPositionEmbeddings
			if configMap.TextConfig.ID2Label != nil {
				configMap.ID2Label = configMap.TextConfig.ID2Label
			}
		}
		if configMap.MaxPositionEmbeddings > 0 {
			model.MaxPositionEmbeddings = configMap.MaxPositionEmbeddings
		}
		if configMap.ID2Label != nil {
			model.IDLabelMap = map[int]string{}
			for k, label := range configMap.ID2Label {
				kInt, kErr := strconv.Atoi(k)
				if kErr != nil {
					return fmt.Errorf("invalid id2label key %q: %w", k, kErr)
				}
				model.IDLabelMap[kInt] = label
			}
		}
	}
	specialTokensPath := fileutil.PathJoinSafe(model.Path, "special_tokens_map.json")
	exists, err = fileutil.FileExists(ctx, specialTokensPath)
	if err != nil {
		return err
	}
	if exists {
		configBytes, readErr := fileutil.ReadFileBytes(ctx, specialTokensPath)
		if readErr != nil {
			return readErr
		}
		var configMap modelconfig.SpecialTokensConfig
		readErr = json.Unmarshal(configBytes, &configMap)
		if readErr != nil {
			return readErr
		}

		if configMap.SepToken.Content != "" {
			model.SeparatorToken = configMap.SepToken.Content
		}
	}
	// Fallback 1: tokenizer_config.json may contain sep_token (common in HF models).
	if model.SeparatorToken == "" {
		tokenizerConfigPath := fileutil.PathJoinSafe(model.Path, "tokenizer_config.json")
		tcExists, tcErr := fileutil.FileExists(ctx, tokenizerConfigPath)
		if tcErr != nil {
			return tcErr
		}
		if tcExists {
			tcBytes, tcReadErr := fileutil.ReadFileBytes(ctx, tokenizerConfigPath)
			if tcReadErr != nil {
				return tcReadErr
			}
			var tcMap modelconfig.TokenizerConfig
			if tcReadErr = json.Unmarshal(tcBytes, &tcMap); tcReadErr != nil {
				return tcReadErr
			}
			if tcMap.SepToken.Content != "" {
				model.SeparatorToken = tcMap.SepToken.Content
			}
			if tcMap.UnknownToken.Content != "" {
				model.UnknownToken = tcMap.UnknownToken.Content
			}
		}
	}
	// Fallback 2: tokenizer.json post_processor.special_tokens may list the separator.
	// We recognise the two canonical HF separators: [SEP] (BERT family) and </s> (RoBERTa family).
	if model.SeparatorToken == "" {
		tokenizerPath := fileutil.PathJoinSafe(model.Path, "tokenizer.json")
		tjExists, tjErr := fileutil.FileExists(ctx, tokenizerPath)
		if tjErr != nil {
			return tjErr
		}
		if tjExists {
			tjBytes, tjReadErr := fileutil.ReadFileBytes(ctx, tokenizerPath)
			if tjReadErr != nil {
				return tjReadErr
			}
			var tjMap modelconfig.TokenizerJSONConfig
			if tjReadErr = json.Unmarshal(tjBytes, &tjMap); tjReadErr != nil {
				return tjReadErr
			}
			for _, candidate := range []string{"[SEP]", "</s>"} {
				if _, found := tjMap.PostProcessor.SpecialTokens[candidate]; found {
					model.SeparatorToken = candidate
					break
				}
			}
		}
	}
	return nil
}

func ReshapeOutput[T float32 | int64 | int32](input []T, meta InputOutputInfo, batchSize int, paddingMask [][]bool, sequenceLength int) any {
	var outArray any
	dimensions := meta.Dimensions.ValuesInt()
	lenDimensions := len(dimensions)
	switch lenDimensions {
	case 1:
		return input
	case 2:
		outArray = flatDataTo2D(input, batchSize, dimensions[lenDimensions-1])
	case 3:
		// If no padding mask is provided (vision models), infer middle dim.
		if len(paddingMask) == 0 || sequenceLength == 0 {
			outArray = flatDataTo3DGeneric(input, batchSize, dimensions[lenDimensions-1])
		} else {
			outArray = flatDataTo3D(input, paddingMask, sequenceLength, dimensions[lenDimensions-1])
		}
	case 4:
		dimension := dimensions[3]
		groupSize := dimensions[1]
		outArray = flatDataTo4D(input, paddingMask, groupSize, dimension)
	}
	return outArray
}

func flatDataTo2D[T float32 | int64 | int32](input []T, batchSize int, dimension int) [][]T {
	// Input string, token, dimension
	output := make([][]T, batchSize)
	if dimension == -1 {
		// it can happen in principle that the embedding dimension is -1 if it was so exported from onnx even though there
		// is a fixed out dimension so we do this.
		dimension = len(input) / batchSize
	}
	counter := 0
	for batchIndex := range batchSize {
		inputEmbedding := make([]T, dimension)
		for i := 0; i < dimension; i++ {
			inputEmbedding[i] = input[counter]
			counter++
		}
		output[batchIndex] = inputEmbedding
	}
	return output
}

func flatDataTo3D[T float32 | int64 | int32](input []T, paddingMask [][]bool, sequenceLength int, dimension int) [][][]T {
	// Input string, token, dimension
	output := make([][][]T, len(paddingMask))
	counter := 0
	for batchIndex, mask := range paddingMask {
		tokenEmbeddings := make([][]T, 0, sequenceLength)
		for _, isValid := range mask {
			if !isValid {
				// skip whole token
				counter = counter + dimension
				continue
			}
			// valid token, create embedding
			embedding := make([]T, dimension)
			for i := range dimension {
				embedding[i] = input[counter]
				counter++
			}
			tokenEmbeddings = append(tokenEmbeddings, embedding)
		}
		output[batchIndex] = tokenEmbeddings
	}
	return output
}

// flatDataTo3DGeneric reshapes flat data into [batchSize][N][dimension] inferring N.
func flatDataTo3DGeneric[T float32 | int64 | int32](input []T, batchSize int, dimension int) [][][]T {
	if dimension == -1 {
		// cannot infer without last dimension; return empty
		return make([][][]T, batchSize)
	}
	total := len(input)
	if batchSize <= 0 || dimension <= 0 || total == 0 {
		return make([][][]T, batchSize)
	}
	perBatch := total / batchSize
	if perBatch%dimension != 0 {
		// fallback: best-effort
		perBatch = (perBatch / dimension) * dimension
	}
	n := perBatch / dimension
	output := make([][][]T, batchSize)
	idx := 0
	for b := range batchSize {
		seq := make([][]T, n)
		for i := range n {
			vec := make([]T, dimension)
			for d := range dimension {
				vec[d] = input[idx]
				idx++
			}
			seq[i] = vec
		}
		output[b] = seq
	}
	return output
}

func flatDataTo4D[T float32 | int64 | int32](input []T, paddingMask [][]bool, groupSize int, dimension int) [][][][]T {
	batchSize := len(paddingMask) // B
	if batchSize == 0 || groupSize <= 0 || dimension <= 0 {
		return make([][][][]T, batchSize)
	}
	sequenceLength := len(paddingMask[0]) // S
	output := make([][][][]T, batchSize)
	counter := 0
	for b := range batchSize {
		group := make([][][]T, groupSize) // A
		for a := range groupSize {
			sequence := make([][]T, sequenceLength)
			for s := range sequenceLength {
				if !paddingMask[b][s] {
					// skip this entire vector
					counter += dimension
					sequence[s] = make([]T, dimension) // fill with zeros or ignore
					continue
				}
				vector := make([]T, dimension)
				for d := range dimension {
					vector[d] = input[counter]
					counter++
				}
				sequence[s] = vector
			}
			group[a] = sequence
		}
		output[b] = group
	}
	return output
}
