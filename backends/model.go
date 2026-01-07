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

	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/util/fileutil"
)

type Model struct {
	ID                    string
	ORTModel              *ORTModel
	GoMLXModel            *GoMLXModel
	Tokenizer             *Tokenizer
	Destroy               func() error
	Pipelines             map[string]Pipeline
	IDLabelMap            map[int]string
	EosTokenIDs           map[int64]bool
	SeparatorToken        string
	Path                  string
	OnnxFilename          string
	OnnxPath              string
	OnnxBytes             []byte
	InputsMeta            []InputOutputInfo
	OutputsMeta           []InputOutputInfo
	MaxPositionEmbeddings int
	NumHiddenLayers       int // Number of key value heads, used for text generation
	NumKeyValueHeads      int
	HeadDim               int
	FixedCacheSize        int
	VocabSize             int
	PadToken              int64
	IsGenerative          bool
}

func LoadModel(path string, onnxFilename string, options *options.Options, isGenerative bool) (*Model, error) {
	model := &Model{
		ID:           path + ":" + onnxFilename,
		Path:         path,
		OnnxFilename: onnxFilename,
		Pipelines:    map[string]Pipeline{},
		IsGenerative: isGenerative,
	}

	if !isGenerative {
		err := loadModelConfig(model)
		if err != nil {
			return nil, err
		}
		err = CreateModelBackend(model, options)
		if err != nil {
			return nil, err
		}
		tkErr := LoadTokenizer(model, options)
		if tkErr != nil {
			return nil, tkErr
		}
	} else {
		// creation of the session. Only one output (either token or sentence embedding).
		if options.Backend != "ORT" {
			return nil, fmt.Errorf("generative models are only supported with ORT backend currently")
		}
		if onnxFilename != "" {
			return nil, fmt.Errorf("onnx filename should not be provided for generative models as we currently rely on genai_config for the onnx backend")
		}
		err := createORTGenerativeSession(model, options)
		if err != nil {
			return nil, err
		}
	}

	model.Destroy = func() error {
		var destroyErr error
		if model.Tokenizer != nil {
			destroyErr = model.Tokenizer.Destroy()
		}
		switch options.Backend {
		case "ORT":
			destroyErr = errors.Join(destroyErr, model.ORTModel.Destroy())
			model.ORTModel = nil
		case "GO", "XLA":
			model.GoMLXModel.Destroy()
			model.GoMLXModel = nil
		}
		return destroyErr
	}
	return model, nil
}

func GetOnnxModelPath(model *Model) error {
	onnxFiles, err := getOnnxFiles(model.Path)
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

func getOnnxFiles(path string) ([][]string, error) {
	var onnxFiles [][]string
	walker := func(_ context.Context, _ string, parent string, info os.FileInfo, _ io.Reader) (toContinue bool, err error) {
		if strings.HasSuffix(info.Name(), ".onnx") {
			onnxFiles = append(onnxFiles, []string{parent, info.Name()})
		}
		return true, nil
	}
	err := fileutil.WalkDir()(context.Background(), path, walker)
	return onnxFiles, err
}

func loadModelConfig(model *Model) error {
	// load config.json if it exists, to determine max_position_embeddings
	configPath := fileutil.PathJoinSafe(model.Path, "config.json")
	exists, err := fileutil.FileExists(configPath)
	if err != nil {
		return err
	}
	if exists {
		configBytes, readErr := fileutil.ReadFileBytes(configPath)
		if readErr != nil {
			return readErr
		}
		configMap := map[string]any{}
		readErr = json.Unmarshal(configBytes, &configMap)
		if readErr != nil {
			return readErr
		}
		if maxPositionEmbeddingsRaw, existsOk := configMap["max_position_embeddings"]; existsOk {
			if maxPositionEmbeddings, castOk := maxPositionEmbeddingsRaw.(float64); castOk {
				model.MaxPositionEmbeddings = int(maxPositionEmbeddings)
			}
		}
		if padTokenRaw, existsOk := configMap["pad_token_id"]; existsOk {
			if padToken, castOk := padTokenRaw.(float64); castOk {
				model.PadToken = int64(padToken)
			}
		}
		if id2LabelRaw, existsOk := configMap["id2label"]; existsOk {
			if id2Label, castOk := id2LabelRaw.(map[string]any); castOk {
				id2labelCast := map[int]string{}
				for k, v := range id2Label {
					kInt, kErr := strconv.Atoi(k)
					if kErr != nil {
						return kErr
					}
					id2labelCast[kInt] = v.(string)
				}
				model.IDLabelMap = id2labelCast
			} else {
				return fmt.Errorf("id2label is not a map")
			}
		}
		if eosRaw, exists := configMap["eos_token_id"]; exists {
			model.EosTokenIDs = map[int64]bool{}
			switch v := eosRaw.(type) {
			case []any:
				for i, item := range v {
					if num, ok := item.(float64); ok {
						model.EosTokenIDs[int64(num)] = true
					} else {
						return fmt.Errorf("eos_token_id contains non-numeric value at index %d", i)
					}
				}
			case float64:
				model.EosTokenIDs[int64(v)] = true
			default:
				return errors.New("eos_token_id must be either a number or an array of numbers")
			}
		}
		if numHiddenLayersRaw, exists := configMap["num_hidden_layers"]; exists {
			if numHiddenLayersFloat, ok := numHiddenLayersRaw.(float64); ok {
				model.NumHiddenLayers = int(numHiddenLayersFloat)
			} else {
				return errors.New("num_hidden_layers is not a number")
			}
		}
		if numKeyValueHeads, exists := configMap["num_key_value_heads"]; exists {
			if numKeyValueHeadsValue, ok := numKeyValueHeads.(float64); ok {
				model.NumKeyValueHeads = int(numKeyValueHeadsValue)
			} else {
				return errors.New("num_key_value_heads is not a number")
			}
		}
		if headDim, exists := configMap["head_dim"]; exists {
			if headDimValue, ok := headDim.(float64); ok {
				model.HeadDim = int(headDimValue)
			} else {
				return errors.New("num_key_value_heads is not a number")
			}
		}
		if vocabSize, exists := configMap["vocab_size"]; exists {
			if vocabSizeValue, ok := vocabSize.(float64); ok {
				model.VocabSize = int(vocabSizeValue)
			} else {
				return errors.New("vocab_size is not a number")
			}
		}
	}
	specialTokensPath := fileutil.PathJoinSafe(model.Path, "special_tokens_map.json")
	exists, err = fileutil.FileExists(specialTokensPath)
	if err != nil {
		return err
	}
	if exists {
		configBytes, readErr := fileutil.ReadFileBytes(specialTokensPath)
		if readErr != nil {
			return readErr
		}
		var configMap map[string]interface{}
		readErr = json.Unmarshal(configBytes, &configMap)
		if readErr != nil {
			return readErr
		}
		if sepToken, exists := configMap["sep_token"]; exists {
			switch v := sepToken.(type) {
			case map[string]interface{}:
				t, contentOk := v["content"]
				if !contentOk {
					return fmt.Errorf("sep_token is map but no content field is available")
				}
				tString, stringOk := t.(string)
				if !stringOk {
					return fmt.Errorf("sep_token cannot be converted to string: %v", t)
				}
				model.SeparatorToken = tString
			case string:
				model.SeparatorToken = v
			default:
				return fmt.Errorf("sep_token has unexpected type: %v", v)
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
			for i := 0; i < dimension; i++ {
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
	batchSize := len(paddingMask)         // B
	sequenceLength := len(paddingMask[0]) // S
	output := make([][][][]T, batchSize)
	counter := 0
	for b := 0; b < batchSize; b++ {
		group := make([][][]T, groupSize) // A
		for a := 0; a < groupSize; a++ {
			sequence := make([][]T, sequenceLength)
			for s := 0; s < sequenceLength; s++ {
				if !paddingMask[b][s] {
					// skip this entire vector
					counter += dimension
					sequence[s] = make([]T, dimension) // fill with zeros or ignore
					continue
				}
				vector := make([]T, dimension)
				for d := 0; d < dimension; d++ {
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
