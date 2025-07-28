package pipelineBackends

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
	"github.com/knights-analytics/hugot/util"
)

type Model struct {
	Path                  string
	OnnxFilename          string
	OnnxBytes             []byte
	ORTModel              *ORTModel
	GoMLXModel            *GoMLXModel
	Tokenizer             *Tokenizer
	InputsMeta            []InputOutputInfo
	OutputsMeta           []InputOutputInfo
	Destroy               func() error
	Pipelines             map[string]Pipeline
	MaxPositionEmbeddings int
	IDLabelMap            map[int]string
	SeparatorToken        string
}

func LoadModel(path string, onnxFilename string, options *options.Options) (*Model, error) {
	model := &Model{
		Path:         path,
		OnnxFilename: onnxFilename,
		Pipelines:    make(map[string]Pipeline),
	}

	err := LoadOnnxModelBytes(model)
	if err != nil {
		return nil, err
	}

	err = loadModelConfig(model)
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

	model.Destroy = func() error {
		destroyErr := model.Tokenizer.Destroy()
		switch options.Backend {
		case "ORT":
			destroyErr = errors.Join(destroyErr, model.ORTModel.Destroy())
		case "GO", "XLA":
			model.GoMLXModel.Destroy()
		}
		return destroyErr
	}
	return model, nil
}

func LoadOnnxModelBytes(model *Model) error {
	var modelOnnxFile string
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
		modelNameFound := false
		for i := range onnxFiles {
			if onnxFiles[i][1] == model.OnnxFilename {
				modelNameFound = true
				modelOnnxFile = util.PathJoinSafe(onnxFiles[i]...)
			}
		}
		if !modelNameFound {
			return fmt.Errorf("file %s not found at %s", model.OnnxFilename, model.Path)
		}
	} else {
		modelOnnxFile = util.PathJoinSafe(onnxFiles[0]...)
	}

	onnxBytes, err := util.ReadFileBytes(modelOnnxFile)
	if err != nil {
		return err
	}

	model.OnnxBytes = onnxBytes

	return err
}

func getOnnxFiles(path string) ([][]string, error) {
	var onnxFiles [][]string
	walker := func(_ context.Context, _ string, parent string, info os.FileInfo, _ io.Reader) (toContinue bool, err error) {
		if strings.HasSuffix(info.Name(), ".onnx") {
			onnxFiles = append(onnxFiles, []string{util.PathJoinSafe(path, parent), info.Name()})
		}
		return true, nil
	}
	err := util.WalkDir()(context.Background(), path, walker)
	return onnxFiles, err
}

func loadModelConfig(model *Model) error {

	// load config.json if it exists, to determine max_position_embeddings
	configPath := util.PathJoinSafe(model.Path, "config.json")

	exists, err := util.FileExists(configPath)
	if err != nil {
		return err
	}
	if exists {
		configBytes, readErr := util.ReadFileBytes(configPath)
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
	}

	specialTokensPath := util.PathJoinSafe(model.Path, "special_tokens_map.json")

	exists, err = util.FileExists(specialTokensPath)
	if err != nil {
		return err
	}
	if exists {
		configBytes, readErr := util.ReadFileBytes(specialTokensPath)
		if readErr != nil {
			return readErr
		}

		var configMap map[string]interface{}
		readErr = json.Unmarshal(configBytes, &configMap)
		if readErr != nil {
			return readErr
		}

		sepToken, ok := configMap["sep_token"]
		if !ok {
			return fmt.Errorf("no sep token detected in special_tokens_map.json at %s", model.Path)
		}

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

	return nil
}

func ReshapeOutput[T float32 | int64](input []T, meta InputOutputInfo, paddingMask [][]bool, sequenceLength int) any {

	var outArray any
	dimensions := meta.Dimensions.ValuesInt()
	lenDimensions := len(dimensions)
	switch lenDimensions {
	case 2:
		outArray = flatDataTo2D(input, paddingMask, dimensions[lenDimensions-1])
	case 3:
		outArray = flatDataTo3D(input, paddingMask, sequenceLength, dimensions[lenDimensions-1])
	}
	return outArray
}

func flatDataTo2D[T float32 | int64](input []T, paddingMask [][]bool, dimension int) [][]T {
	// Input string, token, dimension
	output := make([][]T, len(paddingMask))

	counter := 0
	for batchIndex := range paddingMask {
		inputEmbedding := make([]T, dimension)

		for i := 0; i < dimension; i++ {
			inputEmbedding[i] = input[counter]
			counter++
		}

		output[batchIndex] = inputEmbedding
	}

	return output
}

func flatDataTo3D[T float32 | int64](input []T, paddingMask [][]bool, sequenceLength int, dimension int) [][][]T {
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
