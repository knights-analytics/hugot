package pipelineBackends

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"

	jsoniter "github.com/json-iterator/go"

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
		readErr = jsoniter.Unmarshal(configBytes, &configMap)
		if readErr != nil {
			return readErr
		}

		if maxPositionEmbeddingsRaw, existsOk := configMap["max_position_embeddings"]; existsOk {
			if maxPositionEmbeddings, castOk := maxPositionEmbeddingsRaw.(float64); castOk {
				model.MaxPositionEmbeddings = int(maxPositionEmbeddings)
			}
		}
	}
	return nil
}

func ReshapeOutput(input *[]float32, meta InputOutputInfo, paddingMask [][]bool, sequenceLength int) OutputArray {
	outArray := OutputArray{}

	dimensions := meta.Dimensions.ValuesInt()
	lenDimensions := len(dimensions)
	switch lenDimensions {
	case 2:
		outArray.Result2D = flatDataTo2D(input, paddingMask, dimensions[lenDimensions-1])
	case 3:
		outArray.Result3D = flatDataTo3D(input, paddingMask, sequenceLength, dimensions[lenDimensions-1])
	}
	return outArray
}

func flatDataTo2D(input *[]float32, paddingMask [][]bool, dimension int) [][]float32 {
	// Input string, token, dimension
	output := make([][]float32, len(paddingMask))

	counter := 0
	for batchIndex := range paddingMask {
		inputEmbedding := make([]float32, dimension)

		for i := 0; i < dimension; i++ {
			inputEmbedding[i] = (*input)[counter]
			counter++
		}

		output[batchIndex] = inputEmbedding
	}

	return output
}

func flatDataTo3D(input *[]float32, paddingMask [][]bool, sequenceLength int, dimension int) [][][]float32 {
	// Input string, token, dimension
	output := make([][][]float32, len(paddingMask))

	counter := 0

	for batchIndex, mask := range paddingMask {
		tokenEmbeddings := make([][]float32, 0, sequenceLength)

		for _, isValid := range mask {
			if !isValid {
				// skip whole token
				counter = counter + dimension
				continue
			}

			// valid token, create embedding
			embedding := make([]float32, dimension)

			for i := 0; i < dimension; i++ {
				embedding[i] = (*input)[counter]
				counter++
			}

			tokenEmbeddings = append(tokenEmbeddings, embedding)
		}

		output[batchIndex] = tokenEmbeddings
	}

	return output
}
