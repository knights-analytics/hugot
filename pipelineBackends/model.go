package pipelineBackends

import (
	"errors"

	"github.com/knights-analytics/hugot/options"
)

type Model struct {
	Path         string
	OnnxFilename string
	OnnxBytes    []byte
	ORTModel     *ORTModel
	XLAModel     *XLAModel
	Tokenizer    *Tokenizer
	InputsMeta   []InputOutputInfo
	OutputsMeta  []InputOutputInfo
	Destroy      func() error
}

func LoadModel(path string, onnxFilename string, options *options.Options) (*Model, error) {
	model := &Model{
		Path:         path,
		OnnxFilename: onnxFilename,
	}

	err := LoadOnnxModelBytes(model)
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
		switch options.Runtime {
		case "ORT":
			destroyErr = errors.Join(destroyErr, model.ORTModel.Destroy())
		case "XLA":
			model.XLAModel.Destroy()
		}
		return destroyErr
	}
	return model, nil
}
