package pipelines

import (
	"errors"
	"fmt"
	"math"

	"github.com/knights-analytics/hugot/backends"
)

// RawFeatureOutput retains the selected model output without pooling or normalization.
// HiddenStates is [batch, sequence, hidden] for rank-three outputs; Embeddings is
// [batch, hidden] for rank-two outputs. Exactly one of these fields is populated.
// Dimensions contains the actual tensor shape, including padding tokens.
type RawFeatureOutput struct {
	OutputName   string
	Dimensions   backends.Shape
	HiddenStates [][][]float32
	Embeddings   [][]float32
}

func (o *RawFeatureOutput) GetOutput() []any {
	if o.HiddenStates != nil {
		out := make([]any, len(o.HiddenStates))
		for i, features := range o.HiddenStates {
			out[i] = features
		}
		return out
	}
	out := make([]any, len(o.Embeddings))
	for i, features := range o.Embeddings {
		out[i] = features
	}
	return out
}

func modelPoolerIndex(model *backends.Model) (int, error) {
	if model != nil {
		for i, output := range model.OutputsMeta {
			if output.Name == "pooler_output" {
				if len(output.Dimensions) != 2 {
					return 0, errors.New("model pooler_output must have rank two")
				}
				return i, nil
			}
		}
	}
	return 0, errors.New("model does not expose pooler_output; mean pooling is not a model pooler")
}

func rawFeatures(batch *backends.PipelineBatch, output backends.InputOutputInfo, index int) (*RawFeatureOutput, error) {
	if batch == nil || batch.Size <= 0 {
		return nil, errors.New("raw features require a nonempty batch")
	}
	if index < 0 || index >= len(batch.OutputValues) {
		return nil, errors.New("raw features model returned no selected output")
	}
	result := &RawFeatureOutput{OutputName: output.Name}
	var rows [][]float32
	switch value := batch.OutputValues[index].(type) {
	case [][]float32:
		if len(value) != batch.Size || len(value[0]) == 0 {
			return nil, errors.New("raw feature batch or hidden dimension is invalid")
		}
		result.Dimensions = backends.Shape{int64(len(value)), int64(len(value[0]))}
		rows = value
	case [][][]float32:
		if len(value) != batch.Size || len(value[0]) == 0 || len(value[0][0]) == 0 {
			return nil, errors.New("raw feature batch, sequence or hidden dimension is invalid")
		}
		result.Dimensions = backends.Shape{int64(len(value)), int64(len(value[0])), int64(len(value[0][0]))}
		for _, tokens := range value {
			if len(tokens) != len(value[0]) {
				return nil, errors.New("raw feature sequences have inconsistent dimensions")
			}
			rows = append(rows, tokens...)
		}
	default:
		return nil, fmt.Errorf("raw feature output type %T is not supported", batch.OutputValues[index])
	}
	if len(output.Dimensions) != len(result.Dimensions) {
		return nil, errors.New("raw feature rank does not match model metadata")
	}
	for i, dimension := range output.Dimensions {
		if dimension != -1 && dimension != result.Dimensions[i] {
			return nil, fmt.Errorf("raw feature dimension %d does not match model metadata", i)
		}
	}
	hidden := int(result.Dimensions[len(result.Dimensions)-1])
	copied := make([][]float32, len(rows))
	for i, row := range rows {
		if len(row) != hidden {
			return nil, errors.New("raw feature vectors have inconsistent dimensions")
		}
		for _, value := range row {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return nil, errors.New("raw feature values must be finite")
			}
		}
		copied[i] = append([]float32(nil), row...)
	}
	if len(result.Dimensions) == 2 {
		result.Embeddings = copied
	} else {
		sequence := int(result.Dimensions[1])
		result.HiddenStates = make([][][]float32, batch.Size)
		for i := range result.HiddenStates {
			result.HiddenStates[i] = copied[i*sequence : (i+1)*sequence]
		}
	}
	return result, nil
}
