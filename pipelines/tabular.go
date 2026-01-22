package pipelines

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync/atomic"
	"time"

	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/util/safeconv"
	"github.com/knights-analytics/hugot/util/vectorutil"
)

// TabularPipeline supports classic ML models (e.g., decision trees, random forests)
// exported to ONNX that take numeric feature vectors and output either class logits
// or regression values.
type TabularPipeline struct {
	*backends.BasePipeline
	AggregationFunctionName string         // for classification: SOFTMAX or SIGMOID
	ProblemType             string         // "classification" or "regression"
	IDLabelMap              map[int]string // optional mapping from class IDs to labels
}

type TabularClassificationOutput struct {
	PredictedClass string
	Probabilities  []ClassificationOutput
}

// TabularOutput returns per-input results.
// - For classification: []TabularClassificationOutput
// - For regression: float32
// for each input.
type TabularOutput struct {
	Results []any
}

func (o *TabularOutput) GetOutput() []any { return o.Results }

// Options

func WithRegression() backends.PipelineOption[*TabularPipeline] {
	return func(p *TabularPipeline) error {
		p.ProblemType = "regression"
		return nil
	}
}

func WithClassification() backends.PipelineOption[*TabularPipeline] {
	return func(p *TabularPipeline) error {
		p.ProblemType = "classification"
		return nil
	}
}

func WithTabularSoftmax() backends.PipelineOption[*TabularPipeline] {
	return func(p *TabularPipeline) error {
		p.AggregationFunctionName = "SOFTMAX"
		return nil
	}
}

func WithTabularSigmoid() backends.PipelineOption[*TabularPipeline] {
	return func(p *TabularPipeline) error {
		p.AggregationFunctionName = "SIGMOID"
		return nil
	}
}

func WithIDLabelMap(labels map[int]string) backends.PipelineOption[*TabularPipeline] {
	return func(p *TabularPipeline) error {
		p.IDLabelMap = labels
		return nil
	}
}

// NewTabularPipeline initializes the pipeline.
func NewTabularPipeline(config backends.PipelineConfig[*TabularPipeline], s *options.Options, model *backends.Model) (*TabularPipeline, error) {
	base, err := backends.NewBasePipeline(config, s, model)
	if err != nil {
		return nil, err
	}
	p := &TabularPipeline{BasePipeline: base}
	for _, o := range config.Options {
		if err = o(p); err != nil {
			return nil, err
		}
	}
	if p.ProblemType == "" {
		p.ProblemType = "classification"
	}
	if err = p.Validate(); err != nil {
		return nil, err
	}

	if p.IDLabelMap == nil && p.ProblemType == "classification" {
		// build default IDLabelMap
		p.IDLabelMap = make(map[int]string)
		var numClasses int64
		if len(p.Model.OutputsMeta) == 1 {
			numClasses = p.Model.OutputsMeta[0].Dimensions[1]
		} else if len(p.Model.OutputsMeta) == 2 {
			numClasses = p.Model.OutputsMeta[1].Dimensions[1]
		}
		for i := int64(0); i < numClasses; i++ {
			p.IDLabelMap[int(i)] = fmt.Sprintf("class_%d", i)
		}
	}
	return p, nil
}

// Interface implementation

func (p *TabularPipeline) IsGenerative() bool        { return false }
func (p *TabularPipeline) GetModel() *backends.Model { return p.Model }

func (p *TabularPipeline) GetMetadata() backends.PipelineMetadata {
	return backends.PipelineMetadata{
		OutputsInfo: []backends.OutputInfo{{
			Name:       p.Model.OutputsMeta[0].Name,
			Dimensions: p.Model.OutputsMeta[0].Dimensions,
		}},
	}
}

func (p *TabularPipeline) GetStatistics() backends.PipelineStatistics {
	stats := backends.PipelineStatistics{}
	stats.ComputeOnnxStatistics(p.PipelineTimings)
	return stats
}

func (p *TabularPipeline) Validate() error {
	var errs []error
	if len(p.Model.InputsMeta) < 1 {
		errs = append(errs, fmt.Errorf("model must have at least one input"))
	} else {
		dims := p.Model.InputsMeta[0].Dimensions
		if len(dims) != 2 {
			errs = append(errs, fmt.Errorf("tabular pipeline expects 2D input (batch, features); got %d dims", len(dims)))
		}
	}
	// on the outputs we are now strict:
	// if it's a classification model we expect either:
	// - one output with shape (batch)
	// - or two outputs (class labels and probabilities) with shape (batch, 1) and (batch, num_classes)
	// if it's a regression model we expect one output with shape (batch, 1)
	if p.ProblemType == "classification" {
		if len(p.Model.OutputsMeta) == 1 {
			dims := p.Model.OutputsMeta[0].Dimensions
			if len(dims) != 2 {
				errs = append(errs, fmt.Errorf("classification model with one output must have 2D output (batch, num_classes); got %d dims", len(dims)))
			}
			if p.IDLabelMap != nil {
				numClasses := dims[1]
				if int64(len(p.IDLabelMap)) != numClasses {
					errs = append(errs, fmt.Errorf("IDLabelMap has %d entries but model output has %d classes", len(p.IDLabelMap), numClasses))
				}
			}
		} else if len(p.Model.OutputsMeta) == 2 {
			dims0 := p.Model.OutputsMeta[0].Dimensions
			dims1 := p.Model.OutputsMeta[1].Dimensions
			if len(dims0) != 1 || len(dims1) != 2 {
				errs = append(errs, fmt.Errorf("classification model with two outputs must have 2D outputs; got %d and %d dims", len(dims0), len(dims1)))
			}
			if p.IDLabelMap != nil {
				numClasses := dims1[1]
				if int64(len(p.IDLabelMap)) != numClasses {
					errs = append(errs, fmt.Errorf("IDLabelMap has %d entries but model output has %d classes", len(p.IDLabelMap), numClasses))
				}
			}
		} else {
			errs = append(errs, fmt.Errorf("classification model must have one or two outputs; got %d", len(p.Model.OutputsMeta)))
		}
	}

	if p.ProblemType == "regression" {
		if len(p.Model.OutputsMeta) != 1 {
			errs = append(errs, fmt.Errorf("regression model must have one output; got %d", len(p.Model.OutputsMeta)))
		} else {
			dims := p.Model.OutputsMeta[0].Dimensions
			if len(dims) != 2 || dims[1] != 1 {
				errs = append(errs, fmt.Errorf("regression model output must have shape (batch, 1); got dims %v", dims))
			}
		}
		if p.AggregationFunctionName != "" {
			errs = append(errs, fmt.Errorf("regression model cannot have aggregation function; got %s", p.AggregationFunctionName))
		}
	}
	return errors.Join(errs...)
}

// Preprocess parses inputs strings into [][]float32 and builds tensors.
func (p *TabularPipeline) Preprocess(batch *backends.PipelineBatch, inputs [][]float32) error {
	start := time.Now()
	// Build tensors
	if err := backends.CreateTabularTensors(batch, p.Model, inputs, p.Runtime); err != nil {
		return err
	}
	_ = start // measured but not recorded; tokenizer not used
	return nil
}

func (p *TabularPipeline) Forward(batch *backends.PipelineBatch) error {
	start := time.Now()
	if err := backends.RunSessionOnBatch(batch, p.BasePipeline); err != nil {
		return err
	}
	atomic.AddUint64(&p.PipelineTimings.NumCalls, 1)
	atomic.AddUint64(&p.PipelineTimings.TotalNS, safeconv.DurationToU64(time.Since(start)))
	return nil
}

func (p *TabularPipeline) Postprocess(batch *backends.PipelineBatch) (*TabularOutput, error) {
	results := make([]any, batch.Size)

	if p.ProblemType == "classification" {
		var agg func([]float32) []float32
		switch p.AggregationFunctionName {
		case "SOFTMAX":
			agg = vectorutil.SoftMax
		case "SIGMOID":
			agg = vectorutil.Sigmoid
		}

		switch len(p.Model.OutputsMeta) {
		case 2:
			// two outputs: class labels and probabilities
			outClassesAny := batch.OutputValues[0]
			outLogitsAny := batch.OutputValues[1]

			outClasses, ok := outClassesAny.([]int64)
			if !ok {
				return nil, fmt.Errorf("expected [][]any class labels, got %T", outClassesAny)
			}
			outLogits, ok := outLogitsAny.([][]float32)
			if !ok {
				return nil, fmt.Errorf("expected [][]float32 logits, got %T", outLogitsAny)
			}
			for i := 0; i < batch.Size; i++ {
				predictedClassLabel := outClasses[i]
				predictedLogits := outLogits[i]
				if agg != nil {
					predictedLogits = agg(predictedLogits)
				}
				classOutputs := make([]ClassificationOutput, len(predictedLogits))
				for j := range predictedLogits {
					classOutputs[j] = ClassificationOutput{
						Label: p.IDLabelMap[j],
						Score: predictedLogits[j],
					}
				}
				output := TabularClassificationOutput{
					PredictedClass: p.IDLabelMap[int(predictedClassLabel)],
					Probabilities:  classOutputs,
				}
				results[i] = output
			}
		case 1:
			// one output: logits
			outLogitsAny, ok := batch.OutputValues[0].([][]float32)
			if !ok {
				return nil, fmt.Errorf("expected [][]float32 logits, got %T", outLogitsAny)
			}
			for i := 0; i < batch.Size; i++ {
				predictedLogits := outLogitsAny[i]
				if agg != nil {
					predictedLogits = agg(predictedLogits)
				}
				classOutputs := make([]ClassificationOutput, len(predictedLogits))
				for j := range predictedLogits {
					classOutputs[j] = ClassificationOutput{
						Label: p.IDLabelMap[j],
						Score: predictedLogits[j],
					}
				}
				// find predicted class
				maxIndex, _, err := vectorutil.ArgMax(predictedLogits)
				if err != nil {
					return nil, err
				}
				output := TabularClassificationOutput{
					PredictedClass: p.IDLabelMap[maxIndex],
					Probabilities:  classOutputs,
				}
				results[i] = output
			}
		default:
			return nil, fmt.Errorf("unsupported number of outputs %d for classification", len(p.Model.OutputsMeta))
		}
		return &TabularOutput{Results: results}, nil
	}

	switch v := batch.OutputValues[0].(type) {
	case []float32:
		for i := range v {
			results[i] = v[i]
		}
	default:
		return nil, fmt.Errorf("unsupported regression output type %T", batch.OutputValues[0])
	}
	return &TabularOutput{Results: results}, nil
}

// Run executes the pipeline over inputs.
func (p *TabularPipeline) Run(inputs []string) (backends.PipelineBatchOutput, error) {
	features, err := parseFeatures(inputs)
	if err != nil {
		return nil, err
	}
	return p.RunPipeline(features)
}

func (p *TabularPipeline) RunPipeline(inputs [][]float32) (*TabularOutput, error) {
	var runErrors []error
	batch := backends.NewBatch(len(inputs))
	defer func(*backends.PipelineBatch) {
		runErrors = append(runErrors, batch.Destroy())
	}(batch)

	runErrors = append(runErrors, p.Preprocess(batch, inputs))
	if e := errors.Join(runErrors...); e != nil {
		return nil, e
	}

	runErrors = append(runErrors, p.Forward(batch))
	if e := errors.Join(runErrors...); e != nil {
		return nil, e
	}

	result, postErr := p.Postprocess(batch)
	runErrors = append(runErrors, postErr)
	return result, errors.Join(runErrors...)
}

// parseFeatures accepts each input only as a JSON array ("[1,2,3]").
func parseFeatures(inputs []string) ([][]float32, error) {
	out := make([][]float32, len(inputs))
	for i, s := range inputs {
		sTrim := strings.TrimSpace(s)
		if !strings.HasPrefix(sTrim, "[") {
			return nil, fmt.Errorf("input %d: expected JSON array like \"[1,2,3]\"", i)
		}
		var arr []float64
		if err := json.Unmarshal([]byte(sTrim), &arr); err != nil {
			return nil, fmt.Errorf("input %d: invalid JSON array: %w", i, err)
		}
		vec := make([]float32, len(arr))
		for j := range arr {
			vec[j] = float32(arr[j])
		}
		out[i] = vec
	}
	return out, nil
}
