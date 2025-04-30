package pipelines

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strings"
	"sync/atomic"
	"time"

	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/pipelineBackends"
	"github.com/knights-analytics/hugot/util"

	jsoniter "github.com/json-iterator/go"
)

type ZeroShotClassificationPipeline struct {
	*pipelineBackends.BasePipeline
	IDLabelMap         map[int]string
	Sequences          []string
	Labels             []string
	HypothesisTemplate string
	Multilabel         bool
	entailmentID       int
	separatorToken     string
}

type ZeroShotClassificationPipelineConfig struct {
	IDLabelMap map[int]string `json:"id2label"`
}

type ZeroShotClassificationOutput struct {
	Sequence     string
	SortedValues []struct {
		Key   string
		Value float64
	}
}

type ZeroShotOutput struct {
	ClassificationOutputs []ZeroShotClassificationOutput
}

// options

// WithMultilabel can be used to set whether the pipeline is multilabel.
func WithMultilabel(multilabel bool) pipelineBackends.PipelineOption[*ZeroShotClassificationPipeline] {
	return func(pipeline *ZeroShotClassificationPipeline) {
		pipeline.Multilabel = multilabel
	}
}

// WithLabels can be used to set the labels to classify the examples.
func WithLabels(labels []string) pipelineBackends.PipelineOption[*ZeroShotClassificationPipeline] {
	return func(pipeline *ZeroShotClassificationPipeline) {
		pipeline.Labels = labels
	}
}

// WithHypothesisTemplate can be used to set the hypothesis template for classification.
func WithHypothesisTemplate(hypothesisTemplate string) pipelineBackends.PipelineOption[*ZeroShotClassificationPipeline] {
	return func(pipeline *ZeroShotClassificationPipeline) {
		pipeline.HypothesisTemplate = hypothesisTemplate
	}
}

// GetOutput converts raw output to readable output.
func (t *ZeroShotOutput) GetOutput() []any {
	out := make([]any, len(t.ClassificationOutputs))
	for i, o := range t.ClassificationOutputs {
		out[i] = any(o)
	}
	return out
}

// create all pairs between input sequences and labels.
func createSequencePairs(sequences interface{}, labels []string, hypothesisTemplate string) ([][][]string, []string, error) {
	// Check if labels or sequences are empty
	if len(labels) == 0 || sequences == nil {
		return nil, nil, errors.New("you must include at least one label and at least one sequence")
	}

	// Check if hypothesisTemplate can be formatted with labels
	if fmt.Sprintf(hypothesisTemplate, labels[0]) == hypothesisTemplate {
		return nil, nil, fmt.Errorf(`the provided hypothesis_template "%s" was not able to be formatted with the target labels. Make sure the passed template includes formatting syntax such as {{}} where the label should go`, hypothesisTemplate)
	}

	// Convert sequences to []string if it's a single string
	var seqs []string
	switch v := sequences.(type) {
	case string:
		seqs = []string{v}
	case []string:
		seqs = v
	default:
		return nil, nil, errors.New("sequences must be either a string or a []string")
	}

	// Create sequence_pairs
	var sequencePairs [][][]string
	for _, sequence := range seqs {
		var temp [][]string
		for _, label := range labels {
			hypothesis := strings.Replace(hypothesisTemplate, "{}", label, 1)
			temp = append(temp, []string{sequence, hypothesis})
		}
		sequencePairs = append(sequencePairs, temp)
	}
	return sequencePairs, seqs, nil
}

// NewZeroShotClassificationPipeline create new Zero Shot Classification Pipeline.
func NewZeroShotClassificationPipeline(config pipelineBackends.PipelineConfig[*ZeroShotClassificationPipeline], s *options.Options, model *pipelineBackends.Model) (*ZeroShotClassificationPipeline, error) {

	defaultPipeline, err := pipelineBackends.NewBasePipeline(config, s, model)
	if err != nil {
		return nil, err
	}

	pipeline := &ZeroShotClassificationPipeline{BasePipeline: defaultPipeline}
	for _, o := range config.Options {
		o(pipeline)
	}
	pipeline.entailmentID = -1 // Default value
	pipeline.HypothesisTemplate = "This example is {}."

	if len(pipeline.Labels) == 0 {
		return nil, fmt.Errorf("no labels provided, please provide labels using the WithLabels() option")
	}

	// read id to label map
	configPath := util.PathJoinSafe(model.Path, "config.json")
	pipelineInputConfig := ZeroShotClassificationPipelineConfig{}
	mapBytes, err := util.ReadFileBytes(configPath)
	if err != nil {
		return nil, err
	}
	err = jsoniter.Unmarshal(mapBytes, &pipelineInputConfig)
	if err != nil {
		return nil, err
	}

	// Set IDLabelMap
	pipeline.IDLabelMap = pipelineInputConfig.IDLabelMap

	// Find entailment ID
	for id, label := range pipeline.IDLabelMap {
		if strings.HasPrefix(strings.ToLower(label), "entail") {
			pipeline.entailmentID = id
			break
		}
	}

	configPath1 := util.PathJoinSafe(model.Path, "special_tokens_map.json")
	file, err := os.Open(configPath1)
	if err != nil {
		return nil, fmt.Errorf("cannot read special_tokens_map.json at %s", model.Path)
	}
	defer func() {
		err = file.Close()
	}()

	byteValue, _ := io.ReadAll(file)
	var result map[string]interface{}
	err = json.Unmarshal(byteValue, &result)
	if err != nil {
		return nil, fmt.Errorf("cannot unmarshal special_tokens_map.json at %s", model.Path)
	}

	sepToken, ok := result["sep_token"]
	if !ok {
		return nil, fmt.Errorf("no sep token detected in special_tokens_map.json at %s", model.Path)
	}

	switch v := sepToken.(type) {
	case map[string]interface{}:
		t, ok := v["content"]
		if !ok {
			return nil, fmt.Errorf("sep_token is map but no content field is available")
		}
		tString, ok := t.(string)
		if !ok {
			return nil, fmt.Errorf("sep_token cannot be converted to string: %v", t)
		}
		pipeline.separatorToken = tString
	case string:
		pipeline.separatorToken = v
	default:
		return nil, fmt.Errorf("sep_token has unexpected type: %v", v)
	}

	return pipeline, err
}

func (p *ZeroShotClassificationPipeline) Preprocess(batch *pipelineBackends.PipelineBatch, inputs []string) error {
	start := time.Now()
	pipelineBackends.TokenizeInputs(batch, p.Model.Tokenizer, inputs)
	atomic.AddUint64(&p.Model.Tokenizer.TokenizerTimings.NumCalls, 1)
	atomic.AddUint64(&p.Model.Tokenizer.TokenizerTimings.TotalNS, uint64(time.Since(start)))
	err := pipelineBackends.CreateInputTensors(batch, p.Model.InputsMeta, p.Runtime)
	return err
}

func (p *ZeroShotClassificationPipeline) Forward(batch *pipelineBackends.PipelineBatch) error {
	start := time.Now()
	err := pipelineBackends.RunSessionOnBatch(batch, p.BasePipeline)
	if err != nil {
		return err
	}
	atomic.AddUint64(&p.PipelineTimings.NumCalls, 1)
	atomic.AddUint64(&p.PipelineTimings.TotalNS, uint64(time.Since(start)))
	return nil
}

func (p *ZeroShotClassificationPipeline) Postprocess(outputTensors [][][]float32, labels []string, sequences []string) (*ZeroShotOutput, error) {
	classificationOutputs := make([]ZeroShotClassificationOutput, 0, len(sequences))

	LabelLikelihood := make(map[string]float64)
	if p.Multilabel || len(p.Labels) == 1 {
		for ind, sequence := range outputTensors {
			output := ZeroShotClassificationOutput{
				Sequence: sequences[ind],
			}

			var entailmentLogits []float32
			var contradictionLogits []float32

			var entailmentID int
			var contradictionID int
			switch p.entailmentID {
			case -1:
				entailmentID = len(sequence[0]) - 1
				contradictionID = 0
			default:
				entailmentID = p.entailmentID
				contradictionID = 0
				if entailmentID == 0 {
					contradictionID = len(sequence[0]) - 1
				}
			}

			for _, tensor := range sequence {
				entailmentLogits = append(entailmentLogits, tensor[entailmentID])
				contradictionLogits = append(contradictionLogits, tensor[contradictionID])
			}

			for i := range entailmentLogits {
				logits := []float64{float64(contradictionLogits[i]), float64(entailmentLogits[i])}
				expLogits := []float64{math.Exp(logits[0]), math.Exp(logits[1])}
				sumExpLogits := expLogits[0] + expLogits[1]
				score := expLogits[1] / sumExpLogits
				LabelLikelihood[labels[i]] = score
			}

			// Define ss as a slice of anonymous structs
			var ss []struct {
				Key   string
				Value float64
			}
			for k, v := range LabelLikelihood {
				ss = append(ss, struct {
					Key   string
					Value float64
				}{k, v})
			}

			// Sort the slice by the value field
			sort.Slice(ss, func(i, j int) bool {
				return ss[i].Value > ss[j].Value
			})

			output.SortedValues = ss
			classificationOutputs = append(classificationOutputs, output)
		}
		return &ZeroShotOutput{
			ClassificationOutputs: classificationOutputs,
		}, nil
	}

	for ind, sequence := range outputTensors {
		output := ZeroShotClassificationOutput{}

		var entailmentLogits []float32
		var entailmentID int
		switch p.entailmentID {
		case -1:
			entailmentID = len(sequence[0]) - 1
		default:
			entailmentID = p.entailmentID
		}
		for _, tensor := range sequence {
			entailmentLogits = append(entailmentLogits, tensor[entailmentID])
		}

		var numerator []float64
		var logitSum float64
		for _, logit := range entailmentLogits {
			exp := math.Exp(float64(logit))
			numerator = append(numerator, exp)
			logitSum += exp
		}

		var quotient []float64

		for ind, i := range numerator {
			quotient = append(quotient, i/logitSum)
			LabelLikelihood[labels[ind]] = quotient[ind]
		}

		output.Sequence = sequences[ind]

		// Define ss as a slice of anonymous structs
		var ss []struct {
			Key   string
			Value float64
		}
		for k, v := range LabelLikelihood {
			ss = append(ss, struct {
				Key   string
				Value float64
			}{k, v})
		}

		// Sort the slice by the value field
		sort.Slice(ss, func(i, j int) bool {
			return ss[i].Value > ss[j].Value
		})

		output.SortedValues = ss
		classificationOutputs = append(classificationOutputs, output)
	}
	return &ZeroShotOutput{
		ClassificationOutputs: classificationOutputs,
	}, nil
}

func (p *ZeroShotClassificationPipeline) RunPipeline(inputs []string) (*ZeroShotOutput, error) {
	var outputTensors [][][]float32
	var runErrors []error

	sequencePairs, _, err := createSequencePairs(inputs, p.Labels, p.HypothesisTemplate)
	if err != nil {
		return nil, err
	}

	for _, sequence := range sequencePairs {
		var sequenceTensors [][]float32
		for _, pair := range sequence {

			batch := pipelineBackends.NewBatch()

			// have to do this because python inserts a separator token in between the two clauses when tokenizing
			// separator token isn't universal and depends on its value in special_tokens_map.json of model
			// still isn't perfect because some models (protectai/MoritzLaurer-roberta-base-zeroshot-v2.0-c-onnx for example)
			// insert two separator tokens while others (protectai/deberta-v3-base-zeroshot-v1-onnx and others) only insert one
			// need to find a way to determine how many to insert or find a better way to tokenize inputs
			// The difference in outputs for one separator vs two is very small (differences in the thousandths place), but they
			// definitely are different
			concatenatedString := pair[0] + p.separatorToken + pair[1]
			runErrors = append(runErrors, p.Preprocess(batch, []string{concatenatedString}))
			if e := errors.Join(runErrors...); e != nil {
				return nil, errors.Join(e, batch.Destroy())
			}
			runErrors = append(runErrors, p.Forward(batch))
			if e := errors.Join(runErrors...); e != nil {
				return nil, errors.Join(e, batch.Destroy())
			}
			sequenceTensors = append(sequenceTensors, batch.OutputValues[0].Result2D[0])

			runErrors = append(runErrors, batch.Destroy())
			if e := errors.Join(runErrors...); e != nil {
				return nil, e
			}
		}
		outputTensors = append(outputTensors, sequenceTensors)
	}

	outputs, err := p.Postprocess(outputTensors, p.Labels, inputs)
	runErrors = append(runErrors, err)
	return outputs, errors.Join(runErrors...)
}

// PIPELINE INTERFACE IMPLEMENTATION

func (p *ZeroShotClassificationPipeline) GetModel() *pipelineBackends.Model {
	return p.BasePipeline.Model
}

func (p *ZeroShotClassificationPipeline) GetMetadata() pipelineBackends.PipelineMetadata {
	return pipelineBackends.PipelineMetadata{
		OutputsInfo: []pipelineBackends.OutputInfo{
			{
				Name:       p.Model.OutputsMeta[0].Name,
				Dimensions: p.Model.OutputsMeta[0].Dimensions,
			},
		},
	}
}

func (p *ZeroShotClassificationPipeline) GetStats() []string {
	return []string{
		fmt.Sprintf("Statistics for pipeline: %s", p.PipelineName),
		fmt.Sprintf("Tokenizer: Total time=%s, Execution count=%d, Average query time=%s",
			time.Duration(p.Model.Tokenizer.TokenizerTimings.TotalNS),
			p.Model.Tokenizer.TokenizerTimings.NumCalls,
			time.Duration(float64(p.Model.Tokenizer.TokenizerTimings.TotalNS)/math.Max(1, float64(p.Model.Tokenizer.TokenizerTimings.NumCalls)))),
		fmt.Sprintf("ONNX: Total time=%s, Execution count=%d, Average query time=%s",
			time.Duration(p.PipelineTimings.TotalNS),
			p.PipelineTimings.NumCalls,
			time.Duration(float64(p.PipelineTimings.TotalNS)/math.Max(1, float64(p.PipelineTimings.NumCalls)))),
	}
}

func (p *ZeroShotClassificationPipeline) Run(inputs []string) (pipelineBackends.PipelineBatchOutput, error) {
	return p.RunPipeline(inputs)
}

func (p *ZeroShotClassificationPipeline) Validate() error {
	var validationErrors []error

	if len(p.IDLabelMap) <= 0 {
		validationErrors = append(validationErrors, fmt.Errorf("pipeline configuration invalid: length of id2label map for token classification pipeline must be greater than zero"))
	}

	outDims := p.Model.OutputsMeta[0].Dimensions
	if len(outDims) != 2 {
		validationErrors = append(validationErrors, fmt.Errorf("pipeline configuration invalid: zero shot classification must have 2 dimensional output"))
	}

	dynamicBatch := false
	for _, d := range outDims {
		if d == -1 {
			if dynamicBatch {
				validationErrors = append(validationErrors, fmt.Errorf("pipeline configuration invalid: text classification must have max one dynamic dimensions (input)"))
				break
			}
			dynamicBatch = true
		}
	}
	return errors.Join(validationErrors...)
}
