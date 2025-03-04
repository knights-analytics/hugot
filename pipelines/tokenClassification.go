package pipelines

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"
	"sync/atomic"
	"time"

	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/pipelineBackends"
	"github.com/knights-analytics/hugot/util"

	jsoniter "github.com/json-iterator/go"
)

// TokenClassificationPipeline is a go version of huggingface tokenClassificationPipeline.
// https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/token_classification.py
type TokenClassificationPipeline struct {
	*pipelineBackends.BasePipeline
	IDLabelMap          map[int]string
	AggregationStrategy string
	IgnoreLabels        []string
}

type TokenClassificationPipelineConfig struct {
	IDLabelMap map[int]string `json:"id2label"`
}

type Entity struct {
	Entity    string
	Score     float32
	Scores    []float32
	Index     int
	Word      string
	TokenID   []uint32
	Start     uint
	End       uint
	IsSubword bool
}

type TokenClassificationOutput struct {
	Entities [][]Entity
}

func (t *TokenClassificationOutput) GetOutput() []any {
	out := make([]any, len(t.Entities))
	for i, entity := range t.Entities {
		out[i] = any(entity)
	}
	return out
}

// options

// TODO: need to implement the other types of aggregation (max etc)

// WithSimpleAggregation sets the aggregation strategy for the token labels to simple
// It reproduces simple aggregation from the huggingface implementation.
func WithSimpleAggregation() pipelineBackends.PipelineOption[*TokenClassificationPipeline] {
	return func(pipeline *TokenClassificationPipeline) {
		pipeline.AggregationStrategy = "SIMPLE"
	}
}

func WithAverageAggregation() pipelineBackends.PipelineOption[*TokenClassificationPipeline] {
	return func(pipeline *TokenClassificationPipeline) {
		pipeline.AggregationStrategy = "AVERAGE"
	}
}

// WithoutAggregation returns the token labels.
func WithoutAggregation() pipelineBackends.PipelineOption[*TokenClassificationPipeline] {
	return func(pipeline *TokenClassificationPipeline) {
		pipeline.AggregationStrategy = "NONE"
	}
}

func WithIgnoreLabels(ignoreLabels []string) pipelineBackends.PipelineOption[*TokenClassificationPipeline] {
	return func(pipeline *TokenClassificationPipeline) {
		pipeline.IgnoreLabels = ignoreLabels
	}
}

// NewTokenClassificationPipeline Initializes a feature extraction pipeline.
func NewTokenClassificationPipeline(config pipelineBackends.PipelineConfig[*TokenClassificationPipeline], s *options.Options, model *pipelineBackends.Model) (*TokenClassificationPipeline, error) {

	defaultPipeline, err := pipelineBackends.NewBasePipeline(config, s, model)
	if err != nil {
		return nil, err
	}

	pipeline := &TokenClassificationPipeline{BasePipeline: defaultPipeline}
	for _, o := range config.Options {
		o(pipeline)
	}

	// Id label map
	configPath := util.PathJoinSafe(config.ModelPath, "config.json")
	pipelineInputConfig := TokenClassificationPipelineConfig{}
	mapBytes, err := util.ReadFileBytes(configPath)
	if err != nil {
		return nil, err
	}

	err = jsoniter.Unmarshal(mapBytes, &pipelineInputConfig)
	if err != nil {
		return nil, err
	}
	pipeline.IDLabelMap = pipelineInputConfig.IDLabelMap

	// default strategies if not set
	if pipeline.AggregationStrategy == "" {
		pipeline.AggregationStrategy = "SIMPLE"
	}
	if len(pipeline.IgnoreLabels) == 0 {
		pipeline.IgnoreLabels = []string{"O"}
	}

	// Additional options needed for postprocessing
	pipelineBackends.AllInputTokens(pipeline.BasePipeline)

	err = pipeline.Validate()
	if err != nil {
		return nil, err
	}
	return pipeline, nil
}

// INTERFACE IMPLEMENTATION

func (p *TokenClassificationPipeline) GetModel() *pipelineBackends.Model {
	return p.BasePipeline.Model
}

// GetMetadata returns metadata information about the pipeline, in particular:
// OutputInfo: names and dimensions of the output layer used for token classification.
func (p *TokenClassificationPipeline) GetMetadata() pipelineBackends.PipelineMetadata {
	return pipelineBackends.PipelineMetadata{
		OutputsInfo: []pipelineBackends.OutputInfo{
			{
				Name:       p.Model.OutputsMeta[0].Name,
				Dimensions: p.Model.OutputsMeta[0].Dimensions,
			},
		},
	}
}

// GetStats returns the runtime statistics for the pipeline.
func (p *TokenClassificationPipeline) GetStats() []string {
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

// Validate checks that the pipeline is valid.
func (p *TokenClassificationPipeline) Validate() error {
	var validationErrors []error

	outputDim := p.Model.OutputsMeta[0].Dimensions
	if len(outputDim) != 3 {
		validationErrors = append(validationErrors,
			fmt.Errorf("output for token classification must be three dimensional (input, sequence, logits)"))
	}

	if outputDim[len(outputDim)-1] == -1 {
		validationErrors = append(validationErrors,
			fmt.Errorf("logit dimension cannot be dynamic"))
	}
	if len(p.IDLabelMap) <= 0 {
		validationErrors = append(validationErrors, fmt.Errorf("p configuration invalid: length of id2label map for token classification p must be greater than zero"))
	}
	return errors.Join(validationErrors...)
}

// Preprocess tokenizes the input strings.
func (p *TokenClassificationPipeline) Preprocess(batch *pipelineBackends.PipelineBatch, inputs []string) error {
	start := time.Now()
	pipelineBackends.TokenizeInputs(batch, p.Model.Tokenizer, inputs)
	atomic.AddUint64(&p.Model.Tokenizer.TokenizerTimings.NumCalls, 1)
	atomic.AddUint64(&p.Model.Tokenizer.TokenizerTimings.TotalNS, uint64(time.Since(start)))
	err := pipelineBackends.CreateInputTensors(batch, p.Model.InputsMeta, p.Runtime)
	return err
}

// Forward performs the forward inference of the pipeline.
func (p *TokenClassificationPipeline) Forward(batch *pipelineBackends.PipelineBatch) error {
	start := time.Now()
	err := pipelineBackends.RunSessionOnBatch(batch, p.BasePipeline)
	if err != nil {
		return err
	}
	atomic.AddUint64(&p.PipelineTimings.NumCalls, 1)
	atomic.AddUint64(&p.PipelineTimings.TotalNS, uint64(time.Since(start)))
	return nil
}

// Postprocess function for a token classification pipeline.
func (p *TokenClassificationPipeline) Postprocess(batch *pipelineBackends.PipelineBatch) (*TokenClassificationOutput, error) {
	if len(batch.Input) == 0 {
		return &TokenClassificationOutput{}, nil
	}

	output := batch.OutputValues[0]

	if len(output.Result3D) == 0 {
		return nil, fmt.Errorf("3D output has empty result")
	}

	for batchIndex, tokens := range output.Result3D {
		output.Result3D[batchIndex] = make([][]float32, len(tokens))
		for tokenIndex, tokenLogits := range tokens {
			output.Result3D[batchIndex][tokenIndex] = util.SoftMax(tokenLogits)
		}
	}

	// now convert the logits to the predictions of actual entities
	classificationOutput := TokenClassificationOutput{
		Entities: make([][]Entity, len(batch.Input)),
	}

	for i, input := range batch.Input {
		preEntities := p.GatherPreEntities(input, output.Result3D[i])
		entities, errAggregate := p.Aggregate(input, preEntities)
		if errAggregate != nil {
			return nil, errAggregate
		}
		// Filter anything that is in ignore_labels
		var filteredEntities []Entity
		for _, e := range entities {
			if !slices.Contains(p.IgnoreLabels, e.Entity) && e.Entity != "" {
				filteredEntities = append(filteredEntities, e)
			}
		}
		classificationOutput.Entities[i] = filteredEntities
	}
	return &classificationOutput, nil
}

// GatherPreEntities from batch of logits to list of pre-aggregated outputs.
func (p *TokenClassificationPipeline) GatherPreEntities(input pipelineBackends.TokenizedInput, output [][]float32) []Entity {
	sentence := input.Raw
	var preEntities []Entity

	for j, tokenScores := range output {

		// filter out special tokens (skip them)
		if input.SpecialTokensMask[j] > 0.0 {
			continue
		}
		// TODO: the python code uses id_to_token to get the token here which is a method on the rust tokenizer, check if it's better
		word := input.Tokens[j]
		tokenID := input.TokenIDs[j]
		// TODO: the determination of subword can probably be better done by exporting the words field from the tokenizer directly
		startInd := input.Offsets[j][0]
		endInd := input.Offsets[j][1]
		wordRef := sentence[startInd:endInd]
		isSubword := len(word) != len(wordRef)
		// TODO: check for unknown token here, it's in the config and can be loaded and compared with the token
		// in that case set the subword as in the python code
		preEntities = append(preEntities, Entity{
			Word:      word,
			TokenID:   []uint32{tokenID},
			Scores:    tokenScores,
			Start:     startInd,
			End:       endInd,
			Index:     j,
			IsSubword: isSubword,
		})
	}
	return preEntities
}

func (p *TokenClassificationPipeline) aggregateWord(entities []Entity) (Entity, error) {
	tokens := make([]uint32, len(entities))
	for i, e := range entities {
		tokens[i] = e.TokenID[0]
	}
	var newEntity Entity

	word, err := pipelineBackends.Decode(tokens, p.Model.Tokenizer, true)
	if err != nil {
		return newEntity, err
	}

	switch p.AggregationStrategy {
	case "AVERAGE":
		scores := make([][]float32, len(p.IDLabelMap))
		for _, e := range entities {
			for i, score := range e.Scores {
				scores[i] = append(scores[i], score)
			}
		}
		averages := make([]float32, len(p.IDLabelMap))
		for i, s := range scores {
			averages[i] = util.Mean(s)
		}
		entityIdx, score, _ := util.ArgMax(averages)
		label, ok := p.IDLabelMap[entityIdx]
		if !ok {
			return Entity{}, fmt.Errorf("could not determine entity type for input %s, predicted entity index %d", word, entityIdx)
		}
		newEntity = Entity{
			Entity:  label,
			Score:   score,
			Word:    word,
			TokenID: tokens,
			Start:   entities[0].Start,
			End:     entities[len(entities)-1].End,
		}
	default:
		return Entity{}, fmt.Errorf("aggregation strategy %s not recognized", p.AggregationStrategy)
	}
	return newEntity, nil
}

func (p *TokenClassificationPipeline) aggregateWords(entities []Entity) ([]Entity, error) {
	var wordGroup []Entity
	var wordEntities []Entity

	for _, entity := range entities {
		if len(wordGroup) == 0 {
			wordGroup = []Entity{entity}
		} else if entity.IsSubword {
			wordGroup = append(wordGroup, entity)
		} else {
			aggregated, err := p.aggregateWord(wordGroup)
			if err != nil {
				return nil, err
			}
			wordEntities = append(wordEntities, aggregated)
			wordGroup = []Entity{entity}
		}
	}

	if len(wordGroup) > 0 {
		aggregated, err := p.aggregateWord(wordGroup)
		if err != nil {
			return nil, err
		}
		wordEntities = append(wordEntities, aggregated)
	}
	return wordEntities, nil
}

func (p *TokenClassificationPipeline) Aggregate(input pipelineBackends.TokenizedInput, preEntities []Entity) ([]Entity, error) {
	entities := make([]Entity, len(preEntities))
	var aggregationError error

	if p.AggregationStrategy == "SIMPLE" || p.AggregationStrategy == "NONE" {
		for i, preEntity := range preEntities {
			entityIdx, score, argMaxErr := util.ArgMax(preEntity.Scores)
			if argMaxErr != nil {
				return nil, argMaxErr
			}
			label, ok := p.IDLabelMap[entityIdx]
			if !ok {
				return nil, fmt.Errorf("could not determine entity type for input %s, predicted entity index %d", input.Raw, entityIdx)
			}
			entities[i] = Entity{
				Entity:  label,
				Score:   score,
				Index:   preEntity.Index,
				Word:    preEntity.Word,
				TokenID: preEntity.TokenID,
				Start:   preEntity.Start,
				End:     preEntity.End,
			}
		}
	} else {
		entities, aggregationError = p.aggregateWords(preEntities)
		if aggregationError != nil {
			return nil, aggregationError
		}
	}

	if p.AggregationStrategy == "NONE" {
		return entities, nil
	}
	return p.GroupEntities(entities)
}

func (p *TokenClassificationPipeline) getTag(entityName string) (string, string) {
	var bi string
	var tag string
	if strings.HasPrefix(entityName, "B-") {
		bi = "B"
		tag = entityName[2:]
	} else if strings.HasPrefix(entityName, "I-") {
		bi = "I"
		tag = entityName[2:]
	} else {
		// defaulting to "I" if string is not in B- I- format
		bi = "I"
		tag = entityName
	}
	return bi, tag
}

func (p *TokenClassificationPipeline) groupSubEntities(entities []Entity) (Entity, error) {
	splits := strings.Split(entities[0].Entity, "-")
	var entityType string
	if len(splits) == 1 {
		entityType = splits[0]
	} else {
		entityType = strings.Join(splits[1:], "-")
	}
	scores := make([]float32, len(entities))
	tokens := make([]uint32, len(entities))
	for i, s := range entities {
		scores[i] = s.Score
		tokens = slices.Concat(tokens, s.TokenID)
	}
	score := util.Mean(scores)
	// note: here we directly appeal to the tokenizer decoder with the tokenIds
	// in the python code they pass the words to a token_to_string_method
	word, err := pipelineBackends.Decode(tokens, p.Model.Tokenizer, true)
	if err != nil {
		return Entity{}, err
	}

	return Entity{
		Entity: entityType,
		Score:  score,
		Word:   word,
		Start:  entities[0].Start,
		End:    entities[len(entities)-1].End,
	}, nil
}

// GroupEntities group together adjacent tokens with the same entity predicted.
func (p *TokenClassificationPipeline) GroupEntities(entities []Entity) ([]Entity, error) {
	var entityGroups []Entity
	var currentGroupDisagg []Entity

	for _, e := range entities {
		if len(currentGroupDisagg) == 0 {
			currentGroupDisagg = append(currentGroupDisagg, e)
			continue
		}

		bi, tag := p.getTag(e.Entity)
		_, lastTag := p.getTag(currentGroupDisagg[len(currentGroupDisagg)-1].Entity)
		if tag == lastTag && bi != "B" {
			currentGroupDisagg = append(currentGroupDisagg, e)
		} else {
			// create the grouped entity
			groupedEntity, err := p.groupSubEntities(currentGroupDisagg)
			if err != nil {
				return nil, err
			}
			entityGroups = append(entityGroups, groupedEntity)
			currentGroupDisagg = []Entity{e}
		}
	}

	if len(currentGroupDisagg) > 0 {
		// last entity remaining
		groupedEntity, err := p.groupSubEntities(currentGroupDisagg)
		if err != nil {
			return nil, err
		}
		entityGroups = append(entityGroups, groupedEntity)
	}
	return entityGroups, nil
}

// Run the pipeline on a string batch.
func (p *TokenClassificationPipeline) Run(inputs []string) (pipelineBackends.PipelineBatchOutput, error) {
	return p.RunPipeline(inputs)
}

// RunPipeline is like Run but returns the concrete type rather than the interface.
func (p *TokenClassificationPipeline) RunPipeline(inputs []string) (*TokenClassificationOutput, error) {
	var runErrors []error
	batch := pipelineBackends.NewBatch()
	defer func(*pipelineBackends.PipelineBatch) {
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
