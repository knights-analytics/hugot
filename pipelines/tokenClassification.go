package pipelines

import (
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync/atomic"
	"time"

	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/util/safeconv"
	"github.com/knights-analytics/hugot/util/vectorutil"
)

// TokenClassificationPipeline is a go version of huggingface tokenClassificationPipeline.
// https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/token_classification.py
type TokenClassificationPipeline struct {
	*backends.BasePipeline
	IDLabelMap          map[int]string
	AggregationStrategy string
	IgnoreLabels        []string
	SplitWords          bool
}
type Entity struct {
	Entity    string
	Word      string
	Scores    []float32
	TokenID   []uint32
	Index     int
	Start     uint
	End       uint
	Score     float32
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

// WithSimpleAggregation sets the aggregation strategy for the token labels to simple
// It reproduces simple aggregation from the huggingface implementation.
func WithSimpleAggregation() backends.PipelineOption[*TokenClassificationPipeline] {
	return func(pipeline *TokenClassificationPipeline) error {
		pipeline.AggregationStrategy = "SIMPLE"
		return nil
	}
}

// WithAverageAggregation sets the aggregation strategy for the token labels to average
// It reproduces simple aggregation from the huggingface implementation.
func WithAverageAggregation() backends.PipelineOption[*TokenClassificationPipeline] {
	return func(pipeline *TokenClassificationPipeline) error {
		pipeline.AggregationStrategy = "AVERAGE"
		return nil
	}
}

// WithMaxAggregation sets the aggregation strategy for the token labels to Max
// It reproduces max aggregation from the huggingface implementation.
func WithMaxAggregation() backends.PipelineOption[*TokenClassificationPipeline] {
	return func(pipeline *TokenClassificationPipeline) error {
		pipeline.AggregationStrategy = "MAX"
		return nil
	}
}

// WithFirstAggregation sets the aggregation strategy for the token labels to first
// It reproduces first aggregation from the huggingface implementation.
func WithFirstAggregation() backends.PipelineOption[*TokenClassificationPipeline] {
	return func(pipeline *TokenClassificationPipeline) error {
		pipeline.AggregationStrategy = "FIRST"
		return nil
	}
}

// WithoutAggregation returns the token labels.
func WithoutAggregation() backends.PipelineOption[*TokenClassificationPipeline] {
	return func(pipeline *TokenClassificationPipeline) error {
		pipeline.AggregationStrategy = "NONE"
		return nil
	}
}

func WithIgnoreLabels(ignoreLabels []string) backends.PipelineOption[*TokenClassificationPipeline] {
	return func(pipeline *TokenClassificationPipeline) error {
		pipeline.IgnoreLabels = ignoreLabels
		return nil
	}
}

// WithSplitWords enables word-level alignment like Hugging Face's is_split_into_words.
func WithSplitWords() backends.PipelineOption[*TokenClassificationPipeline] {
	return func(pipeline *TokenClassificationPipeline) error {
		pipeline.SplitWords = true
		return nil
	}
}

// NewTokenClassificationPipeline Initializes a feature extraction pipeline.
func NewTokenClassificationPipeline(config backends.PipelineConfig[*TokenClassificationPipeline], s *options.Options, model *backends.Model) (*TokenClassificationPipeline, error) {
	defaultPipeline, err := backends.NewBasePipeline(config, s, model)
	if err != nil {
		return nil, err
	}
	pipeline := &TokenClassificationPipeline{BasePipeline: defaultPipeline}
	for _, o := range config.Options {
		err = o(pipeline)
		if err != nil {
			return nil, err
		}
	}
	// Id label map
	pipeline.IDLabelMap = model.IDLabelMap
	// default strategies if not set
	if pipeline.AggregationStrategy == "" {
		pipeline.AggregationStrategy = "SIMPLE"
	}
	if len(pipeline.IgnoreLabels) == 0 {
		pipeline.IgnoreLabels = []string{"O"}
	}
	// Additional options needed for postprocessing
	backends.AllInputTokens(pipeline.BasePipeline)
	err = pipeline.Validate()
	if err != nil {
		return nil, err
	}
	return pipeline, nil
}

// INTERFACE IMPLEMENTATION

func (p *TokenClassificationPipeline) GetModel() *backends.Model {
	return p.Model
}

// GetMetadata returns metadata information about the pipeline, in particular:
// OutputInfo: names and dimensions of the output layer used for token classification.
func (p *TokenClassificationPipeline) GetMetadata() backends.PipelineMetadata {
	return backends.PipelineMetadata{
		OutputsInfo: []backends.OutputInfo{
			{
				Name:       p.Model.OutputsMeta[0].Name,
				Dimensions: p.Model.OutputsMeta[0].Dimensions,
			},
		},
	}
}

// GetStatistics returns the runtime statistics for the pipeline.
func (p *TokenClassificationPipeline) GetStatistics() backends.PipelineStatistics {
	statistics := backends.PipelineStatistics{}
	statistics.ComputeTokenizerStatistics(p.Model.Tokenizer.TokenizerTimings)
	statistics.ComputeOnnxStatistics(p.PipelineTimings)
	return statistics
}

// Validate checks that the pipeline is valid.
func (p *TokenClassificationPipeline) Validate() error {
	var validationErrors []error
	if p.Model.Tokenizer == nil {
		validationErrors = append(validationErrors, fmt.Errorf("token classification pipeline requires a tokenizer"))
	}
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
func (p *TokenClassificationPipeline) Preprocess(batch *backends.PipelineBatch, inputs []string) error {
	if p.SplitWords {
		return fmt.Errorf("split-words enabled: use RunWords/PreprocessWords for [][]string inputs")
	}
	start := time.Now()
	backends.TokenizeInputs(batch, p.Model.Tokenizer, inputs)
	atomic.AddUint64(&p.Model.Tokenizer.TokenizerTimings.NumCalls, 1)
	atomic.AddUint64(&p.Model.Tokenizer.TokenizerTimings.TotalNS, safeconv.DurationToU64(time.Since(start)))
	err := backends.CreateInputTensors(batch, p.Model, p.Runtime)
	return err
}

// PreprocessWords tokenizes pre-split words and maps tokens to word IDs via offsets.
func (p *TokenClassificationPipeline) PreprocessWords(batch *backends.PipelineBatch, inputs [][]string) error {
	start := time.Now()
	// Join words with single spaces to simulate pretokenized behavior
	joined := make([]string, len(inputs))
	wordBoundaries := make([][][2]uint, len(inputs))
	// local helper to convert non-negative int to uint safely
	toUintNonNeg := func(i int) uint {
		if i < 0 {
			return 0
		}
		return uint(i)
	}
	for i, words := range inputs {
		joined[i] = strings.Join(words, " ")
		// compute boundaries in joined string
		var boundaries [][2]uint
		pos := 0
		for wIdx, w := range words {
			startPos := pos
			endPos := pos + len(w)
			// clamp to non-negative and convert safely to uint
			// ensure non-negative before converting to uint
			if startPos < 0 {
				startPos = 0
			}
			if endPos < 0 {
				endPos = 0
			}
			boundaries = append(boundaries, [2]uint{toUintNonNeg(startPos), toUintNonNeg(endPos)})
			// add one space after every word except last
			if wIdx < len(words)-1 {
				pos = endPos + 1
			} else {
				pos = endPos
			}
		}
		wordBoundaries[i] = boundaries
	}
	backends.TokenizeInputs(batch, p.Model.Tokenizer, joined)
	atomic.AddUint64(&p.Model.Tokenizer.TokenizerTimings.NumCalls, 1)
	atomic.AddUint64(&p.Model.Tokenizer.TokenizerTimings.TotalNS, safeconv.DurationToU64(time.Since(start)))

	// Map token offsets to word indices
	for i := range batch.Input {
		input := batch.Input[i]
		boundaries := wordBoundaries[i]
		wordIDs := make([]int, len(input.Offsets))
		for t := range input.Offsets {
			if input.SpecialTokensMask[t] > 0 {
				wordIDs[t] = -1
				continue
			}
			tokStart := input.Offsets[t][0]
			tokEnd := input.Offsets[t][1]
			id := -1
			for w := range boundaries {
				b := boundaries[w]
				// assign if token lies within the word span
				if tokStart >= b[0] && tokEnd <= b[1] {
					id = w
					break
				}
			}
			wordIDs[t] = id
		}
		batch.Input[i].WordIDs = wordIDs
		// also set raw to joined string for offsets consistency
		batch.Input[i].Raw = joined[i]
	}
	return backends.CreateInputTensors(batch, p.Model, p.Runtime)
}

// Forward performs the forward inference of the pipeline.
func (p *TokenClassificationPipeline) Forward(batch *backends.PipelineBatch) error {
	start := time.Now()
	err := backends.RunSessionOnBatch(batch, p.BasePipeline)
	if err != nil {
		return err
	}
	atomic.AddUint64(&p.PipelineTimings.NumCalls, 1)
	atomic.AddUint64(&p.PipelineTimings.TotalNS, safeconv.DurationToU64(time.Since(start)))
	return nil
}

// Postprocess function for a token classification pipeline.
func (p *TokenClassificationPipeline) Postprocess(batch *backends.PipelineBatch) (*TokenClassificationOutput, error) {
	if batch.Size == 0 {
		return &TokenClassificationOutput{}, nil
	}
	output := batch.OutputValues[0]
	var outputCast [][][]float32
	switch v := output.(type) {
	case [][][]float32:
		for batchIndex, tokens := range v {
			v[batchIndex] = make([][]float32, len(tokens))
			for tokenIndex, tokenLogits := range tokens {
				v[batchIndex][tokenIndex] = vectorutil.SoftMax(tokenLogits)
			}
		}
		outputCast = v
	default:
		return nil, fmt.Errorf("expected 3D output, got type %T", output)
	}
	// now convert the logits to the predictions of actual entities
	classificationOutput := TokenClassificationOutput{
		Entities: make([][]Entity, batch.Size),
	}
	for i, input := range batch.Input {
		preEntities := p.GatherPreEntities(input, outputCast[i])
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
func (p *TokenClassificationPipeline) GatherPreEntities(input backends.TokenizedInput, output [][]float32) []Entity {
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
		// In split-words mode, grouping will use offsets between tokens rather than IsSubword.
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
	newEntity := Entity{}
	word, err := backends.Decode(tokens, p.Model.Tokenizer, true)
	if err != nil {
		return newEntity, err
	}
	var score float32
	var label string
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
			averages[i] = vectorutil.Mean(s)
		}
		entityIdx, maxScore, err := vectorutil.ArgMax(averages)
		if err != nil {
			return newEntity, err
		}
		entityLabel, ok := p.IDLabelMap[entityIdx]
		if !ok {
			return newEntity, fmt.Errorf("could not determine entity type for input %s, predicted entity index %d", word, entityIdx)
		}
		score = maxScore
		label = entityLabel
	case "MAX":
		var maxScore float32
		var maxIdx int
		for _, e := range entities {
			idx, score, err := vectorutil.ArgMax(e.Scores)
			if err != nil {
				return newEntity, err
			}
			if score >= maxScore {
				maxScore = score
				maxIdx = idx
			}
		}
		entityLabel, ok := p.IDLabelMap[maxIdx]
		if !ok {
			return Entity{}, fmt.Errorf("could not determine entity type for input %s, predicted entity index %d", word, maxIdx)
		}
		score = maxScore
		label = entityLabel
	case "FIRST":
		entityIdx, maxScore, err := vectorutil.ArgMax(entities[0].Scores)
		if err != nil {
			return newEntity, err
		}
		entityLabel, ok := p.IDLabelMap[entityIdx]
		if !ok {
			return Entity{}, fmt.Errorf("could not determine entity type for input %s, predicted entity index %d", word, entityIdx)
		}
		score = maxScore
		label = entityLabel
	default:
		return Entity{}, fmt.Errorf("aggregation strategy %s not recognized", p.AggregationStrategy)
	}
	return Entity{
		Entity:  label,
		Score:   score,
		Word:    word,
		TokenID: tokens,
		Start:   entities[0].Start,
		End:     entities[len(entities)-1].End,
	}, nil
}

func (p *TokenClassificationPipeline) aggregateWords(entities []Entity) ([]Entity, error) {
	var wordGroup []Entity
	var wordEntities []Entity
	for _, entity := range entities {
		if len(wordGroup) == 0 {
			wordGroup = []Entity{entity}
			continue
		}
		// Default behavior: group by IsSubword boundaries
		groupBreak := !entity.IsSubword
		if p.SplitWords {
			// In split-words mode, we group by contiguous tokens of the same word boundary since we pretokenized.
			// Since preEntities don’t carry word IDs directly, simulate grouping by contiguous offsets:
			// break group if there is a gap between previous End and current Start (space) or heuristic non-subword.
			// TODO: eventually we should export word IDs from the tokenizer to avoid this heuristic but the rust tokenizer bindings don't expose this yet
			// and we also use other tokenizers in go backend.
			prev := wordGroup[len(wordGroup)-1]
			// if there is a gap in offsets consider it a new word
			groupBreak = entity.Start > prev.End
		}
		if groupBreak {
			aggregated, err := p.aggregateWord(wordGroup)
			if err != nil {
				return nil, err
			}
			wordEntities = append(wordEntities, aggregated)
			wordGroup = []Entity{entity}
		} else {
			wordGroup = append(wordGroup, entity)
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

func (p *TokenClassificationPipeline) Aggregate(input backends.TokenizedInput, preEntities []Entity) ([]Entity, error) {
	entities := make([]Entity, len(preEntities))
	var aggregationError error
	if p.AggregationStrategy == "SIMPLE" || p.AggregationStrategy == "NONE" {
		for i, preEntity := range preEntities {
			entityIdx, score, argMaxErr := vectorutil.ArgMax(preEntity.Scores)
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
	score := vectorutil.Mean(scores)
	// note: here we directly appeal to the tokenizer decoder with the tokenIds
	// in the python code they pass the words to a token_to_string_method
	word, err := backends.Decode(tokens, p.Model.Tokenizer, true)
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
func (p *TokenClassificationPipeline) Run(inputs []string) (backends.PipelineBatchOutput, error) {
	return p.RunPipeline(inputs)
}

// RunPipeline is like Run but returns the concrete type rather than the interface.
func (p *TokenClassificationPipeline) RunPipeline(inputs []string) (*TokenClassificationOutput, error) {
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

// RunWords runs the pipeline for pre-split word inputs.
// Each input is a slice of words representing a pretokenized sentence.
// This is particularly useful when the user wants to control tokenization because of special tokens,
// hashtags, or other domain-specific tokenization needs.
func (p *TokenClassificationPipeline) RunWords(inputs [][]string) (*TokenClassificationOutput, error) {
	var runErrors []error
	batch := backends.NewBatch(len(inputs))
	defer func(*backends.PipelineBatch) {
		runErrors = append(runErrors, batch.Destroy())
	}(batch)
	runErrors = append(runErrors, p.PreprocessWords(batch, inputs))
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
