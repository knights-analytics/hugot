package evaluation

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"

	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/datasets"
	"github.com/knights-analytics/hugot/pipelines"
	"github.com/knights-analytics/hugot/util/vectorutil"
)

// EvaluateTokenClassification runs a token classification pipeline over annotated examples and
// returns the entity-level scores of its predictions against the examples' tags. Span-annotated
// examples are converted to word tags as for training, and examples not answered "accept" are
// left out.
func EvaluateTokenClassification(ctx context.Context, pipeline *pipelines.TokenClassificationPipeline, examples []datasets.TokenClassificationExample, batchSize int) (EntityReport, error) {
	examples = slices.DeleteFunc(slices.Clone(examples), func(e datasets.TokenClassificationExample) bool { return !e.IsAccepted() })
	gold := make([][]string, len(examples))
	for i, example := range examples {
		_, tags, err := example.Resolve(pipeline)
		if err != nil {
			return EntityReport{}, err
		}
		gold[i] = tags
	}
	predicted, err := PredictWordTags(ctx, pipeline, examples, batchSize)
	if err != nil {
		return EntityReport{}, err
	}
	return ScoreEntities(gold, predicted)
}

// PredictWordTags returns the tag the pipeline's model predicts for each word of each example: the
// most likely label of the word's first token, the same token training assigns the word's tag to.
// A word that has no token (because the input was truncated) is predicted "O".
func PredictWordTags(ctx context.Context, pipeline *pipelines.TokenClassificationPipeline, examples []datasets.TokenClassificationExample, batchSize int) ([][]string, error) {
	if pipeline == nil {
		return nil, errors.New("pipeline is required")
	}
	if batchSize <= 0 {
		return nil, fmt.Errorf("batch size must be greater than 0, got %d", batchSize)
	}
	tags := make([][]string, 0, len(examples))
	for start := 0; start < len(examples); start += batchSize {
		batchTags, err := predictBatch(ctx, pipeline, examples[start:min(start+batchSize, len(examples))])
		if err != nil {
			return nil, err
		}
		tags = append(tags, batchTags...)
	}
	return tags, nil
}

func predictBatch(ctx context.Context, pipeline *pipelines.TokenClassificationPipeline, examples []datasets.TokenClassificationExample) (tags [][]string, err error) {
	texts := make([]string, len(examples))
	words := make([][]string, len(examples))
	for i, example := range examples {
		exampleWords, _, wordsErr := example.Resolve(pipeline)
		if wordsErr != nil {
			return nil, wordsErr
		}
		words[i] = exampleWords
		texts[i] = strings.Join(exampleWords, " ")
	}
	batch := backends.NewBatch(len(examples))
	defer func() {
		err = errors.Join(err, batch.Destroy())
	}()
	backends.TokenizeInputs(batch, pipeline.Model.Tokenizer, texts)
	if err = backends.CreateInputTensors(batch, pipeline.Model); err != nil {
		return nil, err
	}
	if err = backends.RunSessionOnBatch(ctx, batch, pipeline.BasePipeline); err != nil {
		return nil, err
	}
	logits, ok := batch.OutputValues[0].([][][]float32)
	if !ok {
		return nil, fmt.Errorf("expected token logits of shape [batch, sequence, labels], got %T", batch.OutputValues[0])
	}
	tags = make([][]string, len(examples))
	for i := range examples {
		starts := datasets.WordStartTokens(words[i], batch.Input[i], len(logits[i]))
		tags[i] = make([]string, len(words[i]))
		for w, token := range starts {
			if token < 0 {
				tags[i][w] = "O"
				continue
			}
			id, _, argErr := vectorutil.ArgMax(logits[i][token])
			if argErr != nil {
				return nil, argErr
			}
			label, found := pipeline.Model.IDLabelMap[id]
			if !found {
				return nil, fmt.Errorf("predicted label id %d is not in the model's id2label map", id)
			}
			tags[i][w] = label
		}
	}
	return tags, nil
}
