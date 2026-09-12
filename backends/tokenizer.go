package backends

import (
	"context"
	"fmt"

	"github.com/knights-analytics/hugot/util/fileutil"
)

type Tokenizer struct {
	GoTokenizer      *GoTokenizer
	close            func() error
	MaxAllowedTokens int
}

func (t *Tokenizer) Close() error {
	if t == nil || t.close == nil {
		return nil
	}
	return t.close()
}

func LoadTokenizer(ctx context.Context, model *Model) error {
	// Check whether the model contains a tokenizer configuration.
	if exists, err := fileutil.FileExists(ctx, fileutil.PathJoinSafe(model.Path, "tokenizer.json")); err == nil {
		if exists {
			// Read the tokenizer configuration before applying compatibility fixes.
			tokenizerBytes, err := fileutil.ReadFileBytes(ctx, fileutil.PathJoinSafe(model.Path, "tokenizer.json"))
			if err != nil {
				return err
			}
			return loadGoTokenizer(tokenizerBytes, model)
		}
	} else {
		return fmt.Errorf("error checking for existence of tokenizer.json: %w", err)
	}
	return nil
}

func TokenizeInputs(batch *PipelineBatch, tk *Tokenizer, inputs []string) {
	tokenizeInputsGo(batch, tk, inputs)
}

func TokenizeInputPairs(batch *PipelineBatch, tk *Tokenizer, inputs [][2]string, sepToken string) {
	tokenizeInputPairsGo(batch, tk, inputs, sepToken)
}

func patchBertSequenceTokenTypeIDs(batch *PipelineBatch, sepToken string) {
	// Fix token_type_ids for BERT-style models when we manually concatenated the pair as a single sequence.
	// Pattern expected: [CLS] query [SEP] doc [SEP]
	// HF sets token_type_ids=0 up to and including first [SEP], then 1 for remainder (including final [SEP]).
	for index := range batch.Input {
		input := &batch.Input[index]
		// Only adjust if type ids exist and are all zero
		allZero := true
		for _, t := range input.TypeIDs {
			if t != 0 {
				allZero = false
				break
			}
		}
		if !allZero || len(input.TypeIDs) == 0 {
			continue
		}
		// Find first [SEP] token index (skip position 0 which should be [CLS])
		firstSep := -1
		for iTok := 1; iTok < len(input.Tokens); iTok++ {
			if input.Tokens[iTok] == sepToken {
				firstSep = iTok
				break
			}
		}
		if firstSep == -1 || firstSep == len(input.Tokens)-1 { // nothing to split
			continue
		}
		for iTok := firstSep + 1; iTok < len(input.TypeIDs); iTok++ {
			input.TypeIDs[iTok] = 1
		}
	}
}

func AllInputTokens(pipeline *BasePipeline) error {
	return allInputTokensGo(pipeline)
}

func Decode(tokens []uint32, tokenizer *Tokenizer) (string, error) {
	return decodeGo(tokens, tokenizer), nil
}
