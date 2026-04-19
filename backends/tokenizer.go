package backends

import (
	"context"
	"fmt"

	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/util/fileutil"
)

type Tokenizer struct {
	RustTokenizer    *RustTokenizer
	GoTokenizer      *GoTokenizer
	TokenizerTimings *timings
	Destroy          func() error
	Runtime          string
	MaxAllowedTokens int
}

func LoadTokenizer(ctx context.Context, model *Model, s *options.Options) error {
	if exists, err := fileutil.FileExists(ctx, fileutil.PathJoinSafe(model.Path, "tokenizer.json")); err == nil {
		if exists {
			tokenizerBytes, err := fileutil.ReadFileBytes(ctx, fileutil.PathJoinSafe(model.Path, "tokenizer.json"))
			if err != nil {
				return err
			}
			switch s.Backend {
			case "ORT", "XLA":
				return loadRustTokenizer(tokenizerBytes, model)
			case "GO":
				return loadGoTokenizer(tokenizerBytes, model)
			default:
				return fmt.Errorf("runtime %s not recognized", s.Backend)
			}
		}
	} else {
		return fmt.Errorf("error checking for existence of tokenizer.json: %w", err)
	}
	return nil
}

func TokenizeInputs(batch *PipelineBatch, tk *Tokenizer, inputs []string) {
	switch tk.Runtime {
	case "RUST":
		tokenizeInputsRust(batch, tk, inputs)
	case "GO":
		tokenizeInputsGo(batch, tk, inputs)
	}
}

func TokenizeInputPairs(batch *PipelineBatch, tk *Tokenizer, inputs [][2]string, sepToken string) {
	switch tk.Runtime {
	case "RUST":
		tokenizeInputPairsRust(batch, tk, inputs, sepToken)
	case "GO":
		tokenizeInputPairsGo(batch, tk, inputs, sepToken)
	}
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
	switch pipeline.Model.Tokenizer.Runtime {
	case "RUST":
		return allInputTokensRust(pipeline)
	case "GO":
		return allInputTokensGo(pipeline)
	}
	return fmt.Errorf("runtime %s not recognized", pipeline.Model.Tokenizer.Runtime)
}

func Decode(tokens []uint32, tokenizer *Tokenizer) (string, error) {
	switch tokenizer.Runtime {
	case "RUST":
		return decodeRust(tokens, tokenizer, true), nil
	case "GO":
		return decodeGo(tokens, tokenizer), nil
	}
	return "", fmt.Errorf("runtime %s not recognized", tokenizer.Runtime)
}
