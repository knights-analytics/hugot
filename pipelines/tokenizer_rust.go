//go:build !NOORT || XLA || ALL

package pipelines

import (
	"fmt"

	"github.com/daulet/tokenizers"
)

type RustTokenizer struct {
	Tokenizer *tokenizers.Tokenizer
	Options   []tokenizers.EncodeOption
}

func loadRustTokenizer(tokenizerBytes []byte, model *Model) error {
	tk, tkErr := tokenizers.FromBytes(tokenizerBytes)
	if tkErr != nil {
		return tkErr
	}

	// tokenizer init
	rustOptions, optErr := getRustTokenizerOptions(model.InputsMeta)
	if optErr != nil {
		return optErr
	}
	model.Tokenizer = &Tokenizer{Runtime: "RUST", RustTokenizer: &RustTokenizer{Tokenizer: tk, Options: rustOptions}, TokenizerTimings: &timings{}, Destroy: func() error {
		return tk.Close()
	}}
	return nil
}

func getRustTokenizerOptions(inputs []InputOutputInfo) ([]tokenizers.EncodeOption, error) {
	var encodeOptions []tokenizers.EncodeOption
	for _, input := range inputs {
		switch input.Name {
		case "input_ids":
			encodeOptions = append(encodeOptions, tokenizers.WithReturnTokens())
		case "token_type_ids":
			encodeOptions = append(encodeOptions, tokenizers.WithReturnTypeIDs())
		case "attention_mask":
			encodeOptions = append(encodeOptions, tokenizers.WithReturnAttentionMask())
		default:
			return nil, fmt.Errorf("input %s not recognized", input.Name)
		}
	}
	return encodeOptions, nil
}

func tokenizeInputsRust(batch *PipelineBatch, tk *Tokenizer, inputs []string) {
	outputs := make([]TokenizedInput, len(inputs))
	maxSequence := 0
	rustTK := tk.RustTokenizer
	for i, input := range inputs {
		output := rustTK.Tokenizer.EncodeWithOptions(input,
			true,
			rustTK.Options...,
		)

		maxAttentionIndex := 0
		for j, attentionMaskValue := range output.AttentionMask {
			if attentionMaskValue != 0 {
				maxAttentionIndex = j
			}
		}

		outputs[i] = TokenizedInput{
			Raw:               input,
			Tokens:            output.Tokens,
			TokenIDs:          output.IDs,
			TypeIDs:           output.TypeIDs,
			AttentionMask:     output.AttentionMask,
			MaxAttentionIndex: maxAttentionIndex,
			SpecialTokensMask: output.SpecialTokensMask,
			Offsets:           convertRustOffsets(output.Offsets), // we need the offsets here for postprocessing later
		}
		if maxAttentionIndex > maxSequence {
			maxSequence = maxAttentionIndex
		}
	}
	batch.Input = outputs
	batch.MaxSequenceLength = maxSequence + 1
}

func decodeRust(tokens []uint32, tokenizer *Tokenizer) string {
	return tokenizer.RustTokenizer.Tokenizer.Decode(tokens, false)
}

func convertRustOffsets(input []tokenizers.Offset) [][2]uint {
	output := make([][2]uint, len(input))
	for i, x := range input {
		output[i] = [2]uint{x[0], x[1]}
	}
	return output
}

func allInputTokensRust(pipeline *BasePipeline) {
	pipeline.Model.Tokenizer.RustTokenizer.Options = append(pipeline.Model.Tokenizer.RustTokenizer.Options,
		tokenizers.WithReturnSpecialTokensMask(),
		tokenizers.WithReturnOffsets(),
	)
}
