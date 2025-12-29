//go:build ORT || XLA || ALL

package backends

import (
	"errors"
	"fmt"
	"strings"

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
		return errors.Join(optErr, tk.Close())
	}
	model.Tokenizer = &Tokenizer{Runtime: "RUST", RustTokenizer: &RustTokenizer{Tokenizer: tk, Options: rustOptions}, TokenizerTimings: &timings{}, MaxAllowedTokens: model.MaxPositionEmbeddings, Destroy: func() error {
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
		case "position_ids":
			continue
		// GLiNER-specific inputs - handled by the GLiNER pipeline
		case "words_mask", "text_lengths", "span_idx", "span_mask":
			continue
		default:
			// Skip inputs that are handled at the model level
			lowerName := strings.ToLower(input.Name)
			if strings.HasPrefix(lowerName, "past_key_values") ||
				strings.Contains(lowerName, "pixel_values") ||
				strings.Contains(lowerName, "image") {
				continue
			}
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

		if tk.MaxAllowedTokens > 0 && len(output.Tokens) > tk.MaxAllowedTokens {
			output.Tokens = output.Tokens[:tk.MaxAllowedTokens]
			output.IDs = output.IDs[:min(len(output.IDs), tk.MaxAllowedTokens)]
			output.TypeIDs = output.TypeIDs[:min(len(output.TypeIDs), tk.MaxAllowedTokens)]
			output.AttentionMask = output.AttentionMask[:min(len(output.AttentionMask), tk.MaxAllowedTokens)]
			output.SpecialTokensMask = output.SpecialTokensMask[:min(len(output.SpecialTokensMask), tk.MaxAllowedTokens)]
			output.Offsets = output.Offsets[:min(len(output.Offsets), tk.MaxAllowedTokens)]
		}

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

func decodeRust(tokens []uint32, tokenizer *Tokenizer, skipSpecialTokens bool) string {
	return tokenizer.RustTokenizer.Tokenizer.Decode(tokens, skipSpecialTokens)
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
