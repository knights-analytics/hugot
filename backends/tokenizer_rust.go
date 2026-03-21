//go:build cgo && (ORT || XLA || ALL)

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
	rustOptions, optErr := getRustTokenizerOptions(model)
	if optErr != nil {
		return errors.Join(optErr, tk.Close())
	}
	model.Tokenizer = &Tokenizer{Runtime: "RUST", RustTokenizer: &RustTokenizer{Tokenizer: tk, Options: rustOptions}, TokenizerTimings: &timings{}, MaxAllowedTokens: model.MaxPositionEmbeddings, Destroy: func() error {
		return tk.Close()
	}}
	return nil
}

func getRustTokenizerOptions(model *Model) ([]tokenizers.EncodeOption, error) {
	var encodeOptions []tokenizers.EncodeOption
	for _, input := range model.InputsMeta {
		switch input.Name {
		case "input_ids":
			encodeOptions = append(encodeOptions, tokenizers.WithReturnTokens())
		case "token_type_ids":
			encodeOptions = append(encodeOptions, tokenizers.WithReturnTypeIDs())
		case "attention_mask":
			encodeOptions = append(encodeOptions, tokenizers.WithReturnAttentionMask())
		case "position_ids":
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

		ti := TokenizedInput{
			Raw:               input,
			Tokens:            output.Tokens,
			TokenIDs:          output.IDs,
			TypeIDs:           output.TypeIDs,
			AttentionMask:     output.AttentionMask,
			MaxAttentionIndex: maxAttentionIndex,
			SpecialTokensMask: output.SpecialTokensMask,
			Offsets:           convertRustOffsets(output.Offsets), // we need the offsets here for postprocessing later
		}
		outputs[i] = ti
		if maxAttentionIndex > maxSequence {
			maxSequence = maxAttentionIndex
		}
	}
	batch.Input = outputs
	batch.MaxSequenceLength = maxSequence + 1
}

func tokenizeInputPairsRust(batch *PipelineBatch, tk *Tokenizer, inputs [][2]string, sepToken string) {
	outputs := make([]TokenizedInput, len(inputs))
	maxSequence := 0
	rustTK := tk.RustTokenizer
	for i, inputPair := range inputs {
		fullInput := inputPair[0] + sepToken + inputPair[1]
		if sepToken == "</s>" {
			fullInput = inputPair[0] + sepToken + sepToken + inputPair[1]
		}
		output := rustTK.Tokenizer.EncodeWithOptions(fullInput,
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

		// Adjust type IDs. Since we manually concatenated, we should try to find where the second part starts.
		// However, the most robust way if we don't know the separator is to tokenize separately,
		// but daulet/tokenizers EncodeWithOptions is better if it supports pairs.
		// Given I cannot find WithEncodePair, I will use the patch logic if TypeIDs are all zero.

		maxAttentionIndex := 0
		for j, attentionMaskValue := range output.AttentionMask {
			if attentionMaskValue != 0 {
				maxAttentionIndex = j
			}
		}

		ti := TokenizedInput{
			Raw:               fullInput,
			Tokens:            output.Tokens,
			TokenIDs:          output.IDs,
			TypeIDs:           output.TypeIDs,
			AttentionMask:     output.AttentionMask,
			MaxAttentionIndex: maxAttentionIndex,
			SpecialTokensMask: output.SpecialTokensMask,
			Offsets:           convertRustOffsets(output.Offsets), // we need the offsets here for postprocessing later
		}

		outputs[i] = ti
		if maxAttentionIndex > maxSequence {
			maxSequence = maxAttentionIndex
		}
	}
	batch.Input = outputs
	batch.MaxSequenceLength = maxSequence + 1

	if sepToken == "[SEP]" {
		patchBertSequenceTokenTypeIDs(batch, sepToken)
	}
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

func allInputTokensRust(pipeline *BasePipeline) error {
	pipeline.Model.Tokenizer.RustTokenizer.Options = append(pipeline.Model.Tokenizer.RustTokenizer.Options,
		tokenizers.WithReturnSpecialTokensMask(),
		tokenizers.WithReturnOffsets(),
	)
	return nil
}
