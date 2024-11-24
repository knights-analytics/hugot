package pipelines

import (
	"bytes"
	"fmt"
	"log"

	"github.com/daulet/tokenizers"
	gotk "github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"

	util "github.com/knights-analytics/hugot/utils"
)

type Tokenizer struct {
	Runtime          string
	RustTokenizer    *tokenizers.Tokenizer
	RustOptions      []tokenizers.EncodeOption
	GoTokenizer      *gotk.Tokenizer
	TokenizerTimings *timings
}

func loadTokenizer(pipeline *basePipeline) error {
	tokenizerBytes, err := util.ReadFileBytes(util.PathJoinSafe(pipeline.ModelPath, "tokenizer.json"))
	if err != nil {
		return err
	}

	switch pipeline.Runtime {
	case "GO":
		tk, tkErr := pretrained.FromReader(bytes.NewReader(tokenizerBytes))
		if tkErr != nil {
			return tkErr
		}
		pipeline.Tokenizer = &Tokenizer{Runtime: "GO", GoTokenizer: tk, TokenizerTimings: &timings{}}
		return nil
	case "ORT":
		tk, tkErr := tokenizers.FromBytes(tokenizerBytes)
		if tkErr != nil {
			return tkErr
		}

		// tokenizer init
		rustOptions, optErr := getTokenizerOptions(pipeline.InputsMeta)
		if optErr != nil {
			return optErr
		}
		pipeline.Tokenizer = &Tokenizer{Runtime: "RUST", RustTokenizer: tk, RustOptions: rustOptions, TokenizerTimings: &timings{}}
		return nil
	default:
		return fmt.Errorf("runtime %s not recognized", pipeline.Runtime)
	}
}

func getTokenizerOptions(inputs []InputOutputInfo) ([]tokenizers.EncodeOption, error) {
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

func tokenizeInputs(batch *PipelineBatch, tk *Tokenizer, inputs []string) {
	outputs := make([]tokenizedInput, len(inputs))
	maxSequence := 0
	for i, input := range inputs {

		switch tk.Runtime {
		case "GO":
			output, err := tk.GoTokenizer.EncodeSingle(input, true)
			if err != nil {
				log.Fatal(err)
			}

			maxAttentionIndex := 0
			for j, attentionMaskValue := range output.AttentionMask {
				if attentionMaskValue != 0 {
					maxAttentionIndex = j
				}
			}

			outputs[i] = tokenizedInput{
				Raw:               input,
				Tokens:            output.Tokens,
				TokenIDs:          convertIntsToUints(output.Ids),
				TypeIDs:           convertIntsToUints(output.TypeIds),
				AttentionMask:     convertIntsToUints(output.AttentionMask),
				MaxAttentionIndex: maxAttentionIndex,
				SpecialTokensMask: convertIntsToUints(output.SpecialTokenMask),
				Offsets:           convertGoOffsets(output.Offsets), // we need the offsets here for postprocessing later
			}
			if maxAttentionIndex > maxSequence {
				maxSequence = maxAttentionIndex
			}
		case "RUST":
			output := tk.RustTokenizer.EncodeWithOptions(input,
				true,
				tk.RustOptions...,
			)

			maxAttentionIndex := 0
			for j, attentionMaskValue := range output.AttentionMask {
				if attentionMaskValue != 0 {
					maxAttentionIndex = j
				}
			}

			outputs[i] = tokenizedInput{
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
	}
	batch.Input = outputs
	batch.MaxSequenceLength = maxSequence + 1
}

func convertIntsToUints(input []int) []uint32 {
	output := make([]uint32, len(input))
	for i, x := range input {
		output[i] = uint32(x)
	}
	return output
}

func convertUintsToInts(input []uint32) []int {
	output := make([]int, len(input))
	for i, x := range input {
		output[i] = int(x)
	}
	return output
}

func convertGoOffsets(input [][]int) [][2]uint {
	output := make([][2]uint, len(input))
	for i, x := range input {
		output[i] = [2]uint{uint(x[0]), uint(x[1])}
	}
	return output
}

func convertRustOffsets(input []tokenizers.Offset) [][2]uint {
	output := make([][2]uint, len(input))
	for i, x := range input {
		output[i] = [2]uint{x[0], x[1]}
	}
	return output
}
