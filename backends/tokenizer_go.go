package backends

import (
	"fmt"
	"strings"

	"github.com/gomlx/go-huggingface/tokenizers/api"
	"github.com/gomlx/go-huggingface/tokenizers/hftokenizer"
	"github.com/knights-analytics/hugot/util/safeconv"
)

type GoTokenizer struct {
	Tokenizer     api.Tokenizer
	TypeIDs       bool
	AttentionMask bool
}

func loadGoTokenizer(tokenizerBytes []byte, model *Model) error {
	tk, tkErr := hftokenizer.NewFromContent(nil, tokenizerBytes)
	if tkErr != nil {
		return tkErr
	}

	goOptions, typeIDs, attentionMask, optErr := getGoTokenizerOptions(model)
	if optErr != nil {
		return optErr
	}

	optErr = tk.With(goOptions)
	if optErr != nil {
		return optErr
	}

	model.Tokenizer = &Tokenizer{
		Runtime: "GO",
		GoTokenizer: &GoTokenizer{
			Tokenizer:     tk,
			TypeIDs:       typeIDs,
			AttentionMask: attentionMask,
		},
		TokenizerTimings: &timings{},
		MaxAllowedTokens: model.MaxPositionEmbeddings,
		Destroy: func() error {
			return nil
		},
	}
	return nil
}

func getGoTokenizerOptions(model *Model) (api.EncodeOptions, bool, bool, error) {
	var encodeOptions api.EncodeOptions
	var typeIDs, attentionMask bool
	for _, input := range model.InputsMeta {
		switch input.Name {
		case "input_ids":
			continue
		case "token_type_ids":
			typeIDs = true
		case "attention_mask":
			attentionMask = true
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
			return encodeOptions, false, false, fmt.Errorf("input %s not recognized", input.Name)
		}
	}
	encodeOptions.AddSpecialTokens = true
	encodeOptions.MaxLen = model.MaxPositionEmbeddings
	return encodeOptions, typeIDs, attentionMask, nil
}

func tokenizeInputsGo(batch *PipelineBatch, tk *Tokenizer, inputs []string) {
	outputs := make([]TokenizedInput, len(inputs))
	maxSequence := 0
	goTK := tk.GoTokenizer.Tokenizer
	for i, input := range inputs {
		output := goTK.EncodeWithAnnotations(input)

		numTokens := len(output.IDs)
		var typeIDs []int
		if tk.GoTokenizer.TypeIDs {
			typeIDs = make([]int, numTokens) // defaults to 0
		}
		var attentionMask []int
		if tk.GoTokenizer.AttentionMask {
			attentionMask = make([]int, numTokens)
			for j := range numTokens {
				attentionMask[j] = 1
			}
		}

		maxAttentionIndex := 0
		if tk.GoTokenizer.AttentionMask {
			maxAttentionIndex = numTokens - 1
		}

		ti := TokenizedInput{
			Raw:               input,
			Tokens:            getGoTokens(output.IDs, tk),
			TokenIDs:          safeconv.IntSliceToUint32Slice(output.IDs),
			TypeIDs:           safeconv.IntSliceToUint32Slice(typeIDs),
			AttentionMask:     safeconv.IntSliceToUint32Slice(attentionMask),
			MaxAttentionIndex: maxAttentionIndex,
			SpecialTokensMask: safeconv.IntSliceToUint32Slice(output.SpecialTokensMask),
			Offsets:           convertGoOffsets(output.Spans), // we need the offsets here for postprocessing later
		}
		outputs[i] = ti
		if maxAttentionIndex > maxSequence {
			maxSequence = maxAttentionIndex
		}
	}
	batch.Input = outputs
	batch.MaxSequenceLength = maxSequence + 1
}

func tokenizeInputPairsGo(batch *PipelineBatch, tk *Tokenizer, inputs [][2]string, sepToken string) {
	outputs := make([]TokenizedInput, len(inputs))
	maxSequence := 0

	for i, inputPair := range inputs {
		concatenatedString := inputPair[0] + sepToken + inputPair[1]
		if sepToken == "</s>" {
			concatenatedString = inputPair[0] + sepToken + sepToken + inputPair[1]
		}
		tokenizeInputsGo(batch, tk, []string{concatenatedString})
		outputs[i] = batch.Input[0]
		if outputs[i].MaxAttentionIndex > maxSequence {
			maxSequence = outputs[i].MaxAttentionIndex
		}
	}

	batch.Input = outputs
	batch.MaxSequenceLength = maxSequence + 1

	if tk.GoTokenizer.TypeIDs && sepToken == "[SEP]" {
		patchBertSequenceTokenTypeIDs(batch, sepToken)
	}
}

func decodeGo(tokens []uint32, tokenizer *Tokenizer) string {
	return tokenizer.GoTokenizer.Tokenizer.Decode(safeconv.Uint32SliceToIntSlice(tokens))
}

func getGoTokens(ids []int, tokenizer *Tokenizer) []string {
	tokens := make([]string, len(ids))
	for i, id := range ids {
		tokens[i] = tokenizer.GoTokenizer.Tokenizer.Decode([]int{id})
	}
	return tokens
}

func convertGoOffsets(spans []api.TokenSpan) [][2]uint {
	offsets := make([][2]uint, len(spans))
	for j, span := range spans {
		if span.Start == -1 {
			span.Start = 0
		}
		if span.End == -1 {
			span.End = 0
		}
		offsets[j] = [2]uint{uint(span.Start), uint(span.End)}
	}
	return offsets
}

func allInputTokensGo(pipeline *BasePipeline) error {
	err := pipeline.Model.Tokenizer.GoTokenizer.Tokenizer.With(api.EncodeOptions{
		AddSpecialTokens:         true,
		IncludeSpans:             true,
		IncludeSpecialTokensMask: true,
	})
	return err
}
