package backends

import (
	"bytes"
	"log"

	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"

	"github.com/knights-analytics/hugot/util/safeconv"
)

type GoTokenizer struct {
	Tokenizer *tokenizer.Tokenizer
}

func loadGoTokenizer(tokenizerBytes []byte, model *Model) error {
	tk, tkErr := pretrained.FromReader(bytes.NewReader(tokenizerBytes))
	if tkErr != nil {
		return tkErr
	}
	model.Tokenizer = &Tokenizer{Runtime: "GO", GoTokenizer: &GoTokenizer{Tokenizer: tk}, TokenizerTimings: &timings{}, MaxAllowedTokens: model.MaxPositionEmbeddings, Destroy: func() error {
		return nil
	}}
	return nil
}

func tokenizeInputsGo(batch *PipelineBatch, tk *Tokenizer, inputs []string) {
	outputs := make([]TokenizedInput, len(inputs))
	maxSequence := 0
	goTK := tk.GoTokenizer.Tokenizer
	for i, input := range inputs {

		output, err := goTK.EncodeSingle(input, true)
		if err != nil {
			log.Fatal(err)
		}

		if tk.MaxAllowedTokens > 0 && len(output.Tokens) > tk.MaxAllowedTokens {
			output.Tokens = output.Tokens[:tk.MaxAllowedTokens]
			output.Ids = output.Ids[:min(len(output.Ids), tk.MaxAllowedTokens)]
			output.TypeIds = output.TypeIds[:min(len(output.TypeIds), tk.MaxAllowedTokens)]
			output.AttentionMask = output.AttentionMask[:min(len(output.AttentionMask), tk.MaxAllowedTokens)]
			output.SpecialTokenMask = output.SpecialTokenMask[:min(len(output.SpecialTokenMask), tk.MaxAllowedTokens)]
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
			TokenIDs:          safeconv.IntSliceToUint32Slice(output.Ids),
			TypeIDs:           safeconv.IntSliceToUint32Slice(output.TypeIds),
			AttentionMask:     safeconv.IntSliceToUint32Slice(output.AttentionMask),
			MaxAttentionIndex: maxAttentionIndex,
			SpecialTokensMask: safeconv.IntSliceToUint32Slice(output.SpecialTokenMask),
			Offsets:           safeconv.IntOffsetsToUintPairs(output.Offsets), // we need the offsets here for postprocessing later
		}
		if maxAttentionIndex > maxSequence {
			maxSequence = maxAttentionIndex
		}

	}
	batch.Input = outputs
	batch.MaxSequenceLength = maxSequence + 1
}

func decodeGo(tokens []uint32, tokenizer *Tokenizer, skipSpecialTokens bool) string {
	return tokenizer.GoTokenizer.Tokenizer.Decode(safeconv.Uint32SliceToIntSlice(tokens), skipSpecialTokens)
}

// conversion helpers moved to util/safeconv.go
