package datasets

import (
	"regexp"
	"slices"
	"strings"
	"testing"

	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/pipelines"
)

// tokenized builds the tokenizer output for a sentence from (start, end, special) triples.
func tokenized(spans ...[3]uint) backends.TokenizedInput {
	var input backends.TokenizedInput
	for _, span := range spans {
		input.Offsets = append(input.Offsets, [2]uint{span[0], span[1]})
		input.SpecialTokensMask = append(input.SpecialTokensMask, uint32(span[2]))
	}
	return input
}

func TestWordStartTokens(t *testing.T) {
	words := []string{"EU", "rejects", "Germanic"} // "EU rejects Germanic"
	input := tokenized(
		[3]uint{0, 0, 1},   // [CLS]
		[3]uint{0, 2, 0},   // eu
		[3]uint{3, 10, 0},  // rejects
		[3]uint{11, 17, 0}, // german
		[3]uint{17, 19, 0}, // ##ic
		[3]uint{0, 0, 1},   // [SEP]
	)
	if got := WordStartTokens(words, input, 8); !slices.Equal(got, []int{1, 2, 3}) {
		t.Fatalf("first tokens %v, want [1 2 3]", got)
	}
	// Truncated to 3 tokens, the last word has no token left.
	if got := WordStartTokens(words, input, 3); !slices.Equal(got, []int{1, 2, -1}) {
		t.Fatalf("truncated first tokens %v, want [1 2 -1]", got)
	}
}

func TestWordStartTokensSplitWord(t *testing.T) {
	words := []string{"Kowalczyk", "said"} // "Kowalczyk said"
	input := tokenized(
		[3]uint{0, 0, 1},   // [CLS]
		[3]uint{0, 3, 0},   // ko
		[3]uint{3, 6, 0},   // ##wal
		[3]uint{6, 9, 0},   // ##czyk
		[3]uint{10, 14, 0}, // said
		[3]uint{0, 0, 1},   // [SEP]
	)
	if got := WordStartTokens(words, input, 6); !slices.Equal(got, []int{1, 4}) {
		t.Fatalf("first tokens %v, want [1 4]", got)
	}
}

func TestAlignTags(t *testing.T) {
	d, err := NewInMemoryTokenClassificationDataset([]TokenClassificationExample{{Tokens: []string{"x"}, Tags: []string{"O"}}}, 1, nil)
	if err != nil {
		t.Fatal(err)
	}
	d.labelIDs = map[string]int{"O": 0, "B-PER": 1, "B-LOC": 5}
	examples := []TokenClassificationExample{{Tokens: []string{"Kowalczyk", "said"}, Tags: []string{"B-PER", "O"}}}
	input := tokenized([3]uint{0, 0, 1}, [3]uint{0, 3, 0}, [3]uint{3, 9, 0}, [3]uint{10, 14, 0}, [3]uint{0, 0, 1})
	labels, mask, err := d.alignTags([][]string{examples[0].Tokens}, [][]string{examples[0].Tags}, []backends.TokenizedInput{input}, 6)
	if err != nil {
		t.Fatal(err)
	}
	// Only the first token of each word carries a tag; sub-word, special and padding tokens are masked.
	if !slices.Equal(mask, []bool{false, true, false, true, false, false}) {
		t.Fatalf("mask %v", mask)
	}
	if labels[1] != 1 || labels[3] != 0 {
		t.Fatalf("labels %v, want B-PER (1) at token 1 and O (0) at token 3", labels)
	}

	examples[0].Tags[0] = "B-ORG"
	if _, _, err := d.alignTags([][]string{examples[0].Tokens}, [][]string{examples[0].Tags}, []backends.TokenizedInput{input}, 6); err == nil || !strings.Contains(err.Error(), `"B-ORG" is not in the model's id2label map (B-LOC, B-PER, O)`) {
		t.Fatalf("got %v, want an unknown-tag error listing the known tags", err)
	}
}

func TestTokenClassificationValidate(t *testing.T) {
	if _, err := NewInMemoryTokenClassificationDataset([]TokenClassificationExample{{Tokens: []string{"x"}, Tags: []string{"O"}}}, 0, nil); err == nil {
		t.Fatal("expected an error for batch size 0")
	}
	d, _ := NewInMemoryTokenClassificationDataset([]TokenClassificationExample{{Tokens: []string{"x"}, Tags: []string{"O"}}}, 1, nil)
	if err := d.SetTokenizationPipeline(nil); err == nil {
		t.Fatal("expected an error for a nil pipeline")
	}
}

// splitLetterDigit splits on whitespace, then between a letter and the digit that follows it, so
// "Row7" becomes "Row" and "7" while "14C" (digit, then letter) stays whole.
func splitLetterDigit(text string) []string {
	return strings.Fields(letterDigit.ReplaceAllString(text, "$1 $2"))
}

var letterDigit = regexp.MustCompile(`([A-Za-z])(\d)`)

func TestExampleWords(t *testing.T) {
	withSplitter := &pipelines.TokenClassificationPipeline{WordSplitter: splitLetterDigit}
	plain := &pipelines.TokenClassificationPipeline{}

	words, err := TokenClassificationExample{Text: "Seat 14C Row7"}.Words(withSplitter)
	if err != nil || !slices.Equal(words, []string{"Seat", "14C", "Row", "7"}) {
		t.Fatalf("got %q, %v; want the splitter's words", words, err)
	}
	words, err = TokenClassificationExample{Tokens: []string{"a", "b"}}.Words(plain)
	if err != nil || !slices.Equal(words, []string{"a", "b"}) {
		t.Fatalf("got %q, %v; want the tokens as given", words, err)
	}
	// Text and tokens together must agree, so a stale pre-split file is caught rather than trained on.
	if _, err = (TokenClassificationExample{Text: "Row7", Tokens: []string{"Row7"}}).Words(withSplitter); err == nil || !strings.Contains(err.Error(), "differs from its tokens") {
		t.Fatalf("got %v, want a mismatch error", err)
	}
	if _, err = (TokenClassificationExample{Text: "Row7"}).Words(plain); err == nil || !strings.Contains(err.Error(), "no word splitter") {
		t.Fatalf("got %v, want an error for text without a splitter", err)
	}
	if _, err = (TokenClassificationExample{}).Words(plain); err == nil {
		t.Fatal("want an error for an example with no words")
	}
}
