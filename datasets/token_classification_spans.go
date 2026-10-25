package datasets

import (
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"unicode/utf8"

	"github.com/knights-analytics/hugot/pipelines"
)

// EntitySpan is an entity annotated by character offsets into an example's Text, as exported by
// span-based annotation tools such as Prodigy (ner_manual). Start and End count characters
// (Unicode code points, as Python does), not bytes; End is exclusive.
type EntitySpan struct {
	Label string `json:"label"`
	Start int    `json:"start"`
	End   int    `json:"end"`
}

// SourceToken is a token of an example's Text as the annotation tool split it (Prodigy's
// "tokens"), with character offsets like [EntitySpan].
type SourceToken struct {
	Text  string `json:"text"`
	Start int    `json:"start"`
	End   int    `json:"end"`
}

// AnswerAccept is the annotation answer of an example to train on. Prodigy exports also mark
// examples "reject" or "ignore"; the dataset skips those.
const AnswerAccept = "accept"

// IsSpanFormat reports whether the example is annotated with character spans rather than word
// tags.
func (e TokenClassificationExample) IsSpanFormat() bool {
	return e.Spans != nil || len(e.SourceTokens) > 0
}

// IsAccepted reports whether the example should be trained on: it has no annotation answer, or
// the answer is "accept".
func (e TokenClassificationExample) IsAccepted() bool {
	return e.Answer == "" || e.Answer == AnswerAccept
}

// UnmarshalJSON reads both example formats. Word-level lines give "tokens" as a list of words;
// span-based lines (Prodigy) give "tokens" as a list of objects with offsets, and "spans".
func (e *TokenClassificationExample) UnmarshalJSON(data []byte) error {
	var raw struct {
		Text   string          `json:"text"`
		Tokens json.RawMessage `json:"tokens"`
		Tags   []string        `json:"tags"`
		Spans  []EntitySpan    `json:"spans"`
		Answer string          `json:"answer"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	*e = TokenClassificationExample{Text: raw.Text, Tags: raw.Tags, Spans: raw.Spans, Answer: raw.Answer}
	if len(raw.Tokens) == 0 || string(raw.Tokens) == "null" {
		return nil
	}
	if err := json.Unmarshal(raw.Tokens, &e.Tokens); err == nil {
		return nil
	}
	e.Tokens = nil
	if err := json.Unmarshal(raw.Tokens, &e.SourceTokens); err != nil {
		return fmt.Errorf("tokens must be a list of words or a list of {text, start, end} tokens: %w", err)
	}
	return nil
}

// Resolve returns the example's words and one tag per word.
//
// Word-level examples give their tags directly, and their words as Tokens or as Text split by the
// pipeline's word splitter. Span-based examples are converted: the words are the pipeline's word
// splitter applied to Text when it has one, otherwise the annotation tool's SourceTokens; each
// span must start and end on word boundaries, and becomes IOB tags (B-LABEL on its first word,
// I-LABEL on the rest, O elsewhere).
func (e TokenClassificationExample) Resolve(pipeline *pipelines.TokenClassificationPipeline) (words, tags []string, err error) {
	if !e.IsAccepted() {
		return nil, nil, fmt.Errorf("example %q has answer %q; only accepted examples can be used", e.Text, e.Answer)
	}
	if !e.IsSpanFormat() {
		words, err = e.Words(pipeline)
		if err != nil {
			return nil, nil, err
		}
		if len(e.Tags) != len(words) {
			return nil, nil, fmt.Errorf("example %q has %d words but %d tags", strings.Join(words, " "), len(words), len(e.Tags))
		}
		return words, e.Tags, nil
	}
	if len(e.Tags) > 0 {
		return nil, nil, fmt.Errorf("example %q has both spans and tags; give one or the other", e.Text)
	}
	if e.Text == "" {
		return nil, nil, errors.New("a span-annotated example needs its text")
	}
	byteOffset := characterByteOffsets(e.Text)
	words, wordBounds, err := e.spanWords(pipeline, byteOffset)
	if err != nil {
		return nil, nil, err
	}
	tags, err = spanTags(e.Text, e.Spans, wordBounds, byteOffset)
	if err != nil {
		return nil, nil, err
	}
	return words, tags, nil
}

// spanWords splits a span-annotated example into words, with each word's byte range in Text.
func (e TokenClassificationExample) spanWords(pipeline *pipelines.TokenClassificationPipeline, byteOffset []int) ([]string, [][2]int, error) {
	if pipeline != nil && pipeline.WordSplitter != nil {
		words := pipeline.WordSplitter(e.Text)
		bounds := make([][2]int, len(words))
		pos := 0
		for i, word := range words {
			at := strings.Index(e.Text[pos:], word)
			if word == "" || at < 0 {
				return nil, nil, fmt.Errorf("example %q: the word splitter returned %q, which is not the next part of the text; span annotations need a splitter that returns pieces of the text, in order", e.Text, word)
			}
			bounds[i] = [2]int{pos + at, pos + at + len(word)}
			pos = bounds[i][1]
		}
		return words, bounds, nil
	}
	if len(e.SourceTokens) == 0 {
		return nil, nil, fmt.Errorf("example %q: span annotations need words, either from the pipeline's word splitter (pipelines.WithWordSplitter) or from the example's tokens", e.Text)
	}
	words := make([]string, len(e.SourceTokens))
	bounds := make([][2]int, len(e.SourceTokens))
	for i, token := range e.SourceTokens {
		start, end, err := byteRange(e.Text, byteOffset, token.Start, token.End)
		if err != nil {
			return nil, nil, fmt.Errorf("token %d (%q): %w", i, token.Text, err)
		}
		if e.Text[start:end] != token.Text {
			return nil, nil, fmt.Errorf("example %q: token %d is %q, but its offsets [%d, %d) cover %q", e.Text, i, token.Text, token.Start, token.End, e.Text[start:end])
		}
		words[i] = token.Text
		bounds[i] = [2]int{start, end}
	}
	return words, bounds, nil
}

// spanTags turns entity spans into one IOB tag per word.
func spanTags(text string, spans []EntitySpan, wordBounds [][2]int, byteOffset []int) ([]string, error) {
	tags := make([]string, len(wordBounds))
	for i := range tags {
		tags[i] = "O"
	}
	sorted := slices.Clone(spans)
	slices.SortFunc(sorted, func(a, b EntitySpan) int { return a.Start - b.Start })
	lastEnd := -1
	for _, span := range sorted {
		if span.Label == "" {
			return nil, fmt.Errorf("example %q: the span [%d, %d) has no label", text, span.Start, span.End)
		}
		start, end, err := byteRange(text, byteOffset, span.Start, span.End)
		if err != nil {
			return nil, fmt.Errorf("example %q, span %q: %w", text, span.Label, err)
		}
		if start < lastEnd {
			return nil, fmt.Errorf("example %q: the span %q at [%d, %d) overlaps the previous span", text, text[start:end], span.Start, span.End)
		}
		first := slices.IndexFunc(wordBounds, func(b [2]int) bool { return b[0] == start })
		last := slices.IndexFunc(wordBounds, func(b [2]int) bool { return b[1] == end })
		if first < 0 || last < first {
			return nil, fmt.Errorf("example %q: the span %q (%s) does not start and end on word boundaries, so it cannot be tagged word by word", text, text[start:end], span.Label)
		}
		tags[first] = "B-" + span.Label
		for w := first + 1; w <= last; w++ {
			tags[w] = "I-" + span.Label
		}
		lastEnd = end
	}
	return tags, nil
}

// characterByteOffsets maps each character index of text to its byte offset; the extra last entry
// is len(text), so an exclusive end offset maps too.
func characterByteOffsets(text string) []int {
	offsets := make([]int, 0, utf8.RuneCountInString(text)+1)
	for i := range text {
		offsets = append(offsets, i)
	}
	return append(offsets, len(text))
}

// byteRange converts a [start, end) character range into a byte range of text.
func byteRange(text string, byteOffset []int, start, end int) (int, int, error) {
	if start < 0 || end <= start || end >= len(byteOffset) {
		return 0, 0, fmt.Errorf("offsets [%d, %d) are outside the %d characters of the text", start, end, len(byteOffset)-1)
	}
	return byteOffset[start], byteOffset[end], nil
}
