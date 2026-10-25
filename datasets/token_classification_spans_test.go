package datasets

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/knights-analytics/hugot/pipelines"
)

// The accented characters before the spans make character offsets (as Python and Prodigy count
// them) differ from Go's byte offsets, so a conversion error would shift every span.
const prodigyLine = `{"text": "Café owner Anna Weber visited Zürich",
 "spans": [{"start": 30, "end": 36, "label": "LOC", "token_start": 5, "token_end": 5},
           {"start": 11, "end": 21, "label": "PER", "token_start": 2, "token_end": 3}],
 "tokens": [{"text": "Café", "start": 0, "end": 4, "id": 0}, {"text": "owner", "start": 5, "end": 10, "id": 1},
            {"text": "Anna", "start": 11, "end": 15, "id": 2}, {"text": "Weber", "start": 16, "end": 21, "id": 3},
            {"text": "visited", "start": 22, "end": 29, "id": 4}, {"text": "Zürich", "start": 30, "end": 36, "id": 5}],
 "answer": "accept", "_input_hash": 123}`

var (
	spanWantWords = []string{"Café", "owner", "Anna", "Weber", "visited", "Zürich"}
	spanWantTags  = []string{"O", "O", "B-PER", "I-PER", "O", "B-LOC"}
)

func parseExample(t *testing.T, line string) TokenClassificationExample {
	t.Helper()
	var example TokenClassificationExample
	if err := json.Unmarshal([]byte(strings.ReplaceAll(line, "\n", "")), &example); err != nil {
		t.Fatal(err)
	}
	return example
}

func TestSpansFromSourceTokens(t *testing.T) {
	example := parseExample(t, prodigyLine)
	if !example.IsSpanFormat() || len(example.SourceTokens) != 6 || example.Tokens != nil {
		t.Fatalf("parsed %+v, want a span example with 6 source tokens", example)
	}
	words, tags, err := example.Resolve(&pipelines.TokenClassificationPipeline{})
	if err != nil || !slices.Equal(words, spanWantWords) || !slices.Equal(tags, spanWantTags) {
		t.Fatalf("got %q %q, %v; want %q %q", words, tags, err, spanWantWords, spanWantTags)
	}
}

func TestSpansFromWordSplitter(t *testing.T) {
	example := parseExample(t, prodigyLine)
	example.SourceTokens = nil // the splitter, not the annotation tool, decides the words
	words, tags, err := example.Resolve(&pipelines.TokenClassificationPipeline{WordSplitter: strings.Fields})
	if err != nil || !slices.Equal(words, spanWantWords) || !slices.Equal(tags, spanWantTags) {
		t.Fatalf("got %q %q, %v; want %q %q", words, tags, err, spanWantWords, spanWantTags)
	}
	// A splitter that splits inside an entity still tags it, as B- then I-.
	example = TokenClassificationExample{Text: "Seat 14C Row7", Spans: []EntitySpan{{Label: "ROW", Start: 9, End: 13}}}
	words, tags, err = example.Resolve(&pipelines.TokenClassificationPipeline{WordSplitter: splitLetterDigit})
	if err != nil || !slices.Equal(words, []string{"Seat", "14C", "Row", "7"}) || !slices.Equal(tags, []string{"O", "O", "B-ROW", "I-ROW"}) {
		t.Fatalf("got %q %q, %v", words, tags, err)
	}
}

func TestSpanErrors(t *testing.T) {
	plain := &pipelines.TokenClassificationPipeline{WordSplitter: strings.Fields}
	text := "Anna Weber visited Zürich"
	cases := map[string]struct {
		example TokenClassificationExample
		want    string
	}{
		"mid-word start": {TokenClassificationExample{Text: text, Spans: []EntitySpan{{Label: "PER", Start: 1, End: 4}}}, "word boundaries"},
		"mid-word end":   {TokenClassificationExample{Text: text, Spans: []EntitySpan{{Label: "PER", Start: 0, End: 7}}}, "word boundaries"},
		"overlap": {TokenClassificationExample{Text: text, Spans: []EntitySpan{
			{Label: "PER", Start: 0, End: 10}, {Label: "LAST", Start: 5, End: 10}}}, "overlaps"},
		"out of range":      {TokenClassificationExample{Text: text, Spans: []EntitySpan{{Label: "LOC", Start: 19, End: 99}}}, "outside"},
		"no label":          {TokenClassificationExample{Text: text, Spans: []EntitySpan{{Start: 0, End: 4}}}, "no label"},
		"spans and tags":    {TokenClassificationExample{Text: text, Spans: []EntitySpan{}, Tags: []string{"O"}}, "both spans and tags"},
		"rejected":          {TokenClassificationExample{Text: text, Spans: []EntitySpan{}, Answer: "reject"}, "only accepted"},
		"splitter rewrites": {TokenClassificationExample{Text: text, Spans: []EntitySpan{}}, "not the next part"},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			pipeline := plain
			if name == "splitter rewrites" {
				pipeline = &pipelines.TokenClassificationPipeline{WordSplitter: func(s string) []string { return strings.Fields(strings.ToLower(s)) }}
			}
			if _, _, err := tc.example.Resolve(pipeline); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("got %v, want an error containing %q", err, tc.want)
			}
		})
	}
	// Without a splitter, the source tokens must match the text at their offsets.
	example := parseExample(t, prodigyLine)
	example.SourceTokens[5].Text = "Zurich"
	if _, _, err := example.Resolve(&pipelines.TokenClassificationPipeline{}); err == nil || !strings.Contains(err.Error(), "cover") {
		t.Fatalf("got %v, want a token/offset mismatch error", err)
	}
	example.SourceTokens = nil
	if _, _, err := example.Resolve(&pipelines.TokenClassificationPipeline{}); err == nil || !strings.Contains(err.Error(), "need words") {
		t.Fatalf("got %v, want an error for spans with no way to split the text", err)
	}
}

func TestSpanExampleWithoutEntities(t *testing.T) {
	example := parseExample(t, `{"text": "nothing here", "spans": [], "answer": "accept"}`)
	_, tags, err := example.Resolve(&pipelines.TokenClassificationPipeline{WordSplitter: strings.Fields})
	if err != nil || !slices.Equal(tags, []string{"O", "O"}) {
		t.Fatalf("got %q, %v; want all O", tags, err)
	}
}

// Both formats can share one file, and examples not answered "accept" are skipped on read.
func TestMixedFormatFile(t *testing.T) {
	lines := []string{
		`{"tokens": ["EU", "rejects", "German", "call"], "tags": ["B-ORG", "O", "B-MISC", "O"]}`,
		strings.ReplaceAll(prodigyLine, "\n", ""),
		`{"text": "Anna visited Zürich", "spans": [{"start": 0, "end": 4, "label": "PER"}], "answer": "reject"}`,
		`{"text": "Anna visited Zürich", "spans": [], "answer": "ignore"}`,
	}
	path := filepath.Join(t.TempDir(), "mixed.jsonl")
	if err := os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	d, err := NewTokenClassificationDataset(t.Context(), path, 2, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	examples, err := d.Examples()
	if err != nil {
		t.Fatal(err)
	}
	if len(examples) != 2 || examples[0].IsSpanFormat() || !examples[1].IsSpanFormat() {
		t.Fatalf("got %d examples (%+v), want the word-level and the accepted span example", len(examples), examples)
	}
	if !slices.Equal(examples[0].Tokens, []string{"EU", "rejects", "German", "call"}) {
		t.Fatalf("word-level tokens parsed as %q", examples[0].Tokens)
	}

	inMemory, err := NewInMemoryTokenClassificationDataset([]TokenClassificationExample{
		{Tokens: []string{"a"}, Tags: []string{"O"}},
		{Text: "b", Spans: []EntitySpan{}, Answer: "reject"},
	}, 2, nil)
	if err != nil {
		t.Fatal(err)
	}
	if examples, err = inMemory.Examples(); err != nil || len(examples) != 1 {
		t.Fatalf("got %d in-memory examples, %v; want the rejected one dropped", len(examples), err)
	}
}
