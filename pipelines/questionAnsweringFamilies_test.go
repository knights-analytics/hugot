package pipelines

import (
	"context"
	"image"
	"os"
	"strings"
	"testing"

	"github.com/knights-analytics/hugot/backends"
)

func TestQuestionAnsweringFamiliesValidateRequiresGenerativeModel(t *testing.T) {
	model := &backends.Model{}
	config := VisualQuestionAnsweringConfig{}
	pipeline, err := NewVisualQuestionAnsweringPipeline(context.Background(), config, model)
	if err == nil || pipeline != nil {
		t.Fatal("expected visual QA validation to reject a non-generative model")
	}
}

func TestQuestionAnsweringFamiliesValidateNilModel(t *testing.T) {
	if _, err := NewTableQuestionAnsweringPipeline(context.Background(), TableQuestionAnsweringConfig{}, nil); err == nil {
		t.Fatal("expected table QA validation to reject a nil model")
	}
}

func TestQuestionAnsweringFamiliesRunInputValidation(t *testing.T) {
	pipeline := &VisualQuestionAnsweringPipeline{}
	if _, err := pipeline.Run(context.Background(), []string{"image.png"}); err == nil {
		t.Fatal("expected an image/question pair validation error")
	}
	if _, err := (&TableQuestionAnsweringPipeline{}).Run(context.Background(), []string{"not-json"}); err == nil {
		t.Fatal("expected invalid table JSON to fail")
	}
}

func TestQuestionAnsweringFamiliesGenerativeStatusOnNilPipelines(t *testing.T) {
	var visual *VisualQuestionAnsweringPipeline
	var document *DocumentQuestionAnsweringPipeline
	var table *TableQuestionAnsweringPipeline
	var imageToText *ImageToTextPipeline
	var imageTextToText *ImageTextToTextPipeline

	if !visual.IsGenerative() || document.IsGenerative() || table.IsGenerative() ||
		!imageToText.IsGenerative() || !imageTextToText.IsGenerative() {
		t.Fatal("only conversational VQA and image generation use the GenAI model loader")
	}
}

func TestVisualQuestionAnsweringMessagesOneShotBackwardCompat(t *testing.T) {
	got, err := visualQuestionAnsweringMessages([]VisualQuestionAnsweringInput{{ImagePath: "/img/x.png", Question: "what is this?"}})
	if err != nil {
		t.Fatalf("visualQuestionAnsweringMessages() error = %v", err)
	}
	if len(got) != 1 || len(got[0]) != 1 {
		t.Fatalf("want one batch element with one turn, got %#v", got)
	}
	m := got[0][0]
	if m.Role != "user" || m.Content != "what is this?" || len(m.ImageURLs) != 1 || m.ImageURLs[0] != "/img/x.png" {
		t.Fatalf("turn = %#v, want one user turn with image", m)
	}
}

func TestVisualQuestionAnsweringMessagesMultiTurnHistoryAndRole(t *testing.T) {
	got, err := visualQuestionAnsweringMessages([]VisualQuestionAnsweringInput{{
		ImagePath: "/img/x.png",
		Question:  "second question",
		Role:      "assistant",
		History: []backends.Message{
			{Role: "system", Content: "you are a careful reader"},
			{Role: "user", Content: "first question"},
		},
	}})
	if err != nil {
		t.Fatalf("visualQuestionAnsweringMessages() error = %v", err)
	}
	if len(got) != 1 || len(got[0]) != 3 {
		t.Fatalf("want 3 turns per conversation (system + user + final), got %#v", got)
	}
	want := []struct{ role, content, image string }{
		{"system", "you are a careful reader", ""},
		{"user", "first question", ""},
		{"assistant", "second question", "/img/x.png"},
	}
	for i, w := range want {
		m := got[0][i]
		if m.Role != w.role || m.Content != w.content {
			t.Fatalf("turn %d = %#v, want role=%q content=%q", i, m, w.role, w.content)
		}
		if w.image == "" {
			if len(m.ImageURLs) != 0 {
				t.Fatalf("turn %d = %#v, want no image", i, m)
			}
		} else if len(m.ImageURLs) != 1 || m.ImageURLs[0] != w.image {
			t.Fatalf("turn %d = %#v, want image [%s]", i, m, w.image)
		}
	}
}

func TestVisualQuestionAnsweringMessagesRequiresImageAndQuestion(t *testing.T) {
	if _, err := visualQuestionAnsweringMessages([]VisualQuestionAnsweringInput{{ImagePath: ""}}); err == nil {
		t.Fatal("expected missing image to error")
	}
	if _, err := visualQuestionAnsweringMessages([]VisualQuestionAnsweringInput{{Question: "q"}}); err == nil {
		t.Fatal("expected missing question to error")
	}
	if _, err := visualQuestionAnsweringMessages([]VisualQuestionAnsweringInput{{ImagePath: " "}}); err == nil {
		t.Fatal("expected blank image to error")
	}
}

func TestTableQuestionAnsweringMessagesOneShotAndQueryFallback(t *testing.T) {
	question := "sum of column A?"
	gotQ, err := tableQuestionAnsweringMessages([]TableQuestionAnsweringInput{
		{Table: [][]string{{"A", "B"}, {"1", "2"}}, Question: question},
		{Table: [][]string{{"A", "B"}, {"1", "2"}}, Query: question}, // Query fallback
	})
	if err != nil {
		t.Fatalf("tableQuestionAnsweringMessages() error = %v", err)
	}
	if len(gotQ) != 2 {
		t.Fatalf("want 2 conversations, got %d", len(gotQ))
	}
	for i, conv := range gotQ {
		if len(conv) != 1 {
			t.Fatalf("conversation %d: want 1 turn, got %#v", i, conv)
		}
		last := conv[0]
		if last.Role != "user" || last.ImageURLs != nil || !strings.HasPrefix(last.Content, "Answer the question using this table:") {
			t.Fatalf("turn %d = %#v, want user turn with table content", i, last)
		}
		if !strings.Contains(last.Content, question) {
			t.Fatalf("turn %d content = %q, want to contain %q", i, last.Content, question)
		}
	}
}

func TestTableQuestionAnsweringMessagesMultiTurn(t *testing.T) {
	got, err := tableQuestionAnsweringMessages([]TableQuestionAnsweringInput{{
		Table:    [][]string{{"A"}, {"1"}},
		Question: "is A 1?",
		Role:     "assistant",
		History: []backends.Message{
			{Role: "user", Content: "first: what is A?"},
			{Role: "assistant", Content: "A is 1."},
		},
	}})
	if err != nil {
		t.Fatalf("tableQuestionAnsweringMessages() error = %v", err)
	}
	if len(got) != 1 || len(got[0]) != 3 {
		t.Fatalf("want 1 conversation with 3 turns, got %#v", got)
	}
	if got[0][0].Role != "user" || got[0][1].Role != "assistant" || got[0][2].Role != "assistant" {
		t.Fatalf("roles = %v, want [user, assistant, assistant]", []string{got[0][0].Role, got[0][1].Role, got[0][2].Role})
	}
}

// docQAAsVQAInput mirrors the conversion performed by
// (*DocumentQuestionAnsweringPipeline).RunPipeline, which delegates into VQA.
// Factoring it out lets the test cover the DocumentPath->ImagePath fallback
// and Role/History forwarding without instantiating a real model.
func docQAAsVQAInput(input DocumentQuestionAnsweringInput) VisualQuestionAnsweringInput {
	path := input.DocumentPath
	if path == "" {
		path = input.ImagePath
	}
	return VisualQuestionAnsweringInput{ImagePath: path, Question: input.Question, Role: input.Role, History: input.History}
}

func TestDocumentQuestionAnsweringDelegatesToVQA(t *testing.T) {
	input := DocumentQuestionAnsweringInput{
		Question: "what does this doc say?",
		Role:     "user",
		History:  []backends.Message{{Role: "system", Content: "you are accurate"}},
	}

	// DocumentPath preferred.
	input.DocumentPath = "/doc/1.png"
	input.ImagePath = "/img/fallback.png"
	v := docQAAsVQAInput(input)
	if v.ImagePath != "/doc/1.png" || v.Question != input.Question || v.Role != "user" || len(v.History) != 1 {
		t.Fatalf("delegation = %#v, want DocumentPath + forwarded Role/History", v)
	}

	// ImagePath used as fallback when DocumentPath is empty.
	input.DocumentPath = ""
	v = docQAAsVQAInput(input)
	if v.ImagePath != "/img/fallback.png" {
		t.Fatalf("delegation.ImagePath = %q, want ImagePath fallback", v.ImagePath)
	}

	// The mapped input feeds VQA's builder as a full conversation.
	got, err := visualQuestionAnsweringMessages([]VisualQuestionAnsweringInput{v})
	if err != nil {
		t.Fatalf("visualQuestionAnsweringMessages() error = %v", err)
	}
	last := got[0][len(got[0])-1]
	if last.Role != "user" || last.ImageURLs[0] != "/img/fallback.png" || len(got[0]) != 2 {
		t.Fatalf("delegation result = %#v, want [system, user with image]", got[0])
	}
}

func TestVisualQuestionAnsweringRunWithImagesAdaptsInMemoryAndPath(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 4, 4))
	values, cleanup, err := visualQuestionAnsweringRunWithImages([]backends.ImageTextInput{
		{Image: img, Text: "q1"},
		{ImagePath: "/img/c.jpg", Text: "q2"},
	})
	if err != nil {
		t.Fatalf("adapter error = %v", err)
	}
	defer cleanup()
	if len(values) != 2 {
		t.Fatalf("want 2 inputs, got %d", len(values))
	}
	if values[0].Question != "q1" || values[1].Question != "q2" {
		t.Fatalf("questions = %q %q, want q1 q2", values[0].Question, values[1].Question)
	}
	if values[1].ImagePath != "/img/c.jpg" {
		t.Fatalf("ImagePath passthrough = %q, want /img/c.jpg", values[1].ImagePath)
	}
	if values[0].ImagePath == "" {
		t.Fatal("in-memory image produced empty ImagePath")
	}
	if _, statErr := os.Stat(values[0].ImagePath); statErr != nil {
		t.Fatalf("in-memory image temp file missing after adaptation: %v", statErr)
	}
	cleanup()
	if _, statErr := os.Stat(values[0].ImagePath); !os.IsNotExist(statErr) {
		t.Fatalf("expected temp file removed after cleanup, stat=%v", statErr)
	}
}

func TestVisualQuestionAnsweringRunWithImagesRejectsInvalidInputs(t *testing.T) {
	if _, _, err := visualQuestionAnsweringRunWithImages([]backends.ImageTextInput{{}}); err == nil {
		t.Fatal("expected missing image/text to fail validation")
	}
	img := image.NewRGBA(image.Rect(0, 0, 1, 1))
	if _, _, err := visualQuestionAnsweringRunWithImages([]backends.ImageTextInput{{Image: img, Text: "  "}}); err == nil {
		t.Fatal("expected blank text to fail validation")
	}
}

func TestDocumentQuestionAnsweringRunWithImagesFillsDocumentPath(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 2, 2))
	values, cleanup, err := documentQuestionAnsweringRunWithImages([]backends.ImageTextInput{
		{Image: img, Text: "q"},
		{ImagePath: "/img/c.jpg", Text: "q2"},
	})
	if err != nil {
		t.Fatalf("adapter error = %v", err)
	}
	defer cleanup()
	if len(values) != 2 {
		t.Fatalf("want 2 inputs, got %d", len(values))
	}
	if values[0].DocumentPath == "" || values[1].DocumentPath != "/img/c.jpg" {
		t.Fatalf("DocumentPath mapping = %q %q", values[0].DocumentPath, values[1].DocumentPath)
	}
}
