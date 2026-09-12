package pipelines

import (
	"errors"
	"testing"

	"github.com/knights-analytics/hugot/backends"
)

func TestCollectMultimodalResponsesDrainsStreamsAndErrors(t *testing.T) {
	stream := make(chan backends.SequenceDelta, 2)
	stream <- backends.SequenceDelta{Sequence: 0, Token: "partial result"}
	stream <- backends.SequenceDelta{Sequence: 2, Token: "ignored"}
	close(stream)

	wantErr := errors.New("generation failed")
	errs := make(chan error, 1)
	errs <- wantErr
	close(errs)

	responses, err := collectMultimodalResponses(stream, errs, 1)
	if !errors.Is(err, wantErr) {
		t.Fatalf("collectMultimodalResponses() error = %v, want %v", err, wantErr)
	}
	if len(responses) != 1 || responses[0] != "partial result" {
		t.Fatalf("responses = %#v, want partial result", responses)
	}
}

func TestMultimodalMessagesBackwardCompatibleOneShot(t *testing.T) {
	got, err := multimodalMessages([]ImageTextPrompt{{ImagePath: "/img/x.png", Prompt: "describe"}}, "default")
	if err != nil {
		t.Fatalf("multimodalMessages() error = %v", err)
	}
	if len(got) != 1 || len(got[0]) != 1 {
		t.Fatalf("expected one batch element with one turn, got %#v", got)
	}
	m := got[0][0]
	if m.Role != "user" || m.Content != "describe" || len(m.ImageURLs) != 1 || m.ImageURLs[0] != "/img/x.png" {
		t.Fatalf("turn = %#v, want one user turn with image", m)
	}
	if m.AudioURLs != nil {
		t.Fatalf("turn.AudioURLs = %v, want nil", m.AudioURLs)
	}
}

func TestMultimodalMessagesMultiTurnHistoryAndRole(t *testing.T) {
	got, err := multimodalMessages([]ImageTextPrompt{{
		ImagePath: "/img/x.png",
		Prompt:    "second turn",
		Role:      "assistant",
		History: []backends.Message{
			{Role: "system", Content: "you are helpful"},
			{Role: "user", Content: "first turn"},
		},
	}}, "default")
	if err != nil {
		t.Fatalf("multimodalMessages() error = %v", err)
	}
	if len(got) != 1 || len(got[0]) != 3 {
		t.Fatalf("expected 1 conversation with 3 turns (system + history + final), got %#v", got)
	}
	want := []struct{ role, content, image string }{
		{"system", "you are helpful", ""},
		{"user", "first turn", ""},
		{"assistant", "second turn", "/img/x.png"},
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

func TestMultimodalMessagesSystemViaHistory(t *testing.T) {
	// Per-conversation system turn: callers express it as History[0] with
	// Role "system" — there is no per-prompt SystemPrompt field, because the
	// pipeline-level SystemPrompt on the pipeline struct is already applied
	// globally by the ORT backend via createORTMessages.
	got, err := multimodalMessages([]ImageTextPrompt{{
		ImagePath: "/img/x.png",
		Prompt:    "now",
		History:   []backends.Message{{Role: "system", Content: "per-conv system"}},
	}}, "default")
	if err != nil {
		t.Fatalf("multimodalMessages() error = %v", err)
	}
	if len(got[0]) != 2 || got[0][0].Role != "system" || got[0][0].Content != "per-conv system" || got[0][1].Role != "user" {
		t.Fatalf("conversation = %#v, want [system, user(final)]", got[0])
	}
}

func TestMultimodalMessagesEmptyImageStillErrors(t *testing.T) {
	if _, err := multimodalMessages([]ImageTextPrompt{{Prompt: "no image"}}, "default"); err == nil {
		t.Fatal("expected error for missing image path, got nil")
	}
}
