//go:build cgo && (ORT || ALL)

package backends

import (
	"context"
	"errors"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/ortgenai"
)

func TestValidateGenerativeEngineOptions(t *testing.T) {
	tests := []struct {
		name    string
		options *options.OrtOptions
		wantErr bool
	}{
		{name: "defaults", options: &options.OrtOptions{}},
		{name: "cuda provider", options: &options.OrtOptions{CudaOptions: map[string]string{}}, wantErr: true},
		{name: "directml provider", options: &options.OrtOptions{DirectMLOptions: new(int)}, wantErr: true},
		{name: "extra provider", options: &options.OrtOptions{ExtraExecutionProviders: []options.ExtraExecutionProvider{{Name: "custom"}}}, wantErr: true},
		{name: "session options", options: &options.OrtOptions{IntraOpNumThreads: new(int)}, wantErr: true},
		{name: "missing options", wantErr: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := validateGenerativeEngineOptions(test.options)
			if (err != nil) != test.wantErr {
				t.Fatalf("validateGenerativeEngineOptions() error = %v, wantErr %t", err, test.wantErr)
			}
		})
	}
}

func TestCreateORTEngineRejectsUnsupportedOptionsBeforeModelLoad(t *testing.T) {
	err := createORTGenerativeSession(context.Background(), &Model{Path: "missing-model"}, &options.Options{
		ORTOptions: &options.OrtOptions{UseEngine: true, CudaOptions: map[string]string{}},
	})
	if err == nil || !strings.Contains(err.Error(), "custom execution providers") {
		t.Fatalf("createORTGenerativeSession() error = %v, want provider configuration error", err)
	}
}

func TestGenerativeEngineRejectsImageBatch(t *testing.T) {
	pipeline := &BasePipeline{
		SessionContext: context.Background(),
		Model: &Model{ORTModel: &ORTModel{
			Generative: &generativeORTProvider{engine: &engineORTOwner{}},
		}},
	}
	batch := NewBatch(1)
	batch.InputValues = [][]ortgenai.Message{{{Role: "user", Content: "describe"}}}
	batch.Images = struct{}{}

	tokens, errs, err := runGenerativeORTSessionOnBatch(context.Background(), batch, pipeline, 8, nil, nil, nil, nil, nil, nil)
	if err == nil || !strings.Contains(err.Error(), "text-only") {
		t.Fatalf("runGenerativeORTSessionOnBatch() error = %v, want text-only image rejection", err)
	}
	if tokens != nil || errs != nil {
		t.Fatal("image rejection unexpectedly returned generation streams")
	}
}

func TestGenerativeEngineRejectsAudioBatch(t *testing.T) {
	pipeline := &BasePipeline{
		SessionContext: context.Background(),
		Model: &Model{ORTModel: &ORTModel{
			Generative: &generativeORTProvider{engine: &engineORTOwner{}},
		}},
	}
	batch := NewBatch(1)
	batch.InputValues = [][]ortgenai.Message{{{Role: "user", Content: "transcribe"}}}
	batch.Audios = struct{}{}

	tokens, errs, err := runGenerativeORTSessionOnBatch(context.Background(), batch, pipeline, 8, nil, nil, nil, nil, nil, nil)
	if err == nil || !strings.Contains(err.Error(), "text-only") {
		t.Fatalf("runGenerativeORTSessionOnBatch() error = %v, want text-only audio rejection", err)
	}
	if tokens != nil || errs != nil {
		t.Fatal("audio rejection unexpectedly returned generation streams")
	}
}

func TestORTSessionGenerationContextFollowsSessionCancellation(t *testing.T) {
	sessionContext, cancelSession := context.WithCancel(context.Background())
	generateContext, cancelGeneration := newORTSessionGenerationContext(context.Background(), sessionContext)
	defer cancelGeneration()

	cancelSession()
	select {
	case <-generateContext.Done():
	case <-time.After(time.Second):
		t.Fatal("generation context did not follow session cancellation")
	}
}

func TestForwardORTSessionDeltaStopsOnCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	stream := make(chan SequenceDelta)
	done := make(chan bool, 1)
	go func() {
		done <- forwardORTSessionDelta(ctx, stream, SequenceDelta{Token: "blocked"})
	}()
	cancel()
	select {
	case forwarded := <-done:
		if forwarded {
			t.Fatal("delta forwarded after cancellation")
		}
	case <-time.After(time.Second):
		t.Fatal("delta forwarding did not stop after cancellation")
	}
}

func TestFindORTEventTurnPreservesSequence(t *testing.T) {
	firstRequest := &ortgenai.Request{}
	secondRequest := &ortgenai.Request{}
	turn := &engineORTTurn{request: secondRequest, id: 27, sequence: 4}
	got, err := findORTEventTurn(map[*ortgenai.Request]*engineORTTurn{
		firstRequest:  {request: firstRequest, id: 13, sequence: 0},
		secondRequest: turn,
	}, ortgenai.Event{Request: secondRequest, TurnID: 27})
	if err != nil {
		t.Fatal(err)
	}
	if got.sequence != 4 {
		t.Fatalf("event mapped to sequence %d, want 4", got.sequence)
	}
	if _, err := findORTEventTurn(map[*ortgenai.Request]*engineORTTurn{secondRequest: turn}, ortgenai.Event{Request: firstRequest, TurnID: 27}); err == nil {
		t.Fatal("expected unknown request event to fail")
	}
	if _, err := findORTEventTurn(map[*ortgenai.Request]*engineORTTurn{secondRequest: turn}, ortgenai.Event{Request: secondRequest, TurnID: 28}); err == nil {
		t.Fatal("expected mismatched turn event to fail")
	}
}

func TestEngineORTOwnerDoCancellationAndClose(t *testing.T) {
	owner := &engineORTOwner{commands: make(chan engineORTCommand), done: make(chan struct{})}
	go func() {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		defer close(owner.done)
		for command := range owner.commands {
			runEngineORTCommand(&engineORTState{}, command)
		}
	}()

	var called atomic.Bool
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if err := owner.do(canceled, func(*engineORTState) error {
		called.Store(true)
		return nil
	}); !errors.Is(err, context.Canceled) {
		t.Fatalf("do() error = %v, want context cancellation", err)
	}
	if called.Load() {
		t.Fatal("canceled command was executed")
	}

	started := make(chan struct{})
	release := make(chan struct{})
	commandDone := make(chan error, 1)
	go func() {
		commandDone <- owner.do(context.Background(), func(*engineORTState) error {
			close(started)
			<-release
			return nil
		})
	}()
	<-started
	closeDone := make(chan struct{})
	go func() {
		_ = owner.Close()
		close(closeDone)
	}()
	time.Sleep(10 * time.Millisecond)
	select {
	case <-closeDone:
		t.Fatal("Close returned before an accepted owner command completed")
	default:
	}
	close(release)
	if err := <-commandDone; err != nil {
		t.Fatal(err)
	}
	select {
	case <-closeDone:
	case <-time.After(time.Second):
		t.Fatal("Close did not finish after the active command completed")
	}
	if err := owner.do(context.Background(), func(*engineORTState) error { return nil }); err == nil {
		t.Fatal("expected command on closed owner to fail")
	}
	if err := owner.Close(); err != nil {
		t.Fatalf("second Close() error = %v", err)
	}
}

func TestEngineGenerateEmptyBatch(t *testing.T) {
	batch := NewBatch(0)
	var destroyed atomic.Bool
	batch.DestroyInputs = func() error {
		destroyed.Store(true)
		return nil
	}
	tokens, errs, err := (&engineORTOwner{}).generate(context.Background(), nil, batch, nil, 1, nil, nil, nil, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := <-tokens; ok {
		t.Fatal("expected empty token stream to be closed")
	}
	if _, ok := <-errs; ok {
		t.Fatal("expected empty error stream to be closed")
	}
	if !destroyed.Load() {
		t.Fatal("empty batch inputs were not destroyed")
	}
}

func TestMakeEngineORTTurnSettings(t *testing.T) {
	temperature := 0.7
	topP := 0.8
	seed := 42
	settings, err := makeEngineORTTurnSettings(128, []string{"STOP", ""}, &temperature, &topP, &seed, nil)
	if err != nil {
		t.Fatal(err)
	}
	if settings.maxGeneratedTokens != 128 || !settings.doSample || settings.temperature == nil || *settings.temperature != 0.7 || settings.topP == nil || *settings.topP != 0.8 || settings.seed == nil || *settings.seed != 42 {
		t.Fatalf("unexpected engine turn settings: %#v", settings)
	}
	if len(settings.stopStrings) != 1 || settings.stopStrings[0] != "STOP" {
		t.Fatalf("unexpected stop strings: %#v", settings.stopStrings)
	}

	greedy, err := makeEngineORTTurnSettings(4, nil, nil, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if greedy.doSample {
		t.Fatal("expected default engine generation to be greedy")
	}

	negativeSeed := -1
	if _, err := makeEngineORTTurnSettings(4, nil, nil, nil, &negativeSeed, nil); err == nil {
		t.Fatal("expected negative seed to be rejected")
	}
	if _, err := makeEngineORTTurnSettings(4, nil, nil, nil, nil, &Guidance{EnableFFTokens: true}); err == nil {
		t.Fatal("expected force-forward guidance to be rejected")
	}
}

func TestMarshalORTEngineTemplateInput(t *testing.T) {
	messageJSON, toolsJSON, err := marshalORTEngineTemplateInput(
		[]ortgenai.Message{{Role: "user", Content: "hello"}},
		[]string{`{"name":"lookup"}`},
	)
	if err != nil {
		t.Fatal(err)
	}
	if messageJSON != `[{"role":"user","content":"hello"}]` {
		t.Fatalf("unexpected message JSON: %s", messageJSON)
	}
	if toolsJSON != `[{"name":"lookup"}]` {
		t.Fatalf("unexpected tools JSON: %s", toolsJSON)
	}
	if _, _, err := marshalORTEngineTemplateInput(nil, []string{"not-json"}); err == nil {
		t.Fatal("expected invalid tool JSON to be rejected")
	}
}

func TestORTSessionStopFilterExcludesSplitStopString(t *testing.T) {
	filter := newORTSessionStopFilter([]string{"STOP"})
	var output strings.Builder
	for _, token := range []string{"hello S", "TO", "P and more"} {
		text, stopped := filter.push(token)
		output.WriteString(text)
		if stopped {
			break
		}
	}
	if got, want := output.String(), "hello "; got != want {
		t.Fatalf("filtered output = %q, want %q", got, want)
	}
}

func TestORTSessionStopFilterHandlesContainedAndPartialStops(t *testing.T) {
	t.Run("stop in one delta", func(t *testing.T) {
		filter := newORTSessionStopFilter([]string{"<end>", "STOP"})
		got, stopped := filter.push("answer<end>ignored")
		if !stopped || got != "answer" {
			t.Fatalf("push() = (%q, %t), want (%q, true)", got, stopped, "answer")
		}
	})

	t.Run("unfinished prefix is flushed", func(t *testing.T) {
		filter := newORTSessionStopFilter([]string{"STOP"})
		first, stopped := filter.push("answer S")
		if stopped || first != "answer " {
			t.Fatalf("first push() = (%q, %t), want (%q, false)", first, stopped, "answer ")
		}
		last := filter.flush()
		if last != "S" {
			t.Fatalf("flush() = %q, want %q", last, "S")
		}
	})

	t.Run("no stops forwards immediately", func(t *testing.T) {
		filter := newORTSessionStopFilter(nil)
		got, stopped := filter.push("text")
		if stopped || got != "text" {
			t.Fatalf("push() = (%q, %t), want (%q, false)", got, stopped, "text")
		}
	})
}
