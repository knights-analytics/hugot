package backends

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/util/fileutil"
)

const testTokenizerModel = "../models/KnightsAnalytics_all-MiniLM-L6-v2"

// tokenizerModelDir copies the test model's tokenizer.json into a fresh
// directory, so a test can move or remove it.
func tokenizerModelDir(t *testing.T) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(testTokenizerModel, "tokenizer.json"))
	if err != nil {
		t.Skipf("test model not downloaded: %v", err)
	}
	dir := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(dir, "tokenizer.json"), b, 0o600)) //nolint:gosec // writes into the test's own temp dir
	return dir
}

func tokenizerTestModel(path string) *Model {
	return &Model{ModelMetadata: ModelMetadata{
		Path:                  path,
		MaxPositionEmbeddings: 512,
		InputsMeta:            []InputOutputInfo{{Name: "input_ids"}, {Name: "attention_mask"}},
	}}
}

func goOptions() *options.Options { return &options.Options{Backend: options.BackendGo} }

func testCtx() context.Context { return fileutil.WithFileSystem(context.Background(), nil) }

func TestLoadTokenizerAttachesAfterParsing(t *testing.T) {
	model := tokenizerTestModel(tokenizerModelDir(t))
	require.NoError(t, LoadTokenizer(testCtx(), model, goOptions()))
	require.NotNil(t, model.Tokenizer)
	require.Equal(t, TokenizerRuntimeGo, model.Tokenizer.Runtime)
	require.True(t, model.Tokenizer.GoTokenizer.AttentionMask)
	require.Equal(t, 512, model.Tokenizer.MaxAllowedTokens)
}

func TestLoadTokenizerWithoutTokenizerFile(t *testing.T) {
	model := tokenizerTestModel(t.TempDir())
	require.NoError(t, LoadTokenizer(testCtx(), model, goOptions()))
	require.Nil(t, model.Tokenizer)
}

// The file is read before startTokenizerLoad returns, so the model backend may
// change the working directory (as the ORT backend does) while the tokenizer
// is still being parsed.
func TestStartTokenizerLoadReadsTheFileBeforeReturning(t *testing.T) {
	dir := tokenizerModelDir(t)
	model := tokenizerTestModel(dir)
	pending, err := startTokenizerLoad(testCtx(), model, goOptions())
	require.NoError(t, err)
	require.NoError(t, os.Remove(filepath.Join(dir, "tokenizer.json")))
	require.NoError(t, pending.attach(model))
	require.NotNil(t, model.Tokenizer)
}

// The model's inputs decide the encode options, and they are only known once
// the backend exists: attach must read them then, not when parsing started.
func TestTokenizerOptionsComeFromInputsKnownAtAttach(t *testing.T) {
	model := tokenizerTestModel(tokenizerModelDir(t))
	model.InputsMeta = nil
	pending, err := startTokenizerLoad(testCtx(), model, goOptions())
	require.NoError(t, err)
	model.InputsMeta = []InputOutputInfo{{Name: "input_ids"}, {Name: "token_type_ids"}}
	require.NoError(t, pending.attach(model))
	require.True(t, model.Tokenizer.GoTokenizer.TypeIDs)
	require.False(t, model.Tokenizer.GoTokenizer.AttentionMask)
}

func TestTokenizerParseErrorSurfacesAtAttach(t *testing.T) {
	dir := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte("{not json"), 0o600))
	model := tokenizerTestModel(dir)
	pending, err := startTokenizerLoad(testCtx(), model, goOptions())
	require.NoError(t, err)
	require.Error(t, pending.attach(model))
	require.Nil(t, model.Tokenizer)
}

func TestTokenizerAttachErrorFromUnknownInput(t *testing.T) {
	model := tokenizerTestModel(tokenizerModelDir(t))
	model.InputsMeta = []InputOutputInfo{{Name: "something_else"}}
	require.Error(t, LoadTokenizer(testCtx(), model, goOptions()))
	require.Nil(t, model.Tokenizer)
}

func TestTokenizerDiscard(t *testing.T) {
	model := tokenizerTestModel(tokenizerModelDir(t))
	pending, err := startTokenizerLoad(testCtx(), model, goOptions())
	require.NoError(t, err)
	require.NoError(t, pending.discard())
	require.Nil(t, model.Tokenizer)

	none, err := startTokenizerLoad(testCtx(), tokenizerTestModel(t.TempDir()), goOptions())
	require.NoError(t, err)
	require.NoError(t, none.discard(), "no tokenizer.json, nothing to release")

	dir := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte("{not json"), 0o600))
	failed, err := startTokenizerLoad(testCtx(), tokenizerTestModel(dir), goOptions())
	require.NoError(t, err)
	require.NoError(t, failed.discard(), "a tokenizer that failed to parse has nothing to release")
}

func TestStartTokenizerLoadUnknownBackend(t *testing.T) {
	model := tokenizerTestModel(tokenizerModelDir(t))
	_, err := startTokenizerLoad(testCtx(), model, &options.Options{Backend: "nope"})
	require.ErrorContains(t, err, "runtime nope not recognized")
}

func TestStartTokenizerLoadUnreadableFile(t *testing.T) {
	dir := t.TempDir()
	require.NoError(t, os.Mkdir(filepath.Join(dir, "tokenizer.json"), 0o700))
	_, err := startTokenizerLoad(testCtx(), tokenizerTestModel(dir), goOptions())
	require.Error(t, err)
}
