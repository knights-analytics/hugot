package backends

import (
	"context"
	"errors"
	"fmt"

	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/util/fileutil"
)

type Tokenizer struct {
	RustTokenizer    *RustTokenizer
	GoTokenizer      *GoTokenizer
	close            func() error
	Runtime          TokenizerRuntime
	MaxAllowedTokens int
}

func (t *Tokenizer) Close() error {
	if t == nil || t.close == nil {
		return nil
	}
	return t.close()
}

// TokenizerRuntime identifies the tokenizer implementation in use.
type TokenizerRuntime string

const (
	TokenizerRuntimeRust TokenizerRuntime = "RUST"
	TokenizerRuntimeGo   TokenizerRuntime = "GO"
)

func LoadTokenizer(ctx context.Context, model *Model, s *options.Options) error {
	pending, err := startTokenizerLoad(ctx, model, s)
	if err != nil {
		return err
	}
	return pending.attach(model)
}

// parsedTokenizer is a tokenizer parsed from tokenizer.json but not yet
// attached to its model: the encode options depend on the model's inputs,
// which are only known once the model backend exists.
type parsedTokenizer struct {
	attach func(model *Model) error
	close  func() error
}

// pendingTokenizer is a tokenizer being parsed in the background. parsed is
// nil once done when the model has no tokenizer.json.
type pendingTokenizer struct {
	done   chan struct{}
	parsed *parsedTokenizer
	err    error
}

// startTokenizerLoad reads the model's tokenizer.json and parses it in the
// background, so that parsing (hundreds of milliseconds for a large
// vocabulary) overlaps the creation of the model backend. The file is read
// before returning: creating an ORT backend changes the working directory,
// and a relative model path must not be resolved while it does.
func startTokenizerLoad(ctx context.Context, model *Model, s *options.Options) (*pendingTokenizer, error) {
	p := &pendingTokenizer{done: make(chan struct{})}
	tokenizerPath := fileutil.PathJoinSafe(model.Path, "tokenizer.json")
	exists, err := fileutil.FileExists(ctx, tokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("error checking for existence of tokenizer.json: %w", err)
	}
	if !exists {
		close(p.done)
		return p, nil
	}
	tokenizerBytes, err := fileutil.ReadFileBytes(ctx, tokenizerPath)
	if err != nil {
		return nil, err
	}
	var parse func([]byte) (*parsedTokenizer, error)
	switch s.Backend {
	case options.BackendORT, options.BackendXLA:
		parse = parseRustTokenizer
	case options.BackendGo:
		parse = parseGoTokenizer
	default:
		return nil, fmt.Errorf("runtime %s not recognized", s.Backend)
	}
	go func() {
		defer close(p.done)
		p.parsed, p.err = parse(tokenizerBytes)
	}()
	return p, nil
}

// attach waits for the parse to finish and attaches the tokenizer to model.
// Without a tokenizer.json it attaches nothing.
func (p *pendingTokenizer) attach(model *Model) error {
	<-p.done
	if p.err != nil {
		return p.err
	}
	if p.parsed == nil {
		return nil
	}
	if err := p.parsed.attach(model); err != nil {
		return errors.Join(err, p.parsed.close())
	}
	return nil
}

// discard waits for the parse to finish and releases the tokenizer, for when
// the model failed to load and the tokenizer will never be attached. A parse
// that failed left nothing to release.
func (p *pendingTokenizer) discard() error {
	<-p.done
	if p.parsed == nil {
		return nil
	}
	return p.parsed.close()
}

func TokenizeInputs(batch *PipelineBatch, tk *Tokenizer, inputs []string) {
	switch tk.Runtime {
	case TokenizerRuntimeRust:
		tokenizeInputsRust(batch, tk, inputs)
	case TokenizerRuntimeGo:
		tokenizeInputsGo(batch, tk, inputs)
	}
}

func TokenizeInputPairs(batch *PipelineBatch, tk *Tokenizer, inputs [][2]string, sepToken string) {
	switch tk.Runtime {
	case TokenizerRuntimeRust:
		tokenizeInputPairsRust(batch, tk, inputs, sepToken)
	case TokenizerRuntimeGo:
		tokenizeInputPairsGo(batch, tk, inputs, sepToken)
	}
}

func patchBertSequenceTokenTypeIDs(batch *PipelineBatch, sepToken string) {
	// Fix token_type_ids for BERT-style models when we manually concatenated the pair as a single sequence.
	// Pattern expected: [CLS] query [SEP] doc [SEP]
	// HF sets token_type_ids=0 up to and including first [SEP], then 1 for remainder (including final [SEP]).
	for index := range batch.Input {
		input := &batch.Input[index]
		// Only adjust if type ids exist and are all zero
		allZero := true
		for _, t := range input.TypeIDs {
			if t != 0 {
				allZero = false
				break
			}
		}
		if !allZero || len(input.TypeIDs) == 0 {
			continue
		}
		// Find first [SEP] token index (skip position 0 which should be [CLS])
		firstSep := -1
		for iTok := 1; iTok < len(input.Tokens); iTok++ {
			if input.Tokens[iTok] == sepToken {
				firstSep = iTok
				break
			}
		}
		if firstSep == -1 || firstSep == len(input.Tokens)-1 { // nothing to split
			continue
		}
		for iTok := firstSep + 1; iTok < len(input.TypeIDs); iTok++ {
			input.TypeIDs[iTok] = 1
		}
	}
}

func AllInputTokens(pipeline *BasePipeline) error {
	switch pipeline.Model.Tokenizer.Runtime {
	case TokenizerRuntimeRust:
		return allInputTokensRust(pipeline)
	case TokenizerRuntimeGo:
		return allInputTokensGo(pipeline)
	}
	return fmt.Errorf("runtime %s not recognized", pipeline.Model.Tokenizer.Runtime)
}

func Decode(tokens []uint32, tokenizer *Tokenizer) (string, error) {
	switch tokenizer.Runtime {
	case TokenizerRuntimeRust:
		return decodeRust(tokens, tokenizer, true), nil
	case TokenizerRuntimeGo:
		return decodeGo(tokens, tokenizer), nil
	}
	return "", fmt.Errorf("runtime %s not recognized", tokenizer.Runtime)
}
