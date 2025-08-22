package pipelineBackends

import (
	"fmt"

	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/util"
)

type Tokenizer struct {
	Runtime          string
	RustTokenizer    *RustTokenizer
	GoTokenizer      *GoTokenizer
	TokenizerTimings *timings
	MaxAllowedTokens int
	Destroy          func() error
}

func LoadTokenizer(model *Model, s *options.Options) error {
	if exists, err := util.FileExists(util.PathJoinSafe(model.Path, "tokenizer.json")); err == nil {
		if exists {
			tokenizerBytes, err := util.ReadFileBytes(util.PathJoinSafe(model.Path, "tokenizer.json"))
			if err != nil {
				return err
			}

			switch s.Backend {
			case "ORT", "XLA":
				return loadRustTokenizer(tokenizerBytes, model)
			case "GO":
				return loadGoTokenizer(tokenizerBytes, model)
			default:
				return fmt.Errorf("runtime %s not recognized", s.Backend)
			}
		}
	} else {
		return fmt.Errorf("error checking for existence of tokenizer.json: %w", err)
	}
	return nil
}

func TokenizeInputs(batch *PipelineBatch, tk *Tokenizer, inputs []string) {
	switch tk.Runtime {
	case "RUST":
		tokenizeInputsRust(batch, tk, inputs)
	case "GO":
		tokenizeInputsGo(batch, tk, inputs)
	}
}

func AllInputTokens(pipeline *BasePipeline) {
	if pipeline.Model.Tokenizer.Runtime == "RUST" {
		allInputTokensRust(pipeline)
	}
}

func Decode(tokens []uint32, tokenizer *Tokenizer, skipSpecialTokens bool) (string, error) {
	switch tokenizer.Runtime {
	case "RUST":
		return decodeRust(tokens, tokenizer, skipSpecialTokens), nil
	case "GO":
		return decodeGo(tokens, tokenizer, skipSpecialTokens), nil
	}
	return "", fmt.Errorf("runtime %s not recognized", tokenizer.Runtime)
}
