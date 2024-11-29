package pipelines

import (
	"fmt"

	"github.com/knights-analytics/hugot/util"
)

type Tokenizer struct {
	Runtime          string
	RustTokenizer    *RustTokenizer
	TokenizerTimings *timings
	Destroy          func() error
}

func loadTokenizer(pipeline *BasePipeline) error {
	tokenizerBytes, err := util.ReadFileBytes(util.PathJoinSafe(pipeline.ModelPath, "tokenizer.json"))
	if err != nil {
		return err
	}

	switch pipeline.Runtime {
	case "ORT", "XLA":
		return loadRustTokenizer(tokenizerBytes, pipeline)
	default:
		return fmt.Errorf("runtime %s not recognized", pipeline.Runtime)
	}
}

func TokenizeInputs(batch *PipelineBatch, tk *Tokenizer, inputs []string) {
	switch tk.Runtime {
	case "RUST":
		tokenizeInputsRust(batch, tk, inputs)
	}
}

func AllInputTokens(pipeline *BasePipeline) {
	if pipeline.Tokenizer.Runtime == "RUST" {
		allInputTokensRust(pipeline)
	}
}

func Decode(tokens []uint32, tokenizer *Tokenizer) string {
	switch tokenizer.Runtime {
	case "RUST":
		return decodeRust(tokens, tokenizer)
	}
	return ""
}
