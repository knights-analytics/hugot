package pipelines

import (
	"fmt"

	"github.com/knights-analytics/hugot/util"
)

type Tokenizer struct {
	Runtime          string
	RustTokenizer    *RustTokenizer
	GoTokenizer      *GoTokenizer
	TokenizerTimings *timings
	Destroy          func() error
}

func loadTokenizer(pipeline *basePipeline) error {
	tokenizerBytes, err := util.ReadFileBytes(util.PathJoinSafe(pipeline.ModelPath, "tokenizer.json"))
	if err != nil {
		return err
	}

	switch pipeline.Runtime {
	case "GO":
		return loadGoTokenizer(tokenizerBytes, pipeline)
	case "ORT", "XLA":
		return loadRustTokenizer(tokenizerBytes, pipeline)
	default:
		return fmt.Errorf("runtime %s not recognized", pipeline.Runtime)
	}
}

func tokenizeInputs(batch *PipelineBatch, tk *Tokenizer, inputs []string) {
	switch tk.Runtime {
	case "GO":
		tokenizeInputsGo(batch, tk, inputs)
	case "RUST":
		tokenizeInputsRust(batch, tk, inputs)
	}
}

func allInputTokens(pipeline *basePipeline) {
	if pipeline.Tokenizer.Runtime == "RUST" {
		allInputTokensRust(pipeline)
	}
}

func decode(tokens []uint32, tokenizer *Tokenizer) string {
	switch tokenizer.Runtime {
	case "GO":
		return decodeGo(tokens, tokenizer)
	case "RUST":
		return decodeRust(tokens, tokenizer)
	}
	return ""
}
