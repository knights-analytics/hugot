//go:build NOORT && !XLA

package pipelineBackends

type RustTokenizer struct {
}

func loadRustTokenizer(_ []byte, _ *Model) error {
	return nil
}

func tokenizeInputsRust(_ *PipelineBatch, _ *Tokenizer, _ []string) {}

func decodeRust(_ []uint32, _ *Tokenizer) string {
	return ""
}

func allInputTokensRust(_ *BasePipeline) {}
