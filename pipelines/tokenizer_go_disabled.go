//go:build !GO && !ALL

package pipelines

type GoTokenizer struct{}

func loadGoTokenizer(_ []byte, _ *basePipeline) error {
	return nil
}

func tokenizeInputsGo(_ *PipelineBatch, _ *Tokenizer, _ []string) {}

func decodeGo(_ []uint32, _ *Tokenizer) string {
	return ""
}
