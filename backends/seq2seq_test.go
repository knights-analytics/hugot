package backends

import (
	"context"
	"testing"
)

func TestRunSeq2SeqSessionRequiresGenerativeModel(t *testing.T) {
	_, _, err := RunSeq2SeqSessionOnBatch(context.Background(), &PipelineBatch{}, &BasePipeline{Model: &Model{}}, Seq2SeqOptions{MaxLength: 4})
	if err == nil {
		t.Fatal("expected non-generative model to be rejected")
	}
}

func TestRunSeq2SeqSessionRequiresMaximumLength(t *testing.T) {
	_, _, err := RunSeq2SeqSessionOnBatch(context.Background(), &PipelineBatch{}, &BasePipeline{Model: &Model{IsGenerative: true}}, Seq2SeqOptions{})
	if err == nil {
		t.Fatal("expected invalid maximum length to be rejected")
	}
}
