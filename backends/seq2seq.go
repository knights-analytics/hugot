package backends

import (
	"context"
	"errors"
)

// Seq2SeqOptions describes the task-independent controls required by an
// encoder-decoder generation pipeline. Task wrappers may add prefixes and
// forced language tokens before constructing the input batch.
type Seq2SeqOptions struct {
	MaxLength      int
	Temperature    *float64
	TopP           *float64
	Seed           *int
	StopSequences  []string
	DecoderStartID int
	EOSID          int
	ForcedBOSID    int
	TaskPrefix     string
}

// RunSeq2SeqSessionOnBatch is the shared generation entry point for
// summarization, translation, and text-to-text pipelines. The backend owns
// encoder/decoder session details and can reject unsupported cache/layout
// combinations without silently returning incomplete output.
func RunSeq2SeqSessionOnBatch(ctx context.Context, batch *PipelineBatch, pipeline *BasePipeline, options Seq2SeqOptions) (chan SequenceDelta, chan error, error) {
	if pipeline == nil || pipeline.Model == nil {
		return nil, nil, errors.New("seq2seq generation requires a model")
	}
	if !pipeline.Model.IsGenerative {
		return nil, nil, errors.New("seq2seq generation requires an encoder-decoder generative model")
	}
	if options.MaxLength <= 0 {
		return nil, nil, errors.New("seq2seq maximum length must be greater than zero")
	}
	if pipeline.Backend == nil {
		return nil, nil, errors.New("seq2seq generation backend is unavailable")
	}
	return pipeline.Backend.RunGenerativeSessionOnBatch(
		ctx, batch, pipeline, options.MaxLength, options.StopSequences,
		options.Temperature, options.TopP, options.Seed, nil, nil,
	)
}
