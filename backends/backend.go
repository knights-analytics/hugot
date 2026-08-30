package backends

import (
	"context"
	"fmt"

	"github.com/knights-analytics/hugot/options"
)

// Backend owns all operations that depend on the selected inference runtime.
// Pipelines use this contract instead of switching on a runtime name.
type Backend interface {
	RunSessionOnBatch(context.Context, *PipelineBatch, *BasePipeline) error
	RunGenerativeSessionOnBatch(context.Context, *PipelineBatch, *BasePipeline, int, []string, *float64, *float64, *int, []string, *Guidance) (chan SequenceDelta, chan error, error)
	CreateMessages(*PipelineBatch, any, string) error
	CreateInputTensors(*PipelineBatch, *Model, bool) error
	CreateImageTensors(*PipelineBatch, *Model, [][][][]float32) error
	CreateTabularTensors(*PipelineBatch, *Model, [][]float32) error
}

type runtimeBackend struct {
	runtime options.Backend
}

func newBackend(opts *options.Options) (Backend, error) {
	if opts == nil || !opts.Backend.Valid() {
		return nil, fmt.Errorf("invalid backend")
	}
	if opts.Backend == options.BackendORT && !opts.UseGoMLX {
		return runtimeBackend{runtime: options.BackendORT}, nil
	}
	if opts.Backend == options.BackendXLA {
		return runtimeBackend{runtime: options.BackendXLA}, nil
	}
	return runtimeBackend{runtime: options.BackendGo}, nil
}

func (b runtimeBackend) RunSessionOnBatch(ctx context.Context, batch *PipelineBatch, p *BasePipeline) error {
	if b.runtime == options.BackendORT {
		return runORTSessionOnBatch(ctx, batch, p)
	}
	return runGoMLXSessionOnBatch(ctx, batch, p)
}

func (b runtimeBackend) RunGenerativeSessionOnBatch(ctx context.Context, batch *PipelineBatch, p *BasePipeline, maxLength int, stopSequences []string, temperature *float64, topP *float64, seed *int, tools []string, guidance *Guidance) (chan SequenceDelta, chan error, error) {
	if b.runtime == options.BackendORT {
		return runGenerativeORTSessionOnBatch(ctx, batch, p, maxLength, stopSequences, temperature, topP, seed, tools, guidance)
	}
	return nil, nil, &backendUnavailableError{runtime: string(b.runtime), operation: "generative inference"}
}

func (b runtimeBackend) CreateMessages(batch *PipelineBatch, inputs any, systemPrompt string) error {
	if b.runtime == options.BackendORT {
		return CreateMessagesORT(batch, inputs, systemPrompt)
	}
	return &backendUnavailableError{runtime: string(b.runtime), operation: "messages"}
}

func (b runtimeBackend) CreateInputTensors(batch *PipelineBatch, model *Model, training bool) error {
	if b.runtime == options.BackendORT {
		return createInputTensorsORT(batch, model)
	}
	return createInputTensorsGoMLX(batch, model, !training, b.runtime == options.BackendXLA)
}

func (b runtimeBackend) CreateImageTensors(batch *PipelineBatch, model *Model, preprocessed [][][][]float32) error {
	if b.runtime == options.BackendORT {
		return createImageTensorsORT(batch, model, preprocessed)
	}
	return createImageTensorsGoXLA(batch, model, preprocessed)
}

func (b runtimeBackend) CreateTabularTensors(batch *PipelineBatch, model *Model, features [][]float32) error {
	if b.runtime == options.BackendORT {
		return createTabularTensorsORT(batch, model, features)
	}
	return createTabularTensorsGoMLX(batch, model, features)
}

type backendUnavailableError struct {
	runtime   string
	operation string
}

func (e *backendUnavailableError) Error() string {
	return e.runtime + " backend does not support " + e.operation
}
