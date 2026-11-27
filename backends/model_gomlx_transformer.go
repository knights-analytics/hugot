package backends

import (
	"errors"
	"fmt"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/go-huggingface/hub"
	hftransformer "github.com/gomlx/go-huggingface/models/transformer"
	"github.com/gomlx/gomlx/core/graph"
	mlmodel "github.com/gomlx/gomlx/ml/model"
	"github.com/knights-analytics/hugot/options"
)

// createGoMLXTransformerBackend loads a local Hugging Face transformer checkpoint
// (Safetensors or GGUF) and exposes its final hidden states as last_hidden_state.
func createGoMLXTransformerBackend(model *Model, opts *options.Options) error {
	if opts.Backend == options.BackendORT && !opts.UseGoMLX {
		return errors.New("Safetensors and GGUF models require a GoMLX backend")
	}
	hfModel, err := hftransformer.LoadModel(hub.NewLocal(model.Path))
	if err != nil {
		return fmt.Errorf("load GoMLX transformer model: %w", err)
	}

	config := "go"
	if opts.UseGoMLX {
		if opts.ORTOptions == nil || opts.ORTOptions.LibraryPath == nil {
			return errors.New("ONNX Runtime library path is not configured")
		}
		config = "onnx:" + *opts.ORTOptions.LibraryPath
	} else if opts.GoMLXOptions.TPU {
		config = "xla:tpu"
	} else if opts.GoMLXOptions.Cuda {
		config = "xla:cuda"
	} else if opts.GoMLXOptions.XLA {
		config = "xla:cpu"
	}
	backend, err := compute.NewWithConfig(config)
	if err != nil {
		return err
	}
	store := mlmodel.NewStore()
	scope := store.RootScope()
	if err := hfModel.LoadStore(backend, store); err != nil {
		store.Finalize()
		backend.Finalize()
		return fmt.Errorf("load GoMLX transformer weights: %w", err)
	}
	call := func(scope *mlmodel.Scope, inputs []*graph.Node) []*graph.Node {
		var seqLen *graph.Node
		if len(inputs) > 1 {
			seqLen = graph.ReduceSum(inputs[1], 1)
		}
		hidden := hfModel.ForwardGraph(scope, inputs[0], seqLen)
		return []*graph.Node{graph.ConvertDType(hidden, dtypes.Float32)}
	}
	exec, err := mlmodel.NewExec(backend, store, call)
	if err != nil {
		store.Finalize()
		backend.Finalize()
		return err
	}
	maxCache, batchBuckets, sequenceBuckets := getCacheAndBucketSizes(opts, config)
	exec.SetMaxCache(maxCache)
	dynamicAxes := [][]string{{"batch", "sequence"}, {"batch", "sequence"}}
	if !configureDynamicAxes(exec, backend, dynamicAxes) {
		dynamicAxes = nil
	}
	dim := shapes.DynamicDim
	model.GoMLXModel = &GoMLXModel{
		Backend: backend, Scope: scope, Store: store, Exec: exec,
		Call: call, HFTransformer: hfModel,
		inputShapes: []shapes.Shape{
			shapes.Make(dtypes.Int64, dim, dim),
			shapes.Make(dtypes.Int64, dim, dim),
		},
		DynamicAxes: dynamicAxes, DynamicAxesEnabled: len(dynamicAxes) > 0,
		PaddingEnabled: len(dynamicAxes) == 0, MaxCache: maxCache,
		BatchBuckets: batchBuckets, SequenceBuckets: sequenceBuckets,
	}
	model.InputsMeta = []InputOutputInfo{
		{Name: "input_ids", Dimensions: Shape{-1, -1}, DataType: DataTypeInt64},
		{Name: "attention_mask", Dimensions: Shape{-1, -1}, DataType: DataTypeInt64},
	}
	outputName := "last_hidden_state"
	outputDimensions := Shape{-1, -1, int64(hfModel.Config.HiddenSize)}
	if transformerReturnsPooledEmbedding(hfModel) {
		outputName = "sentence_embedding"
		outputDimensions = Shape{-1, int64(hfModel.Config.HiddenSize)}
	}
	model.OutputsMeta = []InputOutputInfo{{
		Name:       outputName,
		Dimensions: outputDimensions,
		DataType:   DataTypeFloat32,
	}}
	model.TransformerModel = true
	model.MaxPositionEmbeddings = hfModel.Config.MaxPositionEmbeddings
	model.ContextLength = hfModel.Config.MaxPositionEmbeddings
	return nil
}

// ForwardGraph switches to SentenceEmbeddingGraph when pooling is configured
// or when a sentence-transformer module stack has no Transformer module (the
// library's fallback for incomplete module metadata).
func transformerReturnsPooledEmbedding(model *hftransformer.Model) bool {
	if model.PoolingConfig != nil {
		return true
	}
	if len(model.Modules) == 0 {
		return false
	}
	hasTransformer := false
	for _, module := range model.Modules {
		switch module.Type {
		case "sentence_transformers.models.Pooling":
			return true
		case "sentence_transformers.models.Transformer":
			hasTransformer = true
		}
	}
	return !hasTransformer
}
