package hugot

import (
	"context"
	"errors"
	"fmt"
	"math"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	mlmodel "github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/loss"
	"github.com/gomlx/gomlx/ml/train/metric"
	"github.com/gomlx/gomlx/ml/train/optimizer"

	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/datasets"
	"github.com/knights-analytics/hugot/evaluation"
	"github.com/knights-analytics/hugot/pipelines"
)

// Defaults for fine-tuning a classifier: AdamW at the learning rate and weight decay that Hugging
// Face's Trainer uses. Plain SGD at its default rate barely moves a pretrained transformer.
const (
	defaultClassificationLearningRate = 5e-5
	defaultClassificationWeightDecay  = 0.01
	// defaultContrastiveLearningRate is sentence-transformers' default for contrastive training.
	defaultContrastiveLearningRate = 2e-5
	// evaluationBatchSize is the inference batch size for the per-epoch F1 metrics.
	evaluationBatchSize = 32
)

// GOMLXTrainingOptions overrides the optimizer and loss function used for training.
type GOMLXTrainingOptions struct {
	Optimizer optimizer.Interface
	Loss      loss.LossFn
}

type stoppingError struct{}

func (e stoppingError) Error() string {
	return "stopping error"
}

// checkTrainable rejects a pipeline type that cannot be trained. It depends only on T, so
// NewTrainer calls it before loading anything: an unsupported type fails with no side effects.
func checkTrainable[T backends.Pipeline]() error {
	var zero T
	switch any(zero).(type) {
	case *pipelines.FeatureExtractionPipeline, *pipelines.TextClassificationPipeline, *pipelines.TokenClassificationPipeline:
		return nil
	default:
		return fmt.Errorf("training is not supported for pipeline type %T", zero)
	}
}

// applyTrainingDefaults fills in the optimizer and loss function for the pipeline type. The type
// has already been accepted by checkTrainable.
func applyTrainingDefaults[T backends.Pipeline](config *TrainerConfig[T]) {
	var zero T
	switch any(zero).(type) {
	case *pipelines.FeatureExtractionPipeline:
		if config.GOMLXOptions == nil {
			config.GOMLXOptions = &GOMLXTrainingOptions{}
		}
		if isContrastive(config.TrainDataset) {
			// sentence-transformers' defaults for this loss: AdamW at 2e-5, weight decay 0.01.
			if config.GOMLXOptions.Optimizer == nil {
				config.GOMLXOptions.Optimizer = optimizer.Adam().
					LearningRate(defaultContrastiveLearningRate).
					WeightDecay(defaultClassificationWeightDecay).
					Done()
			}
			if config.GOMLXOptions.Loss == nil {
				config.GOMLXOptions.Loss = loss.SparseCategoricalCrossEntropyLogits
			}
			return
		}
		if config.GOMLXOptions.Optimizer == nil {
			config.GOMLXOptions.Optimizer = optimizer.StochasticGradientDescent()
		}
		if config.GOMLXOptions.Loss == nil {
			config.GOMLXOptions.Loss = loss.MeanSquaredError
		}
	case *pipelines.TextClassificationPipeline, *pipelines.TokenClassificationPipeline:
		// A token classification dataset has no Problem method, so it trains as single-label: one
		// tag per token, with the untagged tokens masked out by the dataset.
		if config.GOMLXOptions == nil {
			config.GOMLXOptions = &GOMLXTrainingOptions{}
		}
		if config.GOMLXOptions.Optimizer == nil {
			config.GOMLXOptions.Optimizer = optimizer.Adam().
				LearningRate(defaultClassificationLearningRate).
				WeightDecay(defaultClassificationWeightDecay).
				Done()
		}
		if config.GOMLXOptions.Loss == nil {
			if classificationProblem(config.TrainDataset) == datasets.MultiLabel {
				config.GOMLXOptions.Loss = loss.BinaryCrossentropyLogits
			} else {
				config.GOMLXOptions.Loss = loss.SparseCategoricalCrossEntropyLogits
			}
		}
	}
}

// isContrastive reports whether an embedding pipeline trains on anchor/positive pairs with
// in-batch negatives, rather than on scored sentence pairs.
func isContrastive(dataset datasets.Dataset) bool {
	_, ok := dataset.(*datasets.ContrastiveDataset)
	return ok
}

// classificationProblem reports whether a classification dataset is single-label or multi-label. A
// dataset that does not say is treated as single-label.
func classificationProblem(dataset datasets.Dataset) datasets.ClassificationProblem {
	if d, ok := dataset.(interface {
		Problem() datasets.ClassificationProblem
	}); ok {
		return d.Problem()
	}
	return datasets.SingleLabel
}

// classificationAccuracy is the evaluation metric for a classification dataset.
func classificationAccuracy(dataset datasets.Dataset) metric.Interface {
	if classificationProblem(dataset) == datasets.MultiLabel {
		return metric.NewMeanBinaryLogitsAccuracy("Mean Accuracy", "accuracy")
	}
	return metric.NewSparseCategoricalAccuracy("Mean Accuracy", "accuracy")
}

// freezeVariables applies the freezeEmbeddings and freezeLayers settings to the model's variables.
func freezeVariables(store *mlmodel.Store, settings trainerSettings) error {
	freezeAllButLast := slices.Contains(settings.freezeLayers, -1)
	if !settings.freezeEmbeddings && !freezeAllButLast && len(settings.freezeLayers) == 0 {
		return nil
	}
	re := regexp.MustCompile(`layer\.(\d+)`) // identify the layer number in the variable name

	for v := range store.IterVariables() {
		name := v.Name()
		if (settings.freezeEmbeddings || freezeAllButLast) && strings.HasPrefix(name, "embeddings") {
			v.SetTrainable(false)
			continue
		}
		matches := re.FindStringSubmatch(name)
		if matches == nil {
			continue
		}
		layerNum, err := strconv.Atoi(matches[1])
		if err != nil {
			return fmt.Errorf("failed to parse layer number from variable name %s: %w", name, err)
		}
		if freezeAllButLast || slices.Contains(settings.freezeLayers, layerNum) {
			v.SetTrainable(false)
		}
	}
	return nil
}

// trainGoMLX runs the training loop. Every backend trains through GoMLX — an ORT session uses it
// via options.WithGoMLX — so there is no backend dispatch here.
func (t *Trainer[T]) trainGoMLX(ctx context.Context) error {
	var modelFn func(*mlmodel.Scope, any, []*mlmodel.Node) []*mlmodel.Node
	var evalMetrics []metric.Interface
	switch p := any(t.pipeline).(type) {
	case *pipelines.FeatureExtractionPipeline:
		if isContrastive(t.config.TrainDataset) {
			modelFn = contrastiveModelFn(p.Model)
			// The share of anchors whose own positive scores highest: in-batch retrieval accuracy.
			evalMetrics = []metric.Interface{metric.NewSparseCategoricalAccuracy("Mean Accuracy", "accuracy")}
		} else {
			modelFn = semanticSimilarityModelFn(p.Model)
		}
	case *pipelines.TextClassificationPipeline:
		modelFn = classificationModelFn(p.Model.GoMLXModel)
		evalMetrics = []metric.Interface{classificationAccuracy(t.config.TrainDataset)}
	case *pipelines.TokenClassificationPipeline:
		// Logits of shape [batch, sequence, labels]; accuracy is over the tagged tokens only.
		modelFn = classificationModelFn(p.Model.GoMLXModel)
		evalMetrics = []metric.Interface{classificationAccuracy(t.config.TrainDataset)}
	default:
		// Unreachable via NewTrainer, which rejects unsupported types at construction. Kept as a
		// real error rather than a silent no-op, which is what this switch used to do.
		return fmt.Errorf("training is not supported for pipeline type %T", t.pipeline)
	}

	goMLXModel := t.pipeline.GetModel().GoMLXModel
	backend := goMLXModel.Backend
	store := goMLXModel.Store

	if err := freezeVariables(store, t.settings); err != nil {
		return err
	}

	gomlxTrainer := train.NewTrainer(backend,
		store,
		modelFn,
		t.config.GOMLXOptions.Loss,
		t.config.GOMLXOptions.Optimizer,
		nil,
		evalMetrics)

	return t.runTrainingLoop(ctx, gomlxTrainer, len(evalMetrics) > 0)
}

// classificationModelFn returns the model's logits: [batch, labels] for text classification, and
// [batch, sequence, labels] for token classification.
func classificationModelFn(goMLXModel *backends.GoMLXModel) func(*mlmodel.Scope, any, []*mlmodel.Node) []*mlmodel.Node {
	return func(scope *mlmodel.Scope, _ any, inputs []*mlmodel.Node) []*mlmodel.Node {
		return goMLXModel.Call(scope, inputs)[:1]
	}
}

// modelInputLayout returns how many input tensors the model takes per text, and which of them is
// the attention mask.
func modelInputLayout(model *backends.Model) (numInputs, maskIndex int) {
	maskIndex = 1
	for i, input := range model.InputsMeta {
		if input.Name == "attention_mask" {
			maskIndex = i
		}
	}
	return len(model.InputsMeta), maskIndex
}

// embed runs the model on one group of inputs and returns one embedding per row, mean pooling
// over the attention mask when the model returns token embeddings ([batch, seq, hidden]).
func embed(scope *mlmodel.Scope, goMLXModel *backends.GoMLXModel, inputs []*mlmodel.Node, maskIndex int) *mlmodel.Node {
	embedding := goMLXModel.Call(scope, inputs)[0]
	if embedding.Rank() > 2 {
		batchSize := embedding.Shape().Dim(0)
		embeddingSize := embedding.Shape().Dim(-1)
		embedding = graph.Reshape(embedding, batchSize, -1, embeddingSize)
		mask := graph.ConvertDType(graph.BroadcastToShape(graph.Reshape(inputs[maskIndex], batchSize, -1, 1), embedding.Shape()), dtypes.Bool)
		embedding = graph.MaskedReduceMean(embedding, mask, 1)
	}
	return embedding
}

// semanticSimilarityModelFn returns the cosine similarity of the two sentences' embeddings.
func semanticSimilarityModelFn(model *backends.Model) func(*mlmodel.Scope, any, []*mlmodel.Node) []*mlmodel.Node {
	numInputs, maskIndex := modelInputLayout(model)
	return func(scope *mlmodel.Scope, _ any, inputs []*mlmodel.Node) []*mlmodel.Node {
		embeddingLHS := embed(scope, model.GoMLXModel, inputs[:numInputs], maskIndex)
		embeddingRHS := embed(scope, model.GoMLXModel, inputs[numInputs:], maskIndex)
		return []*mlmodel.Node{graph.CosineSimilarity(embeddingLHS, embeddingRHS, -1)}
	}
}

// contrastiveScale multiplies the cosine similarities before the softmax, as the scale of
// sentence-transformers' MultipleNegativesRankingLoss (default 20) does. Cosines lie in [-1, 1],
// too narrow a range for a softmax to separate the right candidate sharply.
const contrastiveScale = 20.0

// contrastiveModelFn scores every anchor against every candidate in the batch. The inputs are the
// anchors, the positives and optionally the negatives, each a group of model inputs; it returns
// the scaled cosine similarities, [anchors, candidates], with anchor i's positive at column i.
func contrastiveModelFn(model *backends.Model) func(*mlmodel.Scope, any, []*mlmodel.Node) []*mlmodel.Node {
	numInputs, maskIndex := modelInputLayout(model)
	return func(scope *mlmodel.Scope, _ any, inputs []*mlmodel.Node) []*mlmodel.Node {
		groups := len(inputs) / numInputs
		anchors := graph.L2Normalize(embed(scope, model.GoMLXModel, inputs[:numInputs], maskIndex), -1)
		candidates := make([]*mlmodel.Node, 0, groups-1)
		for g := 1; g < groups; g++ {
			candidates = append(candidates, embed(scope, model.GoMLXModel, inputs[g*numInputs:(g+1)*numInputs], maskIndex))
		}
		candidateMatrix := graph.L2Normalize(graph.Concatenate(candidates, 0), -1)
		scores := graph.Einsum("ad,cd->ac", anchors, candidateMatrix)
		return []*mlmodel.Node{graph.MulScalar(scores, contrastiveScale)}
	}
}

// resetInference makes the model's next inference use its current weights; see GoMLXModel.ResetExec.
func resetInference(model *backends.Model, needed bool) error {
	if !needed {
		return nil
	}
	return model.GoMLXModel.ResetExec()
}

// runTrainingLoop runs the epochs, evaluating after each one for the statistics and early stopping.
// With withAccuracy, the first evaluation metric after the loss is recorded as the accuracy.
func (t *Trainer[T]) runTrainingLoop(ctx context.Context, gomlxTrainer *train.Trainer, withAccuracy bool) error {
	loop := train.NewLoop(gomlxTrainer)

	if t.config.Verbose {
		if t.settings.earlyStopping != nil {
			fmt.Printf("Training for %d epochs with early stopping\n", t.settings.maxEpochs)
		} else {
			fmt.Printf("Training for %d epochs\n", t.settings.maxEpochs)
		}
	}

	var currentEpoch int
	var trainLosses, evalLosses []float32
	var trainAccuracies, evalAccuracies []float32
	var trainF1s, evalF1s []float32

	var bestLoss float32 = math.MaxFloat32
	epochsWithoutImprovement := 0

	// A backend that prefers constants for variables (the Go backend) bakes the weights into any
	// graph built outside training mode, and the GoMLX trainer caches its evaluation graph, so every
	// evaluation would see the weights of the first one. Dropping the cached graphs makes it rebuild
	// them with the current weights.
	rebuildBeforeEval := gomlxTrainer.Backend().Capabilities().PreferConstantsForVariables
	evalDataset := func(dataset datasets.Dataset) ([]*tensors.Tensor, error) {
		if rebuildBeforeEval {
			gomlxTrainer.ResetComputationGraphs()
		}
		return gomlxTrainer.Eval(dataset)
	}

	// entityF1 is the entity-level F1 on a token classification dataset, or ok=false for any other
	// pipeline or dataset. It runs inference with the current weights.
	// inferenceF1 is the F1 that matters for the pipeline, computed by running inference with the
	// current weights: entity-level F1 for token classification, macro-F1 for text classification.
	// ok is false for any other pipeline or dataset.
	inferenceF1 := func(dataset datasets.Dataset) (f1 float32, ok bool, err error) {
		switch p := any(t.pipeline).(type) {
		case *pipelines.TokenClassificationPipeline:
			d, isTokenDataset := dataset.(*datasets.TokenClassificationDataset)
			if !isTokenDataset {
				return 0, false, nil
			}
			if err := resetInference(p.Model, rebuildBeforeEval); err != nil {
				return 0, false, err
			}
			examples, err := d.Examples()
			if err != nil {
				return 0, false, err
			}
			report, err := evaluation.EvaluateTokenClassification(ctx, p, examples, evaluationBatchSize)
			if err != nil {
				return 0, false, err
			}
			return float32(report.F1), true, nil
		case *pipelines.TextClassificationPipeline:
			d, isTextDataset := dataset.(*datasets.TextClassificationDataset)
			if !isTextDataset {
				return 0, false, nil
			}
			if err := resetInference(p.Model, rebuildBeforeEval); err != nil {
				return 0, false, err
			}
			examples, err := d.Examples()
			if err != nil {
				return 0, false, err
			}
			report, err := evaluation.EvaluateTextClassification(ctx, p, examples, evaluationBatchSize)
			if err != nil {
				return 0, false, err
			}
			return float32(report.MacroF1), true, nil
		}
		return 0, false, nil
	}

	// evaluate records the statistics for a finished epoch, and returns stoppingError when early
	// stopping says to stop.
	evaluate := func(epoch int) error {
		if t.config.TrainEvalDataset != nil {
			if t.config.Verbose {
				fmt.Printf("Running evaluation for epoch %d on trainEvalDataset\n", epoch)
			}
			lossAndMetrics, err := evalDataset(t.config.TrainEvalDataset)
			if err != nil {
				return err
			}
			trainLosses = append(trainLosses, lossAndMetrics[0].Value().(float32))
			if withAccuracy {
				trainAccuracies = append(trainAccuracies, lossAndMetrics[1].Value().(float32))
			}
			if f1, ok, err := inferenceF1(t.config.TrainEvalDataset); err != nil {
				return err
			} else if ok {
				trainF1s = append(trainF1s, f1)
			}
		}

		if t.config.EvalDataset != nil {
			if t.config.Verbose {
				fmt.Printf("Running evaluation for epoch %d on evalDataset\n", epoch)
			}
			lossAndMetrics, err := evalDataset(t.config.EvalDataset)
			if err != nil {
				return err
			}
			meanLoss := lossAndMetrics[0].Value().(float32)
			evalLosses = append(evalLosses, meanLoss)
			if withAccuracy {
				evalAccuracies = append(evalAccuracies, lossAndMetrics[1].Value().(float32))
			}
			if f1, ok, err := inferenceF1(t.config.EvalDataset); err != nil {
				return err
			} else if ok {
				evalF1s = append(evalF1s, f1)
			}
		}

		if t.settings.earlyStopping != nil {
			meanLoss := evalLosses[len(evalLosses)-1]
			if bestLoss-meanLoss > t.settings.earlyStopping.tolerance {
				bestLoss = meanLoss
				epochsWithoutImprovement = 0
				if t.config.Verbose {
					fmt.Printf("New best loss: %.4f at epoch %d\n", bestLoss, epoch)
				}
			} else {
				epochsWithoutImprovement++
				if t.config.Verbose {
					fmt.Printf("No improvement in loss, epochs without improvement: %d\n", epochsWithoutImprovement)
				}
				if epochsWithoutImprovement >= t.settings.earlyStopping.patience {
					if t.config.Verbose {
						fmt.Printf("Early stopping triggered after %d epochs without improvement\n", t.settings.earlyStopping.patience)
					}
					return stoppingError{} // trigger stopping
				}
			}
		}
		return nil
	}

	// The loop has no end-of-epoch hook, so an epoch is evaluated at the first step of the next
	// one, and the last epoch after the loop returns.
	var evaluateEpoch train.OnStepFn = func(loop *train.Loop, _ []*tensors.Tensor) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		if loop.Epoch == currentEpoch {
			return nil
		}
		if err := evaluate(currentEpoch); err != nil {
			return err
		}
		currentEpoch = loop.Epoch
		return nil
	}
	loop.OnStep("evaluateAfterEpoch", train.Priority(1), evaluateEpoch)

	// we rely on try catch because an error is returned if there is an initialization error but
	// a panic will be thrown if e.g. dataset reset fails.
	stopped := false
	if _, err := loop.RunEpochs(t.config.TrainDataset, t.settings.maxEpochs); err != nil {
		if !errors.Is(err, stoppingError{}) {
			return err
		}
		stopped = true
		if t.config.Verbose {
			fmt.Printf("Training stopped after epoch %d\n", currentEpoch)
		}
	}
	if !stopped {
		if err := evaluate(currentEpoch); err != nil && !errors.Is(err, stoppingError{}) {
			return err
		}
	}

	if t.config.Verbose {
		fmt.Println("Training complete")
	}
	t.statistics = TrainingStatistics{
		EpochTrainLosses:     trainLosses,
		EpochEvalLosses:      evalLosses,
		EpochTrainAccuracies: trainAccuracies,
		EpochEvalAccuracies:  evalAccuracies,
		EpochTrainF1:         trainF1s,
		EpochEvalF1:          evalF1s,
	}
	return nil
}
