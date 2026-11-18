package hugot

import (
	"errors"
	"fmt"
	"math"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/gomlx/core/graph"
	mlmodel "github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/train/optimizer"

	"github.com/knights-analytics/hugot/datasets"
)

// LearningRateDecay is how the learning rate falls after the warmup.
type LearningRateDecay int

const (
	// LinearDecay lowers the rate in a straight line, from its peak at the end of the warmup to 0 at
	// the last step. It is the default of Hugging Face's Trainer.
	LinearDecay LearningRateDecay = iota
	// CosineDecay lowers the rate from its peak to 0 along half a cosine: slowly at first, fastest
	// halfway, and slowly again near the end.
	CosineDecay
	// ConstantRate keeps the rate at its peak after the warmup.
	ConstantRate
)

func (d LearningRateDecay) String() string {
	switch d {
	case LinearDecay:
		return "linear"
	case CosineDecay:
		return "cosine"
	case ConstantRate:
		return "constant"
	}
	return fmt.Sprintf("LearningRateDecay(%d)", int(d))
}

// LearningRateSchedule sets how the learning rate changes over a training run: a linear warmup
// from 0 to PeakLearningRate, then a decay. It follows Hugging Face's schedules, step for step.
//
// The schedule is laid over every step of the run, maxEpochs times the batches in one pass over
// the training dataset (see WithMaxEpochs). Early stopping ends the run before the rate has
// decayed all the way, as it does in Hugging Face.
type LearningRateSchedule struct {
	// Decay after the warmup. The zero value is LinearDecay.
	Decay LearningRateDecay
	// WarmupRatio is the share of the run's steps spent warming up, in [0, 1). A few percent is
	// usual: 0.06 to 0.1. Set at most one of WarmupRatio and WarmupSteps.
	WarmupRatio float64
	// WarmupSteps is the number of steps spent warming up.
	WarmupSteps int
	// PeakLearningRate is the rate at the end of the warmup. It replaces the optimizer's own rate.
	// Left at 0, it is the trainer's default for the pipeline, which only applies when the trainer
	// also builds the optimizer: with GOMLXTrainingOptions.Optimizer set, it is required.
	PeakLearningRate float64
}

// WithLearningRateSchedule warms the learning rate up and then decays it, rather than keeping it
// constant. Warming up protects the pretrained weights from the large, noisy gradients of a new or
// resized classification head in the first steps; decaying lets training settle at the end.
//
// The training dataset must implement [datasets.BatchCounter], as all of hugot's datasets do. For
// a .jsonl file, the trainer reads it once more at construction to count its examples.
func WithLearningRateSchedule(schedule LearningRateSchedule) TrainerOption {
	return func(t *trainerSettings) error {
		switch schedule.Decay {
		case LinearDecay, CosineDecay, ConstantRate:
		default:
			return fmt.Errorf("unknown learning rate decay %d", int(schedule.Decay))
		}
		if schedule.WarmupRatio < 0 || schedule.WarmupRatio >= 1 || math.IsNaN(schedule.WarmupRatio) {
			return fmt.Errorf("warmup ratio must be in [0, 1), got %g", schedule.WarmupRatio)
		}
		if schedule.WarmupSteps < 0 {
			return fmt.Errorf("warmup steps must not be negative, got %d", schedule.WarmupSteps)
		}
		if schedule.WarmupRatio > 0 && schedule.WarmupSteps > 0 {
			return errors.New("set at most one of WarmupRatio and WarmupSteps")
		}
		if schedule.PeakLearningRate < 0 || math.IsNaN(schedule.PeakLearningRate) || math.IsInf(schedule.PeakLearningRate, 0) {
			return fmt.Errorf("peak learning rate must be a positive number, got %g", schedule.PeakLearningRate)
		}
		t.schedule = &schedule
		return nil
	}
}

// learningRateSchedule is a LearningRateSchedule resolved against a run: its length in steps and
// its peak rate are known.
type learningRateSchedule struct {
	decay       LearningRateDecay
	warmupSteps int
	totalSteps  int
	peak        float64
}

// resolveSchedule fixes a schedule's peak rate and its length, from the trainer's settings and the
// training dataset. userOptimizer is whether the caller supplied the optimizer, whose rate the
// trainer cannot read.
func resolveSchedule(schedule LearningRateSchedule, settings trainerSettings, dataset datasets.Dataset,
	defaultPeak float64, userOptimizer bool,
) (*learningRateSchedule, error) {
	peak := schedule.PeakLearningRate
	if peak == 0 {
		if userOptimizer {
			return nil, errors.New("a learning rate schedule with GOMLXTrainingOptions.Optimizer set needs " +
				"LearningRateSchedule.PeakLearningRate: the trainer cannot read the optimizer's own rate")
		}
		peak = defaultPeak
	}
	counter, ok := dataset.(datasets.BatchCounter)
	if !ok {
		return nil, fmt.Errorf("a learning rate schedule needs a training dataset that can count its batches, "+
			"and %T cannot (see datasets.BatchCounter)", dataset)
	}
	batches, err := counter.NumBatches()
	if err != nil {
		return nil, fmt.Errorf("failed to count the training batches for the learning rate schedule: %w", err)
	}
	if batches == 0 {
		return nil, errors.New("the training dataset is empty")
	}
	total := settings.maxEpochs * batches
	warmup := schedule.WarmupSteps
	if schedule.WarmupRatio > 0 {
		// Rounded up, as Hugging Face does, so that any ratio above 0 warms up for at least a step.
		warmup = int(math.Ceil(schedule.WarmupRatio * float64(total)))
	}
	if warmup >= total {
		return nil, fmt.Errorf("the warmup (%d steps) must be shorter than the run (%d steps)", warmup, total)
	}
	return &learningRateSchedule{decay: schedule.Decay, warmupSteps: warmup, totalSteps: total, peak: peak}, nil
}

// rate is the learning rate at a step, counting from 0. It is the reference for learningRate, the
// same curve built in the graph, and matches Hugging Face's get_*_schedule_with_warmup: the first
// warmup step has a rate of 0.
func (s *learningRateSchedule) rate(step int) float64 {
	if step < s.warmupSteps {
		return s.peak * float64(step) / float64(s.warmupSteps)
	}
	progress := float64(step-s.warmupSteps) / float64(max(1, s.totalSteps-s.warmupSteps))
	progress = min(progress, 1)
	switch s.decay {
	case LinearDecay:
		return s.peak * (1 - progress)
	case CosineDecay:
		return s.peak * 0.5 * (1 + math.Cos(math.Pi*progress))
	default:
		return s.peak
	}
}

// learningRate builds the rate at step (a scalar counting from 0) into the graph.
func (s *learningRateSchedule) learningRate(step *graph.Node) *graph.Node {
	g := step.Graph()
	dtype := step.DType()
	progress := graph.DivScalar(graph.SubScalar(step, float64(s.warmupSteps)), float64(max(1, s.totalSteps-s.warmupSteps)))
	progress = graph.MinScalar(graph.MaxScalar(progress, 0.0), 1.0)
	var factor *graph.Node
	switch s.decay {
	case LinearDecay:
		factor = graph.OneMinus(progress)
	case CosineDecay:
		factor = graph.MulScalar(graph.AddScalar(graph.Cos(graph.MulScalar(progress, math.Pi)), 1.0), 0.5)
	default:
		factor = graph.OnesLike(step)
	}
	if s.warmupSteps > 0 {
		warming := graph.DivScalar(step, float64(s.warmupSteps))
		factor = graph.Where(graph.LessThan(step, graph.Scalar(g, dtype, s.warmupSteps)), warming, factor)
	}
	return graph.MulScalar(factor, s.peak)
}

// scheduleScope holds the schedule's own step counter, beside the optimizer's variables.
const scheduleScope = "hugot_learning_rate_schedule"

// withSchedule wraps a model function so that each training step first sets the optimizer's
// learning rate from the schedule. The optimizer reads the rate after the model function has run,
// in the same graph, so it trains the step at the scheduled rate. Without a schedule, the model
// function is returned as it is.
func withSchedule(modelFn func(*mlmodel.Scope, any, []*mlmodel.Node) []*mlmodel.Node,
	schedule *learningRateSchedule,
) func(*mlmodel.Scope, any, []*mlmodel.Node) []*mlmodel.Node {
	if schedule == nil {
		return modelFn
	}
	return func(scope *mlmodel.Scope, spec any, inputs []*mlmodel.Node) []*mlmodel.Node {
		g := inputs[0].Graph()
		if scope.IsTraining(g) {
			counterScope := scope.Store().Scope(mlmodel.JoinPath(optimizer.Scope, scheduleScope))
			// The counter returns its value after this step's increment, so the step is one less.
			step := graph.SubScalar(optimizer.IncrementCounter(counterScope, g, "step", dtypes.Float32), 1.0)
			optimizer.LearningRateVarWithValue(scope, dtypes.Float32, schedule.peak).
				SetNodeValue(schedule.learningRate(step))
		}
		return modelFn(scope, spec, inputs)
	}
}

// currentLearningRate reads the optimizer's learning rate from the store: after a training step
// under a schedule, the rate that step used. ok is false before the first step.
func currentLearningRate(store *mlmodel.Store) (rate float32, ok bool, err error) {
	v := store.GetVariable(mlmodel.JoinPath(optimizer.Scope, optimizer.ParamLearningRate))
	if v == nil {
		return 0, false, nil
	}
	value, err := v.Value()
	if err != nil {
		return 0, false, err
	}
	rate, err = scalarValue(value)
	return rate, err == nil, err
}
