package hugot

import (
	"iter"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/stretchr/testify/require"

	"github.com/knights-analytics/hugot/backends"
)

// The curve matches Hugging Face's get_linear_schedule_with_warmup and
// get_cosine_schedule_with_warmup, step for step.
func TestLearningRateScheduleRate(t *testing.T) {
	linear := &learningRateSchedule{decay: LinearDecay, warmupSteps: 2, totalSteps: 10, peak: 1}
	for step, want := range map[int]float64{0: 0, 1: 0.5, 2: 1, 6: 0.5, 9: 0.125, 10: 0, 12: 0} {
		require.InDelta(t, want, linear.rate(step), 1e-12, "linear, step %d", step)
	}
	cosine := &learningRateSchedule{decay: CosineDecay, warmupSteps: 2, totalSteps: 10, peak: 1}
	for step, want := range map[int]float64{1: 0.5, 2: 1, 6: 0.5, 10: 0, 14: 0} {
		require.InDelta(t, want, cosine.rate(step), 1e-12, "cosine, step %d", step)
	}
	constant := &learningRateSchedule{decay: ConstantRate, warmupSteps: 4, totalSteps: 10, peak: 3e-5}
	require.InDelta(t, 1.5e-5, constant.rate(2), 1e-18)
	require.InDelta(t, 3e-5, constant.rate(9), 1e-18)
	noWarmup := &learningRateSchedule{decay: LinearDecay, totalSteps: 4, peak: 1}
	require.InDelta(t, 1, noWarmup.rate(0), 1e-12, "without warmup the first step runs at the peak")
}

// The graph builds the same curve as the Go reference.
func TestLearningRateScheduleGraph(t *testing.T) {
	backend, err := compute.NewWithConfig("go")
	require.NoError(t, err)
	defer backend.Finalize()
	for _, schedule := range []*learningRateSchedule{
		{decay: LinearDecay, warmupSteps: 3, totalSteps: 20, peak: 5e-5},
		{decay: CosineDecay, warmupSteps: 3, totalSteps: 20, peak: 5e-5},
		{decay: ConstantRate, warmupSteps: 3, totalSteps: 20, peak: 5e-5},
		{decay: LinearDecay, totalSteps: 7, peak: 2e-5},
	} {
		exec, err := graph.NewExec(backend, func(step *graph.Node) *graph.Node {
			return schedule.learningRate(step)
		})
		require.NoError(t, err)
		for step := range schedule.totalSteps + 3 {
			got := exec.MustExec1(float32(step)).Value().(float32)
			require.InDelta(t, schedule.rate(step), float64(got), 1e-6*schedule.peak,
				"%s decay, step %d", schedule.decay, step)
		}
		exec.Finalize()
	}
}

// fakeDataset is a dataset that holds nothing and cannot count its batches.
type fakeDataset struct{}

func (*fakeDataset) Name() string { return "fake" }
func (*fakeDataset) Iter() iter.Seq2[train.Batch, error] {
	return func(func(train.Batch, error) bool) {}
}
func (*fakeDataset) Validate() error                                 { return nil }
func (*fakeDataset) SetTokenizationPipeline(backends.Pipeline) error { return nil }
func (*fakeDataset) SetVerbose(bool)                                 {}
func (*fakeDataset) Close() error                                    { return nil }

// countingDataset is a dataset of a fixed number of batches, for resolving schedules.
type countingDataset struct {
	*fakeDataset
	batches int
}

func (d countingDataset) NumBatches() (int, error) { return d.batches, nil }

func TestResolveSchedule(t *testing.T) {
	settings := trainerSettings{maxEpochs: 5}
	dataset := countingDataset{batches: 7}

	s, err := resolveSchedule(LearningRateSchedule{WarmupRatio: 0.1}, settings, dataset, 5e-5, false)
	require.NoError(t, err)
	require.Equal(t, 35, s.totalSteps)
	require.Equal(t, 4, s.warmupSteps, "3.5 warmup steps round up")
	require.InDelta(t, 5e-5, s.peak, 1e-18, "the trainer's default peak")

	s, err = resolveSchedule(LearningRateSchedule{WarmupSteps: 10, PeakLearningRate: 1e-4}, settings, dataset, 5e-5, true)
	require.NoError(t, err)
	require.Equal(t, 10, s.warmupSteps)
	require.InDelta(t, 1e-4, s.peak, 1e-18)

	_, err = resolveSchedule(LearningRateSchedule{}, settings, dataset, 5e-5, true)
	require.ErrorContains(t, err, "PeakLearningRate", "a caller's optimizer hides its rate")
	_, err = resolveSchedule(LearningRateSchedule{WarmupSteps: 35}, settings, dataset, 5e-5, false)
	require.ErrorContains(t, err, "shorter than the run")
	_, err = resolveSchedule(LearningRateSchedule{}, settings, countingDataset{}, 5e-5, false)
	require.ErrorContains(t, err, "empty")
	_, err = resolveSchedule(LearningRateSchedule{}, settings, &fakeDataset{}, 5e-5, false)
	require.ErrorContains(t, err, "BatchCounter")
}

func TestWithLearningRateScheduleValidates(t *testing.T) {
	for name, schedule := range map[string]LearningRateSchedule{
		"unknown decay":   {Decay: LearningRateDecay(9)},
		"ratio of 1":      {WarmupRatio: 1},
		"negative ratio":  {WarmupRatio: -0.1},
		"negative steps":  {WarmupSteps: -1},
		"ratio and steps": {WarmupRatio: 0.1, WarmupSteps: 5},
		"negative peak":   {PeakLearningRate: -1e-5},
	} {
		require.Error(t, WithLearningRateSchedule(schedule)(&trainerSettings{}), name)
	}
	settings := trainerSettings{}
	require.NoError(t, WithLearningRateSchedule(LearningRateSchedule{Decay: CosineDecay, WarmupRatio: 0.06})(&settings))
	require.Equal(t, CosineDecay, settings.schedule.Decay)
}
