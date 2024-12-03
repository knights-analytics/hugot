//go:build !XLA && !ALL

package hugot

type XLATrainingOptions struct{}

func TrainXLA(_ *TrainingSession) error {
	return nil
}
