package util

import (
	"fmt"
	"math"
	"slices"
)

// mean of a float32 vector
func Mean(vector []float32) float32 {
	n := 0
	sum := float32(0.0)
	for _, v := range vector {
		sum = sum + v
		n++
	}
	return sum / float32(n)
}

// SoftMax take a vector and calculate softmax scores of its values
func SoftMax(vector []float32) []float32 {
	maxLogit := slices.Max(vector)
	shiftedExp := make([]float64, len(vector))
	for i, logit := range vector {
		shiftedExp[i] = math.Exp(float64(logit - maxLogit))
	}
	sumExp := SumSlice(shiftedExp)
	scores := make([]float32, len(vector))
	for i, exp := range shiftedExp {
		scores[i] = float32(exp / sumExp)
	}
	return scores
}

func SumSlice(s []float64) float64 {
	sum := 0.0
	for _, v := range s {
		sum += v
	}
	return sum
}

// ArgMax find both index of max value in s and max value
func ArgMax(s []float32) (int, float32, error) {
	if len(s) == 0 {
		return 0, 0, fmt.Errorf("attempted to calculate argmax of empty slice")
	}
	maxIndex := 0
	maxValue := s[0]
	for i, v := range s {
		if v > maxValue {
			maxValue = v
			maxIndex = i
		}
	}
	return maxIndex, maxValue, nil
}
