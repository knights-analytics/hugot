package util

import (
	"fmt"
	"math"
	"slices"
)

// Mean of a float32 vector.
func Mean(vector []float32) float32 {
	n := 0
	sum := float32(0.0)
	for _, v := range vector {
		sum = sum + v
		n++
	}
	return sum / float32(n)
}

// SoftMax take a vector and calculate softmax scores of its values.
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

// ArgMax find both index of max value in s and max value.
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

func Sigmoid(s []float32) []float32 {
	sigmoid := make([]float32, 0, len(s))

	for _, v := range s {
		v64 := float64(v)
		sigmoid = append(sigmoid, float32(1.0/(1.0+math.Exp(-v64))))
	}
	return sigmoid
}

// Norm of a vector.
func Norm(v []float32, p int) float64 {
	sum := 0.0
	pNorm := float64(p)
	for _, e := range v {
		sum += math.Pow(float64(e), pNorm)
	}
	return math.Sqrt(sum)
}

// Normalize single vector according to: https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
func Normalize(embedding []float32, p int) []float32 {
	const eps = 1e-12
	normalizeDenominator := float32(max(Norm(embedding, p), eps))
	for i, v := range embedding {
		embedding[i] = v / normalizeDenominator
	}
	return embedding
}
