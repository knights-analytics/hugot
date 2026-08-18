package vectorutil

import (
	"fmt"
	"math"
	"slices"
)

// Mean of a float32 vector.
func Mean(vector []float32) float32 {
	n := len(vector)
	if n == 0 {
		return float32(math.NaN())
	}

	var s0, s1, s2, s3 float32
	i := 0
	for ; i+3 < n; i += 4 {
		s0 += vector[i]
		s1 += vector[i+1]
		s2 += vector[i+2]
		s3 += vector[i+3]
	}
	sum := (s0 + s1) + (s2 + s3)
	for ; i < n; i++ {
		sum += vector[i]
	}
	return sum / float32(n)
}

// SoftMax take a vector and calculate softmax scores of its values.
func SoftMax(vector []float32) []float32 {
	if len(vector) == 0 {
		return []float32{}
	}

	maxLogit := slices.Max(vector)
	scores := make([]float32, len(vector))

	var s0, s1, s2, s3 float64
	i := 0
	for ; i+3 < len(vector); i += 4 {
		e0 := math.Exp(float64(vector[i] - maxLogit))
		e1 := math.Exp(float64(vector[i+1] - maxLogit))
		e2 := math.Exp(float64(vector[i+2] - maxLogit))
		e3 := math.Exp(float64(vector[i+3] - maxLogit))

		scores[i] = float32(e0)
		scores[i+1] = float32(e1)
		scores[i+2] = float32(e2)
		scores[i+3] = float32(e3)

		s0 += e0
		s1 += e1
		s2 += e2
		s3 += e3
	}
	sumExp := (s0 + s1) + (s2 + s3)

	for ; i < len(vector); i++ {
		e := math.Exp(float64(vector[i] - maxLogit))
		scores[i] = float32(e)
		sumExp += e
	}

	inverseSum := float32(1 / sumExp)
	for i := range scores {
		scores[i] *= inverseSum
	}
	return scores
}

func SumSlice(s []float64) float64 {
	var s0, s1, s2, s3 float64
	i := 0
	for ; i+3 < len(s); i += 4 {
		s0 += s[i]
		s1 += s[i+1]
		s2 += s[i+2]
		s3 += s[i+3]
	}
	sum := (s0 + s1) + (s2 + s3)
	for ; i < len(s); i++ {
		sum += s[i]
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
	for i := 1; i < len(s); i++ {
		if s[i] > maxValue {
			maxValue = s[i]
			maxIndex = i
		}
	}
	return maxIndex, maxValue, nil
}

func Sigmoid(s []float32) []float32 {
	sigmoid := make([]float32, len(s))
	for i, v := range s {
		sigmoid[i] = float32(1 / (1 + math.Exp(-float64(v))))
	}
	return sigmoid
}

// Norm of a vector.
func Norm(v []float32, p int) float64 {
	if p <= 0 {
		return math.NaN()
	}

	if p == 2 {
		var s0, s1, s2, s3 float64
		i := 0
		for ; i+3 < len(v); i += 4 {
			v0 := float64(v[i])
			v1 := float64(v[i+1])
			v2 := float64(v[i+2])
			v3 := float64(v[i+3])
			s0 += v0 * v0
			s1 += v1 * v1
			s2 += v2 * v2
			s3 += v3 * v3
		}

		sum := (s0 + s1) + (s2 + s3)
		for ; i < len(v); i++ {
			x := float64(v[i])
			sum += x * x
		}
		return math.Sqrt(sum)
	}

	sum := 0.0
	pNorm := float64(p)
	for _, e := range v {
		sum += math.Pow(math.Abs(float64(e)), pNorm)
	}
	return math.Pow(sum, 1/pNorm)
}

// Normalize single vector according to: https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
func Normalize(embedding []float32, p int) []float32 {
	denominator := max(float32(Norm(embedding, p)), 1e-12)
	inverseDenominator := 1 / denominator
	for i := range embedding {
		embedding[i] *= inverseDenominator
	}
	return embedding
}
