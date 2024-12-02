package util

import (
	"fmt"
	"math"
	"testing"
)

func TestLogisticWeights(t *testing.T) {
	examples := [][]float64{
		{2.0, 3.0},
		{1.0, 0.5},
		{2.5, 4.5},
		{3.0, 2.0},
		{1.5, 1.0},
	}
	labels := []float64{0, 0, 1, 1, 0}

	weights, err := LogisticRegressionWeights(examples, labels)
	if err != nil {
		t.Fatal(err)
	}
	fmt.Println(weights)

	wf1 := weights[0]
	wf2 := weights[1]
	wb := weights[2]
	var predictions []float64
	for _, e := range examples {
		predictions = append(predictions, 1/(1+math.Exp(-(wf1*e[0]+wf2*e[1]+wb))))
	}
	fmt.Println(predictions)
}
