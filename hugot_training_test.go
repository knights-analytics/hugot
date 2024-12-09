//go:build XLA || ALL

package hugot

import (
	"testing"

	"github.com/knights-analytics/hugot/datasets"
	"github.com/knights-analytics/hugot/pipelines"
)

func TestSemanticSimilarity(t *testing.T) {
	dataset, err := datasets.NewSemanticSimilarityDataset("./data/train.jsonl", 32)
	if err != nil {
		t.Fatal(err)
	}
	session, err := NewXLATrainingSession[*pipelines.FeatureExtractionPipeline](
		TrainingConfig{
			ModelPath: "./models/distilbert-base-uncased",
			Dataset:   dataset,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if e := session.Train(); e != nil {
		t.Fatal(e)
	}
}

// func TestLogisticRegression(t *testing.T) {
// 	session, err := NewXLASession()
// 	check(t, err)
// 	defer func(session *Session) {
// 		destroyErr := session.Destroy()
// 		check(t, destroyErr)
// 	}(session)

// 	examples := [][]float64{
// 		{2.0, 3.0},
// 		{1.0, 0.5},
// 		{2.5, 4.5},
// 		{3.0, 2.0},
// 		{1.5, 1.0},
// 	}
// 	labels := []float64{0, 0, 1, 1, 0}

// 	config := LogisticRegressionConfig{
// 		Name: "testPipeline",
// 		Options: []LogisticRegressionOption{
// 			taskPipelines.WithCovariates(len(examples)),
// 		},
// 	}
// 	pipeline, err := NewPipeline(session, config)
// 	check(t, err)

// 	data := training.LogisticRegressionDataset{
// 		X: examples,
// 		Y: labels,
// 	}

// 	trainer, err := training.NewTrainer(pipeline, &data)
// 	if err != nil {
// 		t.Fatal(err)
// 	}

// 	if err := trainer.Train(); err != nil {
// 		t.Fatal(err)
// 	}

// 	// wf1 := weights[0]
// 	// wf2 := weights[1]
// 	// wb := weights[2]
// 	// var predictions []float64
// 	// for _, e := range examples {
// 	// 	predictions = append(predictions, 1/(1+math.Exp(-(wf1*e[0]+wf2*e[1]+wb))))
// 	// }
// 	// fmt.Println(predictions)
// }
