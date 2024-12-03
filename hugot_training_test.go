//go:build XLA || ALL

package hugot

import (
	"os"
	"testing"

	"github.com/knights-analytics/hugot/datasets"
	"github.com/knights-analytics/hugot/pipelines"
)

func TestSemanticSimilarity(t *testing.T) {
	// create a new dataset to fine-tune semantic similarity of embeddings
	// you need to specify the batch size for the dataset.
	// For cpu low batch sizes seem to perform best.
	dataset, err := datasets.NewSemanticSimilarityDataset("./testData/semanticSimilarityTest.jsonl", 1)
	if err != nil {
		t.Fatal(err)
	}

	// create a new xla training session. Currently training is only possible by loading an onnx model
	// into xla, fine-tuning it with gomlx, and then writing it back to onnx. Hugot deals with the details
	// for you here.
	session, err := NewXLATrainingSession[*pipelines.FeatureExtractionPipeline](
		TrainingConfig{
			ModelPath: "./models/sentence-transformers_all-MiniLM-L6-v2",
			Dataset:   dataset,
			Epochs:    1,
			Cuda:      false,
			Verbose:   true,
		},
	)
	if err != nil {
		t.Fatal(err)
	}

	// train the model
	if e := session.Train(); e != nil {
		t.Fatal(e)
	}

	// we now write the fine-tuned pipeline back to disk as an onnx model
	if e := session.Save("./models/testTrain.onnx"); e != nil {
		t.Fatal(e)
	}
	if _, err := os.Stat("./models/testTrain.onnx"); err != nil {
		t.Fatal(err)
	}
	if err = os.Remove("./models/testTrain.onnx"); err != nil {
		t.Fatal(err)
	}
}

func TestSemanticSimilarityCuda(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}

	dataset, err := datasets.NewSemanticSimilarityDataset("./testData/semanticSimilarityTest.jsonl", 32)
	if err != nil {
		t.Fatal(err)
	}

	session, err := NewXLATrainingSession[*pipelines.FeatureExtractionPipeline](
		TrainingConfig{
			ModelPath: "./models/sentence-transformers_all-MiniLM-L6-v2",
			Dataset:   dataset,
			Epochs:    1,
			Cuda:      true, // use cuda
			Verbose:   true,
		},
	)
	if err != nil {
		t.Fatal(err)
	}

	// train the model
	if e := session.Train(); e != nil {
		t.Fatal(e)
	}

	// we now write the fine-tuned pipeline back to disk as an onnx model
	if e := session.Save("./models/testTrain.onnx"); e != nil {
		t.Fatal(e)
	}
	if _, err := os.Stat("./models/testTrain.onnx"); err != nil {
		t.Fatal(err)
	}
	if err = os.Remove("./models/testTrain.onnx"); err != nil {
		t.Fatal(err)
	}
}
