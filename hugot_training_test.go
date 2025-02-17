//go:build XLA || ALL

package hugot

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/train/losses"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/stretchr/testify/assert"

	"github.com/knights-analytics/hugot/datasets"
	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/pipelines"
)

func cosineSimilarity(x []float32, y []float32) float64 {
	var sum, s1, s2 float64
	for i := 0; i < len(x); i++ {
		sum += float64(x[i]) * float64(y[i])
		s1 += math.Pow(float64(x[i]), 2)
		s2 += math.Pow(float64(y[i]), 2)
	}
	if s1 == 0 || s2 == 0 {
		return 0.0
	}
	return sum / (math.Sqrt(s1) * math.Sqrt(s2))
}

func runModel(t *testing.T, runtime string, examplesLeft, examplesRight []string, modelPath string) []float64 {
	t.Helper()
	var session *Session
	var err error

	switch runtime {
	case "ORT":
		opts := []options.WithOption{
			options.WithOnnxLibraryPath(onnxRuntimeSharedLibrary),
		}
		session, err = NewORTSession(opts...)
		check(t, err)
	case "XLA":
		session, err = NewXLASession()
		check(t, err)
	default:
		t.Fatal("unknown runtime")
	}

	defer func() {
		check(t, session.Destroy())
	}()

	config := FeatureExtractionConfig{
		ModelPath:    modelPath,
		Name:         "testPipeline",
		OnnxFilename: "model.onnx",
	}
	pipeline, err := NewPipeline(session, config)
	check(t, err)

	resultsLeft, err := pipeline.RunPipeline(examplesLeft)
	check(t, err)
	resultsRight, err := pipeline.RunPipeline(examplesRight)
	check(t, err)

	// calculate cosine similarity between the embeddings
	var similarities []float64

	for i := 0; i < len(resultsLeft.Embeddings); i++ {
		similarities = append(similarities, cosineSimilarity(resultsLeft.Embeddings[i], resultsRight.Embeddings[i]))
	}
	return similarities
}

func round2decimals(x float64) float64 {
	return math.Round(x*100) / 100
}

func TestSemanticSimilarity(t *testing.T) {
	modelPath := "./models/KnightsAnalytics_all-MiniLM-L6-v2"

	// each line in this dataset is an example. In training we will use the dataset object but for inference
	// we just load the strings here.
	data, err := os.ReadFile("./testData/semanticSimilarityTest.jsonl")
	check(t, err)
	lines := bytes.Split(data, []byte("\n"))

	var examplesLeft []string
	var examplesRight []string
	var scores []float64

	for _, line := range lines {
		var example map[string]any
		err = json.Unmarshal(line, &example)
		if err != nil {
			t.Fatal(err)
		}
		examplesLeft = append(examplesLeft, example["sentence1"].(string))
		examplesRight = append(examplesRight, example["sentence2"].(string))
		scores = append(scores, example["score"].(float64))
	}

	// first we run the untrained onnx model with onnxruntime backend
	similaritiesOnnxruntime := runModel(t, "ORT", examplesLeft, examplesRight, modelPath)

	// we do the same for XLA and check the forward pass results match
	similaritiesXLA := runModel(t, "XLA", examplesLeft, examplesRight, modelPath)

	for i := range similaritiesOnnxruntime {
		assert.Equal(t, round2decimals(similaritiesOnnxruntime[i]), round2decimals(similaritiesXLA[i]))
	}

	// TRAINING: We now fine-tune the model

	// first we create a dataset object. This allows us to loop over the dataset for potentially multiple epochs.
	// We need to specify the batch size. For cpu lower batches seem to be faster.
	dataset, err := datasets.NewSemanticSimilarityDataset("./testData/semanticSimilarityTest.jsonl", 1)
	if err != nil {
		t.Fatal(err)
	}

	// create a new xla training session. Currently training is only possible by loading an onnx model
	// into xla, fine-tuning it with gomlx, and then writing it back to onnx. Hugot deals with the details
	// for you here.
	trainingSession, err := NewXLATrainingSession[*pipelines.FeatureExtractionPipeline](
		TrainingConfig{
			ModelPath: modelPath,
			Dataset:   dataset,
			Epochs:    1, // we just train for one epoch
			Cuda:      false,
			Verbose:   true,
		},
	)
	if err != nil {
		t.Fatal(err)
	}

	defer func() {
		check(t, trainingSession.Destroy())
	}()

	// train the model
	if e := trainingSession.Train(); e != nil {
		t.Fatal(e)
	}

	// we now write the fine-tuned pipeline back to disk as an onnx model.
	// This will also copy the tokenizer files for you. If your models are on s3
	// this can also work (see documentation).
	if e := trainingSession.Save("./models/testTrain"); e != nil {
		t.Fatal(e)
	}
	if _, err := os.Stat("./models/testTrain"); err != nil {
		t.Fatal(err)
	}

	defer func() {
		if err = os.RemoveAll("./models/testTrain"); err != nil {
			t.Fatal(err)
		}
	}()

	// we now load the newly trained onnx model and generate the predictions with onnxruntime backend
	similaritiesXLATrained := runModel(t, "ORT", examplesLeft, examplesRight, "./models/testTrain")

	fmt.Println("XLA trained model predictions:")
	for i := 0; i < len(similaritiesXLATrained); i++ {
		fmt.Printf("Example %d: untrained similarity %f, trained similarity %f, label %f\n", i, similaritiesXLA[i], similaritiesXLATrained[i], scores[i])
	}

	// on the training set, we expect the model to have improved
	rmseUntrained := rmse(similaritiesXLA, scores)
	rmseTrained := rmse(similaritiesXLATrained, scores)
	fmt.Printf("RMSE untrained is: %f\n", rmseUntrained)
	fmt.Printf("RMSE trained is: %f\n", rmseTrained)
	assert.Less(t, rmseTrained, rmseUntrained)
	assert.Equal(t, 0.06, round2decimals(rmseTrained))
}

func rmse(predictions []float64, labels []float64) float64 {
	var sum float64
	for i := 0; i < len(predictions); i++ {
		sum += math.Pow(predictions[i]-labels[i], 2)
	}
	return math.Sqrt(sum / float64(len(predictions)))
}

func TestSemanticSimilarityCuda(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}

	dataset, err := datasets.NewSemanticSimilarityDataset("./testData/semanticSimilarityTest.jsonl", 32)
	if err != nil {
		t.Fatal(err)
	}

	modelPath := "./models/KnightsAnalytics_all-MiniLM-L6-v2"

	session, err := NewXLATrainingSession[*pipelines.FeatureExtractionPipeline](
		TrainingConfig{
			ModelPath: modelPath,
			Dataset:   dataset,
			Epochs:    1,
			Cuda:      true, // use cuda acceleration
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

func TestCosineSimilarity(t *testing.T) {
	gradTest := func(x, y, label *graph.Node) (output, gradX, gradY *graph.Node) {
		sim := CosineSimilarity(x, y)
		output = losses.MeanSquaredError([]*graph.Node{label}, []*graph.Node{sim})
		grads := graph.Gradient(output, x, y)
		gradX, gradY = grads[0], grads[1]
		return output, gradX, gradY
	}

	exec := graph.NewExec(backends.New(), gradTest)
	x := tensors.FromFlatDataAndDimensions([]float32{0, 0, 0}, 1, 3)
	y := tensors.FromFlatDataAndDimensions([]float32{10, 20, 30}, 1, 3)
	label := tensors.FromFlatDataAndDimensions([]float32{1}, 1, 1)

	results := exec.Call(x, y, label)
	fmt.Printf("f(x=%v, y=%v)=%v,\n\tdf/dx=%v,\n\tdf/dy=%v\n", x, y, results[0].Value(), results[1].Value(), results[2].Value())
	d := tensors.CopyFlatData[float32](results[1])
	expected := []float32{0.022892393, 0, -0.022892393}
	for i := range d {
		assert.Equal(t, expected[i], d[i])
	}
}
