//go:build (ORT && XLA) || ALL

package hugot

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/knights-analytics/hugot/datasets"
	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/pipelines"
	"github.com/knights-analytics/hugot/util"
)

func cosineSimilarityTester(x []float32, y []float32) float64 {
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
		session, err = NewORTSession(options.WithOnnxLibraryPath(onnxRuntimeSharedLibrary))
		checkT(t, err)
	case "GO":
		session, err = NewGoSession()
		checkT(t, err)
	case "XLA":
		session, err = NewXLASession()
		checkT(t, err)
	default:
		t.Fatal("unknown runtime")
	}

	defer func() {
		checkT(t, session.Destroy())
	}()

	config := FeatureExtractionConfig{
		ModelPath:    modelPath,
		Name:         "testPipeline",
		OnnxFilename: "model.onnx",
	}
	pipeline, err := NewPipeline(session, config)
	checkT(t, err)

	resultsLeft, err := pipeline.RunPipeline(examplesLeft)
	checkT(t, err)
	resultsRight, err := pipeline.RunPipeline(examplesRight)
	checkT(t, err)

	// calculate cosine similarity between the embeddings
	var similarities []float64

	for i := 0; i < len(resultsLeft.Embeddings); i++ {
		similarities = append(similarities, cosineSimilarityTester(resultsLeft.Embeddings[i], resultsRight.Embeddings[i]))
	}
	return similarities
}

func round3decimals(x float64) float64 {
	return math.Round(x*1000) / 1000
}

func trainSimilarity(t *testing.T,
	config TrainingConfig,
	examplesLhs,
	examplesRhs []string) []float64 {
	// Create a new GoMLX training session. Currently, training is only possible by loading an onnx model
	// into GoMLX, fine-tuning it, and then writing it back to onnx. Hugot deals with the details
	// for you here.
	trainingSession, err := NewXLATrainingSession[*pipelines.FeatureExtractionPipeline](config)
	if err != nil {
		t.Fatal(err)
	}

	defer func() {
		checkT(t, trainingSession.Destroy())
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
	return runModel(t, "ORT", examplesLhs, examplesRhs, "./models/testTrain")
}

func TestTrainSemanticSimilarity(t *testing.T) {
	modelPath := "./models/KnightsAnalytics_all-MiniLM-L6-v2"

	// each line in this dataset is an example. In training we will use the dataset object but for inference
	// we just load the strings here.
	data, err := os.ReadFile("./testData/semanticSimilarityTest.jsonl")
	checkT(t, err)
	lines := bytes.Split(data, []byte("\n"))

	var examplesLhs []string
	var examplesRhs []string
	var scores []float64

	for _, line := range lines {
		var example map[string]any
		err = json.Unmarshal(line, &example)
		if err != nil {
			t.Fatal(err)
		}
		examplesLhs = append(examplesLhs, example["sentence1"].(string))
		examplesRhs = append(examplesRhs, example["sentence2"].(string))
		scores = append(scores, example["score"].(float64))
	}

	// first we run the untrained onnx model with onnxruntime backend
	similaritiesOnnxruntime := runModel(t, "ORT", examplesLhs, examplesRhs, modelPath)

	// we do the same for GoMLX and check the forward pass results match
	similaritiesGoMLX := runModel(t, "XLA", examplesLhs, examplesRhs, modelPath)

	for i := range similaritiesOnnxruntime {
		assert.Equal(t, round3decimals(similaritiesOnnxruntime[i]), round3decimals(similaritiesGoMLX[i]))
	}

	// TRAINING: We now fine-tune the model

	// first we create a dataset object. This allows us to loop over the dataset for potentially multiple epochs.
	// We need to specify the batch size. For cpu lower batches seem to be faster.
	// The datasets.NewSemanticSimilarityDataset function also accepts a custom function that will be applied
	// to all examples in a batch before they are passed to the model. This can be used to apply whatever preprocessing
	// you need.
	dataset, err := datasets.NewSemanticSimilarityDataset("./testData/semanticSimilarityTest.jsonl", 1, nil)
	if err != nil {
		t.Fatal(err)
	}

	// we now train the model with the dataset
	trainingConfig := TrainingConfig{
		ModelPath: modelPath,
		Dataset:   dataset,
		Options: []TrainingOption{
			WithEpochs(2),
		},
		Verbose: true,
	}
	similaritiesGoMLXTrained := trainSimilarity(t, trainingConfig, examplesLhs, examplesRhs)

	fmt.Println("GoMLX trained model predictions:")
	for i := range similaritiesGoMLXTrained {
		fmt.Printf("Example %d: untrained similarity %f, trained similarity %f, label %f\n", i, similaritiesGoMLX[i], similaritiesGoMLXTrained[i], scores[i])
	}

	// on the training set, we expect the model to have improved
	rmseUntrained := rmse(similaritiesGoMLX, scores)
	rmseTrained := rmse(similaritiesGoMLXTrained, scores)
	fmt.Printf("RMSE untrained is: %f\n", rmseUntrained)
	fmt.Printf("RMSE trained is: %f\n", rmseTrained)
	assert.Less(t, rmseTrained, rmseUntrained)
	assert.Equal(t, 0.047, round3decimals(rmseTrained))

	// we can also train a model using an in memory dataset. For this we create the slice of examples manually.
	var examples []datasets.SemanticSimilarityExample
	for i := 0; i < len(examplesLhs); i++ {
		examples = append(examples, datasets.SemanticSimilarityExample{
			Sentence1: examplesLhs[i],
			Sentence2: examplesRhs[i],
			Score:     float32(scores[i]),
		})
	}
	inMemoryDataset, err := datasets.NewInMemorySemanticSimilarityDataset(examples, 1, nil)
	checkT(t, err)
	trainingConfig.Dataset = inMemoryDataset
	similaritiesGoMLXTrainedInMemory := trainSimilarity(t, trainingConfig, examplesLhs, examplesRhs)
	for i := range similaritiesGoMLXTrainedInMemory {
		assert.Equal(t, round3decimals(similaritiesGoMLXTrained[i]), round3decimals(similaritiesGoMLXTrainedInMemory[i]))
	}

	// we can also train but freeze the embeddings and/or layers.
	trainingConfig.Options = append(trainingConfig.Options, WithFreezeLayers([]int{-1})) // freeze all layers but the last one
	similaritiesGoMLXTrainedFrozen := trainSimilarity(t, trainingConfig, examplesLhs, examplesRhs)

	fmt.Println("GoMLX trained model predictions freezing all layers but the last one:")
	for i := range similaritiesGoMLXTrainedFrozen {
		fmt.Printf("Example %d: untrained similarity %f, trained similarity with frozen weights %f, label %f\n", i, similaritiesGoMLX[i], similaritiesGoMLXTrainedFrozen[i], scores[i])
	}
}

func rmse(predictions []float64, labels []float64) float64 {
	var sum float64
	for i := 0; i < len(predictions); i++ {
		sum += math.Pow(predictions[i]-labels[i], 2)
	}
	return math.Sqrt(sum / float64(len(predictions)))
}

func TestTrainSemanticSimilarityCuda(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}

	dataset, err := datasets.NewSemanticSimilarityDataset("./testData/semanticSimilarityTest.jsonl", 32, nil)
	if err != nil {
		t.Fatal(err)
	}

	modelPath := "./models/KnightsAnalytics_all-MiniLM-L6-v2"

	session, err := NewXLATrainingSession[*pipelines.FeatureExtractionPipeline](
		TrainingConfig{
			ModelPath: modelPath,
			Dataset:   dataset,
			Options: []TrainingOption{
				WithEpochs(1),
				WithCuda(), // enable cuda
			},
			Verbose: true,
		},
	)
	if err != nil {
		t.Fatal(err)
	}

	// train the model
	if err = session.Train(); err != nil {
		t.Fatal(err)
	}

	// we now write the fine-tuned pipeline back to disk as an onnx model
	if e := session.Save("./models/testTrain"); e != nil {
		t.Fatal(e)
	}
	if exists, existsErr := util.FileExists("./models/testTrain"); existsErr != nil {
		t.Fatal(err)
	} else if !exists {
		t.Fatal("model file ./models/testTrain does not exist")
	}
	if err = util.DeleteFile("./models/testTrain"); err != nil {
		t.Fatal(err)
	}
}

func TestTrainSemanticSimilarityGo(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}

	dataset, err := datasets.NewSemanticSimilarityDataset("./testData/semanticSimilarityTest.jsonl", 1, nil)
	if err != nil {
		t.Fatal(err)
	}

	modelPath := "./models/KnightsAnalytics_all-MiniLM-L6-v2"

	session, err := NewGoTrainingSession[*pipelines.FeatureExtractionPipeline](
		TrainingConfig{
			ModelPath: modelPath,
			Dataset:   dataset,
			Options: []TrainingOption{
				WithEpochs(1),
			},
			Verbose: true,
		},
	)
	if err != nil {
		t.Fatal(err)
	}

	// train the model
	if err = session.Train(); err != nil {
		t.Fatal(err)
	}

	// we now write the fine-tuned pipeline back to disk as an onnx model
	if e := session.Save("./models/testTrain"); e != nil {
		t.Fatal(e)
	}
	if exists, existsErr := util.FileExists("./models/testTrain"); existsErr != nil {
		t.Fatal(err)
	} else if !exists {
		t.Fatal("model file ./models/testTrain does not exist")
	}
	if err = util.DeleteFile("./models/testTrain"); err != nil {
		t.Fatal(err)
	}
}

func TestTrainEval(t *testing.T) {
	modelPath := "./models/KnightsAnalytics_all-MiniLM-L6-v2"

	trainDataset, err := datasets.NewSemanticSimilarityDataset("./testData/semanticSimilarityTest.jsonl", 1, nil)
	if err != nil {
		t.Fatal(err)
	}
	evalDataset, err := datasets.NewSemanticSimilarityDataset("./testData/semanticSimilarityTestEval.jsonl", 1, nil)
	if err != nil {
		t.Fatal(err)
	}

	trainingConfig := TrainingConfig{
		ModelPath:   modelPath,
		Dataset:     trainDataset,
		EvalDataset: evalDataset,
		Options: []TrainingOption{
			WithEarlyStoppingParams(2, 1e-4),
		},
		Verbose: true,
	}

	trainingSession, err := NewXLATrainingSession[*pipelines.FeatureExtractionPipeline](trainingConfig)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		checkT(t, trainingSession.Destroy())
	}()

	// train the model
	if trainErr := trainingSession.Train(); trainErr != nil {
		t.Fatal(trainErr)
	}
}
