//go:build cgo && ((ORT && XLA) || ALL) && TRAINING

package training_test

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/options"
	testutil "github.com/knights-analytics/hugot/tests"

	"github.com/stretchr/testify/assert"

	"github.com/knights-analytics/hugot/datasets"
	"github.com/knights-analytics/hugot/pipelines"
)

func cosineSimilarityTester(x []float32, y []float32) float64 {
	var sum, s1, s2 float64
	for i := range x {
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
	var session *hugot.Session
	var err error

	switch runtime {
	case "ORT":
		session, err = hugot.NewORTSession(t.Context())
		testutil.CheckT(t, err)
	case "GO":
		session, err = hugot.NewGoSession(t.Context())
		testutil.CheckT(t, err)
	case "XLA":
		session, err = hugot.NewXLASession(t.Context(), options.WithGoMLXBatchBuckets([]int{35}))
		testutil.CheckT(t, err)
	default:
		t.Fatal("unknown runtime")
	}

	defer func() {
		testutil.CheckT(t, session.Destroy())
	}()

	config := hugot.FeatureExtractionConfig{
		ModelPath:    modelPath,
		Name:         "testPipeline",
		OnnxFilename: "model.onnx",
	}
	pipeline, err := hugot.NewPipeline(session, config)
	testutil.CheckT(t, err)

	resultsLeft, err := pipeline.RunPipeline(t.Context(), examplesLeft)
	testutil.CheckT(t, err)
	resultsRight, err := pipeline.RunPipeline(t.Context(), examplesRight)
	testutil.CheckT(t, err)

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
	config hugot.TrainerConfig[*pipelines.FeatureExtractionPipeline],
	trainerOptions []hugot.TrainerOption,
	examplesLHS,
	examplesRHS []string,
) []float64 {
	t.Helper()
	// Training always runs through GoMLX: an onnx model is loaded into GoMLX, fine-tuned, and
	// written back to onnx. An ORT session therefore has to be created with options.WithGoMLX().
	session, err := hugot.NewORTSession(t.Context(), options.WithGoMLX())
	if err != nil {
		t.Fatal(err)
	}

	// The trainer's model belongs to the session, so destroying the session is the only teardown
	// needed.
	defer func() {
		testutil.CheckT(t, session.Destroy())
	}()

	trainer, err := session.NewTrainer(config, trainerOptions...)
	if err != nil {
		t.Fatal(err)
	}

	// train the model
	if e := trainer.Train(t.Context()); e != nil {
		t.Fatal(e)
	}

	// we now write the fine-tuned pipeline back to disk as an onnx model.
	// This will also copy the tokenizer files for you. If your models are on s3
	// this can also work (see documentation).
	if e := trainer.Save(t.Context(), testutil.ModelsFolder+"testTrain"); e != nil {
		t.Fatal(e)
	}
	if _, err := os.Stat(testutil.ModelsFolder + "testTrain"); err != nil {
		t.Fatal(err)
	}

	defer func() {
		if err = os.RemoveAll(testutil.ModelsFolder + "testTrain"); err != nil {
			t.Fatal(err)
		}
	}()

	// we now load the newly trained onnx model and generate the predictions with onnxruntime backend
	return runModel(t, "ORT", examplesLHS, examplesRHS, testutil.ModelsFolder+"testTrain")
}

func TestTrainSemanticSimilarity(t *testing.T) {
	modelPath := testutil.ModelsFolder + "KnightsAnalytics_all-MiniLM-L6-v2"

	// each line in this dataset is an example. In training we will use the dataset object but for inference
	// we just load the strings here.
	data, err := os.ReadFile(testutil.TestCasesFolder + "semanticSimilarityTest.jsonl")
	testutil.CheckT(t, err)
	lines := bytes.Split(data, []byte("\n"))

	var examplesLHS []string
	var examplesRHS []string
	var scores []float64

	for _, line := range lines {
		var example map[string]any
		err = json.Unmarshal(line, &example)
		if err != nil {
			t.Fatal(err)
		}
		examplesLHS = append(examplesLHS, example["sentence1"].(string))
		examplesRHS = append(examplesRHS, example["sentence2"].(string))
		scores = append(scores, example["score"].(float64))
	}

	// first we run the untrained onnx model with onnxruntime backend
	similaritiesOnnxruntime := runModel(t, "ORT", examplesLHS, examplesRHS, modelPath)

	// we do the same for GoMLX and check the forward pass results match
	similaritiesGoMLX := runModel(t, "XLA", examplesLHS, examplesRHS, modelPath)

	for i := range similaritiesOnnxruntime {
		assert.Equal(t, round3decimals(similaritiesOnnxruntime[i]), round3decimals(similaritiesGoMLX[i]))
	}

	// TRAINING: We now fine-tune the model

	// first we create a train dataset object. This allows us to loop over the dataset for potentially multiple epochs.
	// We need to specify the batch size. For cpu lower batches seem to be faster.
	// The datasets.NewSemanticSimilarityDataset function also accepts a custom function that will be applied
	// to all examples in a batch before they are passed to the model. This can be used to apply whatever preprocessing
	// you need.
	trainDataset, err := datasets.NewSemanticSimilarityDataset(t.Context(), testutil.TestCasesFolder+"semanticSimilarityTest.jsonl", 1, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	// next we create a trainEvalDataset. This is the same as the train dataset, but it will be used to evaluate the model on
	// in-sample data at the end of each epoch.
	// We can also specify an eval dataset with early stopping (see test below).
	trainEvalDataset, err := datasets.NewSemanticSimilarityDataset(t.Context(), testutil.TestCasesFolder+"semanticSimilarityTest.jsonl", 1, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	// we now train the model with the dataset
	trainingConfig := hugot.TrainerConfig[*pipelines.FeatureExtractionPipeline]{
		ModelPath:        modelPath,
		TrainDataset:     trainDataset,
		TrainEvalDataset: trainEvalDataset,
		Verbose:          true,
	}
	trainerOptions := []hugot.TrainerOption{hugot.WithEpochs(2)}
	similaritiesGoMLXTrained := trainSimilarity(t, trainingConfig, trainerOptions, examplesLHS, examplesRHS)

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
	assert.Greater(t, 0.06, round3decimals(rmseTrained))
	assert.Less(t, 0.04, round3decimals(rmseTrained))

	// we can also train a model using an in memory dataset. For this we create the slice of examples manually.
	var examples []datasets.SemanticSimilarityExample
	for i := 0; i < len(examplesLHS); i++ {
		examples = append(examples, datasets.SemanticSimilarityExample{
			Sentence1: examplesLHS[i],
			Sentence2: examplesRHS[i],
			Score:     float32(scores[i]),
		})
	}
	inMemoryDataset, err := datasets.NewInMemorySemanticSimilarityDataset(examples, 1, nil)
	testutil.CheckT(t, err)
	trainingConfig.TrainDataset = inMemoryDataset
	similaritiesGoMLXTrainedInMemory := trainSimilarity(t, trainingConfig, trainerOptions, examplesLHS, examplesRHS)
	for i := range similaritiesGoMLXTrainedInMemory {
		assert.Equal(t, round3decimals(similaritiesGoMLXTrained[i]), round3decimals(similaritiesGoMLXTrainedInMemory[i]))
	}

	// we can also freeze layers
	frozenOptions := append(trainerOptions, hugot.WithFreezeLayers([]int{-1})) // freeze all layers but the last one
	similaritiesGoMLXTrainedFrozen := trainSimilarity(t, trainingConfig, frozenOptions, examplesLHS, examplesRHS)

	fmt.Println("GoMLX trained model predictions freezing all layers but the last one:")
	for i := range similaritiesGoMLXTrainedFrozen {
		fmt.Printf("Example %d: untrained similarity %f, trained similarity with frozen weights %f, label %f\n", i, similaritiesGoMLX[i], similaritiesGoMLXTrainedFrozen[i], scores[i])
	}
}

func rmse(predictions []float64, labels []float64) float64 {
	var sum float64
	for i := 0; i < len(predictions); i++ {
		sum += (predictions[i] - labels[i]) * (predictions[i] - labels[i])
	}
	return math.Sqrt(sum / float64(len(predictions)))
}

func TestTrainSemanticSimilarityCuda(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}

	dataset, err := datasets.NewSemanticSimilarityDataset(t.Context(), testutil.TestCasesFolder+"semanticSimilarityTest.jsonl", 32, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	modelPath := testutil.ModelsFolder + "KnightsAnalytics_all-MiniLM-L6-v2"

	// cuda is a property of the session, not of the training run
	session, err := hugot.NewXLASession(t.Context(), options.WithCuda(nil))
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		testutil.CheckT(t, session.Destroy())
	}()

	trainer, err := session.NewTrainer(
		hugot.TrainerConfig[*pipelines.FeatureExtractionPipeline]{
			ModelPath:    modelPath,
			TrainDataset: dataset,
			Verbose:      true,
		},
		hugot.WithEpochs(1),
	)
	if err != nil {
		t.Fatal(err)
	}

	// train the model
	if err = trainer.Train(t.Context()); err != nil {
		t.Fatal(err)
	}

	// we now write the fine-tuned pipeline back to disk as an onnx model
	if e := trainer.Save(t.Context(), testutil.ModelsFolder+"testTrain"); e != nil {
		t.Fatal(e)
	}
	if _, err := os.Stat(testutil.ModelsFolder + "testTrain"); err != nil {
		t.Fatal(err)
	}
	if err = os.RemoveAll(testutil.ModelsFolder + "testTrain"); err != nil {
		t.Fatal(err)
	}
}

func TestTrainSemanticSimilarityGo(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}

	dataset, err := datasets.NewSemanticSimilarityDataset(t.Context(), testutil.TestCasesFolder+"semanticSimilarityTest.jsonl", 1, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	modelPath := testutil.ModelsFolder + "KnightsAnalytics_all-MiniLM-L6-v2"

	session, err := hugot.NewGoSession(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		testutil.CheckT(t, session.Destroy())
	}()

	trainer, err := session.NewTrainer(
		hugot.TrainerConfig[*pipelines.FeatureExtractionPipeline]{
			ModelPath:    modelPath,
			TrainDataset: dataset,
			Verbose:      true,
		},
		hugot.WithEpochs(1),
	)
	if err != nil {
		t.Fatal(err)
	}

	// train the model
	if err = trainer.Train(t.Context()); err != nil {
		t.Fatal(err)
	}

	// we now write the fine-tuned pipeline back to disk as an onnx model
	if e := trainer.Save(t.Context(), testutil.ModelsFolder+"testTrain"); e != nil {
		t.Fatal(e)
	}
	if _, err := os.Stat(testutil.ModelsFolder + "testTrain"); err != nil {
		t.Fatal(err)
	}
	if err = os.RemoveAll(testutil.ModelsFolder + "testTrain"); err != nil {
		t.Fatal(err)
	}
}

func TestEarlyStopping(t *testing.T) {
	modelPath := testutil.ModelsFolder + "KnightsAnalytics_all-MiniLM-L6-v2"

	trainDataset, err := datasets.NewSemanticSimilarityDataset(t.Context(), testutil.TestCasesFolder+"semanticSimilarityTest.jsonl", 1, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	evalDataset, err := datasets.NewSemanticSimilarityDataset(t.Context(), testutil.TestCasesFolder+"semanticSimilarityTestEval.jsonl", 1, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	defer func() {
		err = os.RemoveAll(testutil.ModelsFolder + "testTrainEval")
		if err != nil {
			t.Fatal(err)
		}
	}()

	trainingConfig := hugot.TrainerConfig[*pipelines.FeatureExtractionPipeline]{
		ModelPath:    modelPath,
		TrainDataset: trainDataset,
		EvalDataset:  evalDataset,
		Verbose:      true,
	}

	session, err := hugot.NewXLASession(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		testutil.CheckT(t, session.Destroy())
	}()

	trainer, err := session.NewTrainer(trainingConfig, hugot.WithEarlyStoppingParams(2, 1e-4))
	if err != nil {
		t.Fatal(err)
	}

	// train the model
	if trainErr := trainer.Train(t.Context()); trainErr != nil {
		t.Fatal(trainErr)
	}

	// save the model
	if saveErr := trainer.Save(t.Context(), testutil.ModelsFolder+"testTrainEval"); saveErr != nil {
		t.Fatal(saveErr)
	}
}
