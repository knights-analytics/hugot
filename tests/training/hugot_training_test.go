//go:build cgo && ((ORT && XLA) || ALL) && TRAINING

package training_test

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/options"
	testutil "github.com/knights-analytics/hugot/tests"

	"github.com/gomlx/gomlx/ml/train/optimizer"
	"github.com/stretchr/testify/assert"

	"github.com/knights-analytics/hugot/datasets"
	"github.com/knights-analytics/hugot/evaluation"
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
	pipeline, err := session.NewPipeline(config)
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
	defer func() {
		testutil.CheckT(t, trainer.Close())
	}()

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
	defer func() {
		testutil.CheckT(t, trainer.Close())
	}()

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
	defer func() {
		testutil.CheckT(t, trainer.Close())
	}()

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
	defer func() {
		testutil.CheckT(t, trainer.Close())
	}()

	// train the model
	if trainErr := trainer.Train(t.Context()); trainErr != nil {
		t.Fatal(trainErr)
	}

	// save the model
	if saveErr := trainer.Save(t.Context(), testutil.ModelsFolder+"testTrainEval"); saveErr != nil {
		t.Fatal(saveErr)
	}
}

// An untrainable pipeline type must be rejected before anything is loaded. The model path does not
// exist, so reaching the loader would fail with a different error; and the session must hold no model.
func TestNewTrainerRejectsUntrainablePipeline(t *testing.T) {
	dataset, err := datasets.NewSemanticSimilarityDataset(t.Context(), testutil.TestCasesFolder+"semanticSimilarityTest.jsonl", 1, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	session, err := hugot.NewGoSession(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		testutil.CheckT(t, session.Destroy())
	}()

	_, err = session.NewTrainer(hugot.TrainerConfig[*pipelines.FillMaskPipeline]{
		ModelPath:    "does/not/exist",
		TrainDataset: dataset,
	})
	assert.ErrorContains(t, err, "training is not supported for pipeline type")
	assert.Empty(t, session.GetModels())
}

// An ORT session without options.WithGoMLX cannot train, and must be rejected before a native ORT
// model is loaded. The model path does not exist, so reaching the loader would fail differently.
func TestNewTrainerRejectsORTWithoutGoMLX(t *testing.T) {
	dataset, err := datasets.NewSemanticSimilarityDataset(t.Context(), testutil.TestCasesFolder+"semanticSimilarityTest.jsonl", 1, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	session, err := hugot.NewORTSession(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		testutil.CheckT(t, session.Destroy())
	}()

	_, err = session.NewTrainer(hugot.TrainerConfig[*pipelines.FeatureExtractionPipeline]{
		ModelPath:    "does/not/exist",
		TrainDataset: dataset,
	})
	assert.ErrorContains(t, err, "options.WithGoMLX()")
	assert.Empty(t, session.GetModels())
}

// Close releases the trainer's model and removes it from the session, and a closed trainer can no
// longer train or save. Nothing is trained here, so this runs on the Go backend.
func TestTrainerClose(t *testing.T) {
	dataset, err := datasets.NewSemanticSimilarityDataset(t.Context(), testutil.TestCasesFolder+"semanticSimilarityTest.jsonl", 1, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	session, err := hugot.NewGoSession(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		testutil.CheckT(t, session.Destroy())
	}()

	trainer, err := session.NewTrainer(hugot.TrainerConfig[*pipelines.FeatureExtractionPipeline]{
		ModelPath:    testutil.ModelsFolder + "KnightsAnalytics_all-MiniLM-L6-v2",
		TrainDataset: dataset,
	})
	if err != nil {
		t.Fatal(err)
	}
	assert.Len(t, session.GetModels(), 1)

	assert.NoError(t, trainer.Close())
	assert.Empty(t, session.GetModels())
	assert.ErrorContains(t, trainer.Train(t.Context()), "trainer has been closed")
	assert.ErrorContains(t, trainer.Save(t.Context(), t.TempDir()), "trainer has been closed")
	assert.NoError(t, trainer.Close(), "closing twice is a no-op")
}

// The runners mirror those of the pipeline tests: XLA runs in CI, CUDA needs a GPU and so runs
// locally only, and the Go backend runs locally only because training on it is slow.
func runXLATraining(t *testing.T, run func(*testing.T, *hugot.Session), opts ...options.WithOption) {
	t.Helper()
	session, err := hugot.NewXLASession(t.Context(), opts...)
	testutil.CheckT(t, err)
	defer func() { testutil.CheckT(t, session.Destroy()) }()
	run(t, session)
}

func runXLATrainingCuda(t *testing.T, run func(*testing.T, *hugot.Session)) {
	t.Helper()
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	runXLATraining(t, run, options.WithCuda(map[string]string{"device_id": "0"}))
}

func runGoTraining(t *testing.T, run func(*testing.T, *hugot.Session)) {
	t.Helper()
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	session, err := hugot.NewGoSession(t.Context())
	testutil.CheckT(t, err)
	defer func() { testutil.CheckT(t, session.Destroy()) }()
	run(t, session)
}

const sst2ModelPath = testutil.ModelsFolder + "Xenova_distilbert-base-uncased-finetuned-sst-2-english"

// readTextClassificationExamples reads a text classification .jsonl file.
func readTextClassificationExamples(t *testing.T, path string) []datasets.TextClassificationExample {
	t.Helper()
	data, err := os.ReadFile(path)
	testutil.CheckT(t, err)
	var examples []datasets.TextClassificationExample
	for _, line := range bytes.Split(bytes.TrimSpace(data), []byte("\n")) {
		var example datasets.TextClassificationExample
		testutil.CheckT(t, json.Unmarshal(line, &example))
		examples = append(examples, example)
	}
	return examples
}

// classificationAccuracy runs a text classification model for inference and returns its accuracy
// on the examples' labels.
func classificationAccuracy(t *testing.T, session *hugot.Session, modelPath, onnxFilename string, examples []datasets.TextClassificationExample) float64 {
	t.Helper()
	pipeline, err := session.NewPipeline(hugot.TextClassificationConfig{
		ModelPath:    modelPath,
		OnnxFilename: onnxFilename,
		Name:         fmt.Sprintf("accuracy:%s", modelPath),
	})
	testutil.CheckT(t, err)
	texts := make([]string, len(examples))
	for i, example := range examples {
		texts[i] = example.Text
	}
	output, err := pipeline.RunPipeline(t.Context(), texts)
	testutil.CheckT(t, err)
	correct := 0
	for i, example := range examples {
		if output.ClassificationOutputs[i][0].Label == example.Label {
			correct++
		}
	}
	return float64(correct) / float64(len(examples))
}

// Fine-tunes the SST-2 sentiment model on deliberately inverted labels. The original model gets
// almost all of them wrong, so the saved model only scores well if training changed its weights,
// and if Save wrote a model the inference pipeline can load, labels included.
func textClassificationTraining(t *testing.T, session *hugot.Session) {
	trainPath := testutil.TestCasesFolder + "textClassificationInvertedSentiment.jsonl"
	examples := readTextClassificationExamples(t, trainPath)

	before := classificationAccuracy(t, session, sst2ModelPath, "", examples)
	assert.LessOrEqual(t, before, 0.25, "the original model should disagree with the inverted labels")

	trainDataset, err := datasets.NewTextClassificationDataset(t.Context(), trainPath, 4, datasets.SingleLabel, nil, nil)
	testutil.CheckT(t, err)
	trainEvalDataset, err := datasets.NewTextClassificationDataset(t.Context(), trainPath, 8, datasets.SingleLabel, nil, nil)
	testutil.CheckT(t, err)

	// The default learning rate suits real datasets of hundreds of steps; this one has 4 steps an
	// epoch, so a higher rate keeps the test short. It also exercises overriding the optimizer.
	const epochs = 6
	trainer, err := session.NewTrainer(hugot.TrainerConfig[*pipelines.TextClassificationPipeline]{
		ModelPath:        sst2ModelPath,
		TrainDataset:     trainDataset,
		TrainEvalDataset: trainEvalDataset,
		GOMLXOptions: &hugot.GOMLXTrainingOptions{
			Optimizer: optimizer.Adam().LearningRate(2e-4).Done(),
		},
	}, hugot.WithEpochs(epochs))
	testutil.CheckT(t, err)
	defer func() { testutil.CheckT(t, trainer.Close()) }()
	testutil.CheckT(t, trainer.Train(t.Context()))

	stats := trainer.Statistics()
	t.Logf("accuracy before training %.2f; per epoch losses %v, accuracies %v", before, stats.EpochTrainLosses, stats.EpochTrainAccuracies)
	if assert.Len(t, stats.EpochTrainAccuracies, epochs) {
		assert.Less(t, stats.EpochTrainLosses[epochs-1], stats.EpochTrainLosses[0], "the loss should fall")
	}

	savePath := t.TempDir()
	testutil.CheckT(t, trainer.Save(t.Context(), savePath))
	_, err = os.Stat(savePath + "/config.json")
	testutil.CheckT(t, err)

	after := classificationAccuracy(t, session, savePath, "model.onnx", examples)
	t.Logf("accuracy of the saved model %.2f", after)
	assert.GreaterOrEqual(t, after, 0.9, "the fine-tuned model should have learned the inverted labels")
	// The last epoch's evaluation ran on the final weights, so it must agree with the saved model.
	assert.InDelta(t, after, stats.EpochTrainAccuracies[epochs-1], 1e-6, "evaluation must see the current weights")

	// Macro-F1 per epoch, cross-checked against scoring the saved model independently.
	if assert.Len(t, stats.EpochTrainF1, epochs) {
		pipeline, err := session.NewPipeline(hugot.TextClassificationConfig{ModelPath: savePath, OnnxFilename: "model.onnx", Name: "macroF1"})
		testutil.CheckT(t, err)
		report, err := evaluation.EvaluateTextClassification(t.Context(), pipeline, examples, 8)
		testutil.CheckT(t, err)
		t.Logf("macro-F1 per epoch %v; saved model %.2f", stats.EpochTrainF1, report.MacroF1)
		assert.InDelta(t, report.MacroF1, stats.EpochTrainF1[epochs-1], 1e-6, "the per-epoch macro-F1 must see the current weights")
		assert.GreaterOrEqual(t, report.MacroF1, 0.9)
	}
}

func TestTrainTextClassificationXLA(t *testing.T) {
	runXLATraining(t, textClassificationTraining)
}

func TestTrainTextClassificationXLACuda(t *testing.T) {
	runXLATrainingCuda(t, textClassificationTraining)
}

func TestTrainTextClassificationGo(t *testing.T) {
	runGoTraining(t, textClassificationTraining)
}

// A multi-label dataset trains with a per-label sigmoid loss and multi-hot labels.
func textClassificationMultiLabelTraining(t *testing.T, session *hugot.Session) {
	dataset, err := datasets.NewInMemoryTextClassificationDataset([]datasets.TextClassificationExample{
		{Text: "Wonderful and awful at the same time.", Labels: []string{"NEGATIVE", "POSITIVE"}},
		{Text: "A lovely day.", Labels: []string{"POSITIVE"}},
		{Text: "Nothing to say about it.", Labels: nil},
		{Text: "Dreadful.", Labels: []string{"NEGATIVE"}},
	}, 2, datasets.MultiLabel, nil)
	testutil.CheckT(t, err)

	trainer, err := session.NewTrainer(hugot.TrainerConfig[*pipelines.TextClassificationPipeline]{
		ModelPath:        sst2ModelPath,
		TrainDataset:     dataset,
		TrainEvalDataset: dataset,
	}, hugot.WithEpochs(2))
	testutil.CheckT(t, err)
	defer func() { testutil.CheckT(t, trainer.Close()) }()
	testutil.CheckT(t, trainer.Train(t.Context()))

	stats := trainer.Statistics()
	assert.Len(t, stats.EpochTrainLosses, 2)
	assert.Len(t, stats.EpochTrainF1, 2, "multi-label training records macro-F1 too")
	for _, l := range stats.EpochTrainLosses {
		assert.False(t, math.IsNaN(float64(l)) || math.IsInf(float64(l), 0), "loss must be finite, got %v", l)
	}
}

func TestTrainTextClassificationMultiLabelXLA(t *testing.T) {
	runXLATraining(t, textClassificationMultiLabelTraining)
}

func TestTrainTextClassificationMultiLabelXLACuda(t *testing.T) {
	runXLATrainingCuda(t, textClassificationMultiLabelTraining)
}

func TestTrainTextClassificationMultiLabelGo(t *testing.T) {
	runGoTraining(t, textClassificationMultiLabelTraining)
}

// A text classification trainer only accepts a text classification dataset, and says so.
func TestTrainTextClassificationRejectsWrongDataset(t *testing.T) {
	dataset, err := datasets.NewSemanticSimilarityDataset(t.Context(), testutil.TestCasesFolder+"semanticSimilarityTest.jsonl", 1, nil, nil)
	testutil.CheckT(t, err)
	session, err := hugot.NewGoSession(t.Context())
	testutil.CheckT(t, err)
	defer func() { testutil.CheckT(t, session.Destroy()) }()

	_, err = session.NewTrainer(hugot.TrainerConfig[*pipelines.TextClassificationPipeline]{
		ModelPath:    sst2ModelPath,
		TrainDataset: dataset,
	})
	assert.ErrorContains(t, err, "tokenization pipeline must be a FeatureExtractionPipeline")
	assert.Empty(t, session.GetModels())
}

const nerModelPath = testutil.ModelsFolder + "KnightsAnalytics_distilbert-NER"

func readTokenClassificationExamples(t *testing.T, path string) []datasets.TokenClassificationExample {
	t.Helper()
	data, err := os.ReadFile(path)
	testutil.CheckT(t, err)
	var examples []datasets.TokenClassificationExample
	for _, line := range bytes.Split(bytes.TrimSpace(data), []byte("\n")) {
		var example datasets.TokenClassificationExample
		testutil.CheckT(t, json.Unmarshal(line, &example))
		examples = append(examples, example)
	}
	return examples
}

// wordTagAccuracy runs a token classification model for inference and returns the share of words
// whose first token gets the gold tag, over all words and over the words tagged focusTag.
func wordTagAccuracy(t *testing.T, session *hugot.Session, modelPath string, examples []datasets.TokenClassificationExample, focusTag string) (all, focus float64) {
	t.Helper()
	pipeline, err := session.NewPipeline(hugot.TokenClassificationConfig{
		ModelPath: modelPath,
		Name:      fmt.Sprintf("accuracy:%s", modelPath),
		// Without aggregation the pipeline returns one entity per token, but drops "O" tokens.
		Options: []hugot.TokenClassificationOption{pipelines.WithoutAggregation()},
	})
	testutil.CheckT(t, err)
	texts := make([]string, len(examples))
	for i, example := range examples {
		texts[i] = strings.Join(example.Tokens, " ")
	}
	output, err := pipeline.RunPipeline(t.Context(), texts)
	testutil.CheckT(t, err)
	var correct, total, focusCorrect, focusTotal int
	for i, example := range examples {
		predicted := map[uint]string{} // token start offset -> tag
		for _, entity := range output.Entities[i] {
			if _, seen := predicted[entity.Start]; !seen {
				predicted[entity.Start] = entity.Entity
			}
		}
		var start uint
		for w, word := range example.Tokens {
			tag, found := predicted[start]
			if !found {
				tag = "O" // dropped by the pipeline
			}
			hit := tag == example.Tags[w]
			total++
			if hit {
				correct++
			}
			if example.Tags[w] == focusTag {
				focusTotal++
				if hit {
					focusCorrect++
				}
			}
			start += uint(len(word)) + 1
		}
	}
	return float64(correct) / float64(total), float64(focusCorrect) / float64(focusTotal)
}

// Fine-tunes the NER model to tag cities as organisations. The original model tags them as
// locations, so the saved model only gets them right if training changed its weights; the other
// words keep their usual tags, so overall accuracy shows the rest of the model survived.
func tokenClassificationTraining(t *testing.T, session *hugot.Session) {
	trainPath := testutil.TestCasesFolder + "tokenClassificationCitiesAsOrg.jsonl"
	examples := readTokenClassificationExamples(t, trainPath)

	allBefore, citiesBefore := wordTagAccuracy(t, session, nerModelPath, examples, "B-ORG")
	assert.LessOrEqual(t, citiesBefore, 0.2, "the original model should tag the cities as locations")
	reportBefore := entityReport(t, session, nerModelPath, examples)
	assert.Zero(t, reportBefore.PerType["ORG"].Correct, "no city is an ORG for the original model")

	trainDataset, err := datasets.NewTokenClassificationDataset(t.Context(), trainPath, 4, nil, nil)
	testutil.CheckT(t, err)
	trainEvalDataset, err := datasets.NewTokenClassificationDataset(t.Context(), trainPath, 12, nil, nil)
	testutil.CheckT(t, err)
	// Given without early stopping: it must still be evaluated every epoch.
	evalDataset, err := datasets.NewTokenClassificationDataset(t.Context(), trainPath, 12, nil, nil)
	testutil.CheckT(t, err)

	const epochs = 6
	trainer, err := session.NewTrainer(hugot.TrainerConfig[*pipelines.TokenClassificationPipeline]{
		ModelPath:        nerModelPath,
		TrainDataset:     trainDataset,
		TrainEvalDataset: trainEvalDataset,
		EvalDataset:      evalDataset,
		GOMLXOptions: &hugot.GOMLXTrainingOptions{
			Optimizer: optimizer.Adam().LearningRate(1.5e-5).Done(),
		},
	}, hugot.WithEpochs(epochs))
	testutil.CheckT(t, err)
	defer func() { testutil.CheckT(t, trainer.Close()) }()
	testutil.CheckT(t, trainer.Train(t.Context()))

	stats := trainer.Statistics()
	t.Logf("before training: words %.2f, cities %.2f; per epoch losses %v, tagged-token accuracies %v", allBefore, citiesBefore, stats.EpochTrainLosses, stats.EpochTrainAccuracies)
	t.Logf("entity F1 before %.2f; per epoch on train eval %v, on eval %v", reportBefore.F1, stats.EpochTrainF1, stats.EpochEvalF1)
	if assert.Len(t, stats.EpochTrainAccuracies, epochs) {
		assert.Less(t, stats.EpochTrainLosses[epochs-1], stats.EpochTrainLosses[0], "the loss should fall")
	}
	assert.Len(t, stats.EpochEvalLosses, epochs, "EvalDataset must be evaluated without early stopping")
	assert.Len(t, stats.EpochEvalF1, epochs)
	if assert.Len(t, stats.EpochTrainF1, epochs) {
		assert.Greater(t, stats.EpochTrainF1[epochs-1], float32(reportBefore.F1), "entity F1 should rise")
	}

	savePath := t.TempDir()
	testutil.CheckT(t, trainer.Save(t.Context(), savePath))
	reportAfter := entityReport(t, session, savePath, examples)
	t.Logf("saved model entity F1 %.2f, ORG %.2f, PER %.2f", reportAfter.F1, reportAfter.PerType["ORG"].F1, reportAfter.PerType["PER"].F1)
	assert.GreaterOrEqual(t, reportAfter.F1, 0.9)
	assert.GreaterOrEqual(t, reportAfter.PerType["ORG"].F1, 0.9, "the cities should now be ORG entities")
	// The trainer's last entity F1 ran inference on the final weights, so it must match.
	assert.InDelta(t, reportAfter.F1, stats.EpochTrainF1[epochs-1], 1e-6, "the per-epoch entity F1 must see the current weights")
	allAfter, citiesAfter := wordTagAccuracy(t, session, savePath, examples, "B-ORG")
	t.Logf("saved model: words %.2f, cities %.2f", allAfter, citiesAfter)
	assert.GreaterOrEqual(t, citiesAfter, 0.9, "the fine-tuned model should tag the cities as organisations")
	assert.GreaterOrEqual(t, allAfter, 0.9, "the other words should keep their tags")
	// The last evaluation ran on the final weights, over the same tagged tokens.
	assert.InDelta(t, allAfter, stats.EpochTrainAccuracies[epochs-1], 1e-6, "evaluation must see the current weights")
}

func TestTrainTokenClassificationXLA(t *testing.T) {
	runXLATraining(t, tokenClassificationTraining)
}

func TestTrainTokenClassificationXLACuda(t *testing.T) {
	runXLATrainingCuda(t, tokenClassificationTraining)
}

func TestTrainTokenClassificationGo(t *testing.T) {
	runGoTraining(t, tokenClassificationTraining)
}

// entityReport scores a saved or original token classification model with the evaluation package.
func entityReport(t *testing.T, session *hugot.Session, modelPath string, examples []datasets.TokenClassificationExample) evaluation.EntityReport {
	t.Helper()
	pipeline, err := session.NewPipeline(hugot.TokenClassificationConfig{
		ModelPath: modelPath,
		Name:      fmt.Sprintf("entities:%s", modelPath),
	})
	testutil.CheckT(t, err)
	report, err := evaluation.EvaluateTokenClassification(t.Context(), pipeline, examples, 8)
	testutil.CheckT(t, err)
	return report
}

// Early stopping that actually stops. A tolerance far above any real improvement makes every epoch
// after the first count as no improvement, so with patience 1 training must stop after the second
// epoch, well short of the maximum.
func earlyStoppingStopsTraining(t *testing.T, session *hugot.Session) {
	trainDataset, err := datasets.NewSemanticSimilarityDataset(t.Context(), testutil.TestCasesFolder+"semanticSimilarityTest.jsonl", 1, nil, nil)
	testutil.CheckT(t, err)
	evalDataset, err := datasets.NewSemanticSimilarityDataset(t.Context(), testutil.TestCasesFolder+"semanticSimilarityTestEval.jsonl", 1, nil, nil)
	testutil.CheckT(t, err)

	trainer, err := session.NewTrainer(hugot.TrainerConfig[*pipelines.FeatureExtractionPipeline]{
		ModelPath:    testutil.ModelsFolder + "KnightsAnalytics_all-MiniLM-L6-v2",
		TrainDataset: trainDataset,
		EvalDataset:  evalDataset,
	}, hugot.WithEpochs(5), hugot.WithEarlyStoppingParams(1, 10))
	testutil.CheckT(t, err)
	defer func() { testutil.CheckT(t, trainer.Close()) }()
	testutil.CheckT(t, trainer.Train(t.Context()))

	assert.Len(t, trainer.Statistics().EpochEvalLosses, 2, "training should stop after the second epoch")
}

func TestEarlyStoppingStopsTrainingXLA(t *testing.T) {
	runXLATraining(t, earlyStoppingStopsTraining)
}

func TestEarlyStoppingStopsTrainingXLACuda(t *testing.T) {
	runXLATrainingCuda(t, earlyStoppingStopsTraining)
}

func TestEarlyStoppingStopsTrainingGo(t *testing.T) {
	runGoTraining(t, earlyStoppingStopsTraining)
}

// retrievalAccuracy embeds the anchors and positives with a feature extraction model and returns
// the share of anchors whose most similar positive is their own.
func retrievalAccuracy(t *testing.T, session *hugot.Session, modelPath, onnxFilename string, examples []datasets.ContrastiveExample) float64 {
	t.Helper()
	pipeline, err := session.NewPipeline(hugot.FeatureExtractionConfig{
		ModelPath:    modelPath,
		OnnxFilename: onnxFilename,
		Name:         fmt.Sprintf("retrieval:%s", modelPath),
	})
	testutil.CheckT(t, err)
	anchors := make([]string, len(examples))
	positives := make([]string, len(examples))
	for i, example := range examples {
		anchors[i], positives[i] = example.Anchor, example.Positive
	}
	anchorEmbeddings, err := pipeline.RunPipeline(t.Context(), anchors)
	testutil.CheckT(t, err)
	positiveEmbeddings, err := pipeline.RunPipeline(t.Context(), positives)
	testutil.CheckT(t, err)
	correct := 0
	for i, anchor := range anchorEmbeddings.Embeddings {
		best, bestScore := -1, math.Inf(-1)
		for j, positive := range positiveEmbeddings.Embeddings {
			if score := cosineSimilarityTester(anchor, positive); score > bestScore {
				best, bestScore = j, score
			}
		}
		if best == i {
			correct++
		}
	}
	return float64(correct) / float64(len(examples))
}

// Fine-tunes an embedding model with in-batch negatives on deliberately arbitrary pairs (fruit
// sentences paired with animal sentences). The original model can only match them by chance, so
// the saved model only retrieves them well if training changed its weights.
func contrastiveTraining(t *testing.T, session *hugot.Session) {
	trainPath := testutil.TestCasesFolder + "contrastiveArbitraryPairs.jsonl"
	modelPath := testutil.ModelsFolder + "KnightsAnalytics_all-MiniLM-L6-v2"

	trainDataset, err := datasets.NewContrastiveDataset(t.Context(), trainPath, 4, nil, nil)
	testutil.CheckT(t, err)
	// One batch of all 8 pairs, so the in-batch accuracy is retrieval accuracy over the whole set.
	trainEvalDataset, err := datasets.NewContrastiveDataset(t.Context(), trainPath, 8, nil, nil)
	testutil.CheckT(t, err)
	examples, err := trainEvalDataset.Examples()
	testutil.CheckT(t, err)

	before := retrievalAccuracy(t, session, modelPath, "model.onnx", examples)
	assert.LessOrEqual(t, before, 0.5, "the original model should not know the arbitrary pairs")

	const epochs = 8
	trainer, err := session.NewTrainer(hugot.TrainerConfig[*pipelines.FeatureExtractionPipeline]{
		ModelPath:        modelPath,
		TrainDataset:     trainDataset,
		TrainEvalDataset: trainEvalDataset,
		GOMLXOptions: &hugot.GOMLXTrainingOptions{
			Optimizer: optimizer.Adam().LearningRate(1e-4).Done(),
		},
	}, hugot.WithEpochs(epochs))
	testutil.CheckT(t, err)
	defer func() { testutil.CheckT(t, trainer.Close()) }()
	testutil.CheckT(t, trainer.Train(t.Context()))

	stats := trainer.Statistics()
	t.Logf("retrieval before %.2f; per epoch losses %v, in-batch accuracies %v", before, stats.EpochTrainLosses, stats.EpochTrainAccuracies)
	if assert.Len(t, stats.EpochTrainAccuracies, epochs) {
		assert.Less(t, stats.EpochTrainLosses[epochs-1], stats.EpochTrainLosses[0], "the loss should fall")
	}

	savePath := t.TempDir()
	testutil.CheckT(t, trainer.Save(t.Context(), savePath))
	after := retrievalAccuracy(t, session, savePath, "model.onnx", examples)
	t.Logf("retrieval of the saved model %.2f", after)
	assert.GreaterOrEqual(t, after, 0.9, "the fine-tuned model should retrieve the trained pairs")
	assert.InDelta(t, after, stats.EpochTrainAccuracies[epochs-1], 1e-6, "in-batch accuracy over one batch of everything is retrieval accuracy")
}

func TestTrainContrastiveXLA(t *testing.T) {
	runXLATraining(t, contrastiveTraining)
}

func TestTrainContrastiveXLACuda(t *testing.T) {
	runXLATrainingCuda(t, contrastiveTraining)
}

func TestTrainContrastiveGo(t *testing.T) {
	runGoTraining(t, contrastiveTraining)
}

// A contrastive dataset needs at least two examples per batch to have negatives at all.
func TestContrastiveDatasetRejectsBatchOfOne(t *testing.T) {
	_, err := datasets.NewInMemoryContrastiveDataset([]datasets.ContrastiveExample{{Anchor: "a", Positive: "b"}}, 1, nil)
	assert.ErrorContains(t, err, "at least 2")
}

// splitTrailingDot splits on whitespace and detaches a sentence-final full stop, so
// "yesterday." becomes "yesterday" and ".": a stand-in for any custom word boundary rule.
func splitTrailingDot(text string) []string {
	return strings.Fields(strings.ReplaceAll(text, ".", " ."))
}

// Trains from raw text split by a word splitter, given to the trainer as a pipeline option, and
// runs inference with the same option. Training and inference must split identically: the dataset
// and the evaluation split each example's text with the trainer's pipeline, and Run splits its
// inputs with the inference pipeline's splitter, so Run labels the same words as RunWords.
func tokenClassificationWordSplitterTraining(t *testing.T, session *hugot.Session) {
	examples := readTokenClassificationExamples(t, testutil.TestCasesFolder+"tokenClassificationCitiesAsOrg.jsonl")
	textExamples := make([]datasets.TokenClassificationExample, len(examples))
	for i, example := range examples {
		// "Anna flew to Paris yesterday ." -> "Anna flew to Paris yesterday."
		text := strings.ReplaceAll(strings.Join(example.Tokens, " "), " .", ".")
		assert.Equal(t, example.Tokens, splitTrailingDot(text), "the splitter must recover the original words")
		textExamples[i] = datasets.TokenClassificationExample{Text: text, Tags: example.Tags}
	}
	splitter := []hugot.TokenClassificationOption{pipelines.WithWordSplitter(splitTrailingDot)}

	trainDataset, err := datasets.NewInMemoryTokenClassificationDataset(textExamples, 4, nil)
	testutil.CheckT(t, err)
	evalDataset, err := datasets.NewInMemoryTokenClassificationDataset(textExamples, 12, nil)
	testutil.CheckT(t, err)

	const epochs = 6
	trainer, err := session.NewTrainer(hugot.TrainerConfig[*pipelines.TokenClassificationPipeline]{
		ModelPath:       nerModelPath,
		TrainDataset:    trainDataset,
		EvalDataset:     evalDataset,
		PipelineOptions: splitter,
		GOMLXOptions: &hugot.GOMLXTrainingOptions{
			Optimizer: optimizer.Adam().LearningRate(1.5e-5).Done(),
		},
	}, hugot.WithEpochs(epochs))
	testutil.CheckT(t, err)
	defer func() { testutil.CheckT(t, trainer.Close()) }()
	testutil.CheckT(t, trainer.Train(t.Context()))
	stats := trainer.Statistics()
	t.Logf("per epoch eval losses %v, entity F1 %v", stats.EpochEvalLosses, stats.EpochEvalF1)
	if assert.Len(t, stats.EpochEvalF1, epochs, "the evaluation must split the text examples") {
		assert.Less(t, stats.EpochEvalLosses[epochs-1], stats.EpochEvalLosses[0], "the loss should fall")
	}

	savePath := t.TempDir()
	testutil.CheckT(t, trainer.Save(t.Context(), savePath))
	pipeline, err := session.NewPipeline(hugot.TokenClassificationConfig{
		ModelPath: savePath,
		Name:      "word-splitter-inference",
		Options:   splitter,
	})
	testutil.CheckT(t, err)
	report, err := evaluation.EvaluateTokenClassification(t.Context(), pipeline, textExamples, 8)
	testutil.CheckT(t, err)
	t.Logf("saved model entity F1 %.2f, ORG %.2f", report.F1, report.PerType["ORG"].F1)
	assert.GreaterOrEqual(t, report.PerType["ORG"].F1, 0.9, "the cities should now be ORG entities")

	text := "Anna flew to Paris yesterday."
	fromText, err := pipeline.RunPipeline(t.Context(), []string{text})
	testutil.CheckT(t, err)
	fromWords, err := pipeline.RunWords(t.Context(), [][]string{splitTrailingDot(text)})
	testutil.CheckT(t, err)
	assert.Equal(t, fromWords.Entities, fromText.Entities, "Run must label the splitter's words exactly as RunWords does")
	found := false
	for _, entity := range fromText.Entities[0] {
		if entity.Word == "Paris" {
			found = true
			assert.Equal(t, "ORG", entity.Entity)
		}
	}
	assert.True(t, found, "Paris should be found as an entity in %v", fromText.Entities[0])
}

func TestTrainTokenClassificationWordSplitterXLA(t *testing.T) {
	runXLATraining(t, tokenClassificationWordSplitterTraining)
}

func TestTrainTokenClassificationWordSplitterXLACuda(t *testing.T) {
	runXLATrainingCuda(t, tokenClassificationWordSplitterTraining)
}

func TestTrainTokenClassificationWordSplitterGo(t *testing.T) {
	runGoTraining(t, tokenClassificationWordSplitterTraining)
}

// toSpanExample turns a word-level example into the span format annotation tools export: the text,
// its tokens with character offsets, and one span per IOB entity.
func toSpanExample(example datasets.TokenClassificationExample) datasets.TokenClassificationExample {
	span := datasets.TokenClassificationExample{Spans: []datasets.EntitySpan{}, Answer: datasets.AnswerAccept}
	pos := 0
	for i, word := range example.Tokens {
		if i > 0 {
			pos++ // the space
		}
		start, end := pos, pos+len([]rune(word))
		span.SourceTokens = append(span.SourceTokens, datasets.SourceToken{Text: word, Start: start, End: end})
		tag := example.Tags[i]
		switch {
		case strings.HasPrefix(tag, "B-"):
			span.Spans = append(span.Spans, datasets.EntitySpan{Label: tag[2:], Start: start, End: end})
		case strings.HasPrefix(tag, "I-"):
			span.Spans[len(span.Spans)-1].End = end
		}
		pos = end
	}
	span.Text = strings.Join(example.Tokens, " ")
	return span
}

// Fine-tunes from span-annotated examples (the format Prodigy exports), with no word splitter, so
// the words are the annotation tool's tokens. The training data, the per-epoch F1 and the final
// evaluation all go through the span-to-tag conversion; a rejected example is mixed in and must
// be skipped rather than trained on.
func tokenClassificationSpansTraining(t *testing.T, session *hugot.Session) {
	examples := readTokenClassificationExamples(t, testutil.TestCasesFolder+"tokenClassificationCitiesAsOrg.jsonl")
	spanExamples := make([]datasets.TokenClassificationExample, 0, len(examples)+1)
	for _, example := range examples {
		spanExample := toSpanExample(example)
		_, tags, err := spanExample.Resolve(nil)
		testutil.CheckT(t, err)
		assert.Equal(t, example.Tags, tags, "the span conversion must give back the word tags")
		spanExamples = append(spanExamples, spanExample)
	}
	rejected := toSpanExample(examples[0])
	rejected.Spans, rejected.Answer = []datasets.EntitySpan{{Label: "NOT_A_LABEL", Start: 0, End: 4}}, "reject"
	spanExamples = append(spanExamples, rejected)

	trainDataset, err := datasets.NewInMemoryTokenClassificationDataset(spanExamples, 4, nil)
	testutil.CheckT(t, err)
	evalDataset, err := datasets.NewInMemoryTokenClassificationDataset(spanExamples, 12, nil)
	testutil.CheckT(t, err)

	const epochs = 6
	trainer, err := session.NewTrainer(hugot.TrainerConfig[*pipelines.TokenClassificationPipeline]{
		ModelPath:    nerModelPath,
		TrainDataset: trainDataset,
		EvalDataset:  evalDataset,
		GOMLXOptions: &hugot.GOMLXTrainingOptions{
			Optimizer: optimizer.Adam().LearningRate(1.5e-5).Done(),
		},
	}, hugot.WithEpochs(epochs))
	testutil.CheckT(t, err)
	defer func() { testutil.CheckT(t, trainer.Close()) }()
	testutil.CheckT(t, trainer.Train(t.Context()))
	stats := trainer.Statistics()
	t.Logf("per epoch eval losses %v, entity F1 %v", stats.EpochEvalLosses, stats.EpochEvalF1)
	assert.Len(t, stats.EpochEvalF1, epochs)

	savePath := t.TempDir()
	testutil.CheckT(t, trainer.Save(t.Context(), savePath))
	report := entityReport(t, session, savePath, spanExamples)
	t.Logf("saved model entity F1 %.2f, ORG %.2f", report.F1, report.PerType["ORG"].F1)
	assert.GreaterOrEqual(t, report.PerType["ORG"].F1, 0.9, "the cities should now be ORG entities")
}

func TestTrainTokenClassificationSpansXLA(t *testing.T) {
	runXLATraining(t, tokenClassificationSpansTraining)
}

func TestTrainTokenClassificationSpansXLACuda(t *testing.T) {
	runXLATrainingCuda(t, tokenClassificationSpansTraining)
}

func TestTrainTokenClassificationSpansGo(t *testing.T) {
	runGoTraining(t, tokenClassificationSpansTraining)
}

// Fine-tunes a two-class sentiment model from Prodigy-format annotations, read from a file: the
// inverted-sentiment examples rewritten as binary accepts, binary rejects of the other class
// (which, for a two-class model, name the class the text is) and choice annotations, plus an
// ignored line that must be skipped. The saved model must learn the same labels as from the plain
// file, and the per-epoch F1 and the evaluation must read the annotations the same way.
func textClassificationProdigyTraining(t *testing.T, session *hugot.Session) {
	examples := readTextClassificationExamples(t, testutil.TestCasesFolder+"textClassificationInvertedSentiment.jsonl")
	other := map[string]string{"POSITIVE": "NEGATIVE", "NEGATIVE": "POSITIVE"}
	var lines []string
	for i, example := range examples {
		var line map[string]any
		switch i % 3 {
		case 0:
			line = map[string]any{"text": example.Text, "label": example.Label, "answer": "accept"}
		case 1:
			line = map[string]any{"text": example.Text, "label": other[example.Label], "answer": "reject"}
		default:
			line = map[string]any{"text": example.Text, "options": []map[string]string{{"id": "POSITIVE"}, {"id": "NEGATIVE"}}, "accept": []string{example.Label}, "answer": "accept"}
		}
		encoded, err := json.Marshal(line)
		testutil.CheckT(t, err)
		lines = append(lines, string(encoded))
	}
	lines = append(lines, `{"text": "An ignored line with a label the model does not have", "label": "NOT_A_LABEL", "answer": "ignore"}`)
	trainPath := t.TempDir() + "/prodigy.jsonl"
	testutil.CheckT(t, os.WriteFile(trainPath, []byte(strings.Join(lines, "\n")+"\n"), 0o600))

	trainDataset, err := datasets.NewTextClassificationDataset(t.Context(), trainPath, 4, datasets.SingleLabel, nil, nil)
	testutil.CheckT(t, err)
	evalDataset, err := datasets.NewTextClassificationDataset(t.Context(), trainPath, 8, datasets.SingleLabel, nil, nil)
	testutil.CheckT(t, err)

	const epochs = 6
	trainer, err := session.NewTrainer(hugot.TrainerConfig[*pipelines.TextClassificationPipeline]{
		ModelPath:    sst2ModelPath,
		TrainDataset: trainDataset,
		EvalDataset:  evalDataset,
		GOMLXOptions: &hugot.GOMLXTrainingOptions{
			Optimizer: optimizer.Adam().LearningRate(2e-4).Done(),
		},
	}, hugot.WithEpochs(epochs))
	testutil.CheckT(t, err)
	defer func() { testutil.CheckT(t, trainer.Close()) }()
	testutil.CheckT(t, trainer.Train(t.Context()))
	stats := trainer.Statistics()
	t.Logf("per epoch eval losses %v, macro-F1 %v", stats.EpochEvalLosses, stats.EpochEvalF1)
	assert.Len(t, stats.EpochEvalF1, epochs)

	savePath := t.TempDir()
	testutil.CheckT(t, trainer.Save(t.Context(), savePath))
	after := classificationAccuracy(t, session, savePath, "model.onnx", examples)
	t.Logf("accuracy of the saved model on the plain labels %.2f", after)
	assert.GreaterOrEqual(t, after, 0.9, "the annotations must teach the same labels as the plain file")

	prodigyExamples, err := evalDataset.Examples()
	testutil.CheckT(t, err)
	assert.Len(t, prodigyExamples, len(examples), "the ignored line must be skipped")
	pipeline, err := session.NewPipeline(hugot.TextClassificationConfig{ModelPath: savePath, OnnxFilename: "model.onnx", Name: "prodigyF1"})
	testutil.CheckT(t, err)
	report, err := evaluation.EvaluateTextClassification(t.Context(), pipeline, prodigyExamples, 8)
	testutil.CheckT(t, err)
	assert.InDelta(t, report.MacroF1, stats.EpochEvalF1[epochs-1], 1e-6, "evaluation must read the annotations as training does")
	assert.GreaterOrEqual(t, report.MacroF1, 0.9)
}

func TestTrainTextClassificationProdigyXLA(t *testing.T) {
	runXLATraining(t, textClassificationProdigyTraining)
}

func TestTrainTextClassificationProdigyXLACuda(t *testing.T) {
	runXLATrainingCuda(t, textClassificationProdigyTraining)
}

func TestTrainTextClassificationProdigyGo(t *testing.T) {
	runGoTraining(t, textClassificationProdigyTraining)
}
