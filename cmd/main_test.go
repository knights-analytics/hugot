package main

import (
	"fmt"
	"os"
	"path"
	"testing"

	_ "embed"

	"github.com/urfave/cli/v2"
)

//go:embed testData/textClassification.jsonl
var textClassificationData []byte

//go:embed testData/tokenClassification.jsonl
var tokenClassificationData []byte

func TestTextClassificationCli(t *testing.T) {
	app := &cli.App{
		Name:     "hugot",
		Usage:    "Huggingface transformers from the command line - alpha",
		Commands: []*cli.Command{runCommand},
	}
	baseArgs := os.Args[0:1]

	modelFolder := os.Getenv("TEST_MODELS_FOLDER")
	if modelFolder == "" {
		modelFolder = "../models/"
	}
	testModel := path.Join(modelFolder, "distilbert-base-uncased-finetuned-sst-2-english")

	// write the test data and test recursive reads and processing from folder
	testDataDir := path.Join(os.TempDir(), "hugoTestData")
	recurseDir := path.Join(testDataDir, "cliRecurseTest")
	err := os.MkdirAll(recurseDir, os.ModePerm)
	check(t, err)
	err = os.WriteFile(path.Join(testDataDir, "test-0.jsonl"), textClassificationData, os.ModePerm)
	check(t, err)
	err = os.WriteFile(path.Join(recurseDir, "test-1.jsonl"), textClassificationData, os.ModePerm)
	check(t, err)
	defer func() {
		err := os.RemoveAll(testDataDir)
		check(t, err)
	}()

	args := append(baseArgs, "run", fmt.Sprintf("--input=%s", testDataDir), fmt.Sprintf("--model=%s", testModel), "--type=textClassification")
	if err := app.Run(args); err != nil {
		check(t, err)
	}
}

func TestTokenClassificationCli(t *testing.T) {
	app := &cli.App{
		Name:     "hugot",
		Usage:    "Huggingface transformers from the command line - alpha",
		Commands: []*cli.Command{runCommand},
	}
	baseArgs := os.Args[0:1]

	modelFolder := os.Getenv("TEST_MODELS_FOLDER")
	if modelFolder == "" {
		modelFolder = "../models/"
	}
	testModel := path.Join(modelFolder, "distilbert-NER")

	testDataDir := path.Join(os.TempDir(), "hugoTestData")
	err := os.MkdirAll(testDataDir, os.ModePerm)
	check(t, err)
	err = os.WriteFile(path.Join(testDataDir, "test-token-classification.jsonl"), tokenClassificationData, os.ModePerm)
	check(t, err)
	defer func() {
		err := os.RemoveAll(testDataDir)
		check(t, err)
	}()

	args := append(baseArgs, "run", fmt.Sprintf("--input=%s", path.Join(testDataDir, "test-token-classification.jsonl")),
		fmt.Sprintf("--model=%s", testModel), "--type=tokenClassification", fmt.Sprintf("--output=%s", testDataDir))
	if err := app.Run(args); err != nil {
		check(t, err)
	}
	result, err := os.ReadFile(path.Join(testDataDir, "result-0.jsonl"))
	check(t, err)
	fmt.Println(string(result))
}

func TestFeatureExtractionCli(t *testing.T) {
	app := &cli.App{
		Name:     "hugot",
		Usage:    "Huggingface transformers from the command line - alpha",
		Commands: []*cli.Command{runCommand},
	}
	baseArgs := os.Args[0:1]

	modelFolder := os.Getenv("TEST_MODELS_FOLDER")
	if modelFolder == "" {
		modelFolder = "../models/"
	}
	testModel := path.Join(modelFolder, "all-MiniLM-L6-v2")

	testDataDir := path.Join(os.TempDir(), "hugoTestData")
	err := os.MkdirAll(testDataDir, os.ModePerm)
	check(t, err)
	err = os.WriteFile(path.Join(testDataDir, "test-feature-extraction.jsonl"), tokenClassificationData, os.ModePerm)
	check(t, err)
	defer func() {
		err := os.RemoveAll(testDataDir)
		check(t, err)
	}()

	args := append(baseArgs, "run", fmt.Sprintf("--input=%s", path.Join(testDataDir, "test-feature-extraction.jsonl")),
		fmt.Sprintf("--model=%s", testModel), "--type=featureExtraction", fmt.Sprintf("--output=%s", testDataDir))
	if err := app.Run(args); err != nil {
		check(t, err)
	}
	result, err := os.ReadFile(path.Join(testDataDir, "result-0.jsonl"))
	check(t, err)
	fmt.Println(string(result))
}

func check(t *testing.T, err error) {
	if err != nil {
		t.Fatalf("%s", err.Error())
	}
}
