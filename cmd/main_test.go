//go:build cgo && (ORT || ALL)

package main

import (
	"context"
	"fmt"
	"os"
	"path"
	"testing"

	"github.com/urfave/cli/v3"

	"github.com/knights-analytics/hugot/testcases/embedded"
	"github.com/knights-analytics/hugot/util/fileutil"
)

func TestTextClassificationCli(t *testing.T) {
	app := &cli.Command{
		Name:     "hugot",
		Usage:    "Huggingface transformers from the command line - alpha",
		Commands: []*cli.Command{runCommand},
	}
	baseArgs := os.Args[0:1]

	testModel := path.Join("../models", "KnightsAnalytics_distilbert-base-uncased-finetuned-sst-2-english")

	// write the test data and test recursive reads and processing from folder
	testDataDir := path.Join(os.TempDir(), "hugoTestData")
	recurseDir := path.Join(testDataDir, "cliRecurseTest")
	err := os.MkdirAll(recurseDir, os.ModePerm)
	check(t, err)
	err = os.WriteFile(path.Join(testDataDir, "test-0.jsonl"), embedded.TextClassificationData, 0o600)
	check(t, err)
	err = os.WriteFile(path.Join(recurseDir, "test-1.jsonl"), embedded.TextClassificationData, 0o600)
	check(t, err)
	defer func() {
		err := os.RemoveAll(testDataDir)
		check(t, err)
	}()

	args := append(baseArgs, "run",
		fmt.Sprintf("--input=%s", testDataDir),
		fmt.Sprintf("--model=%s", testModel),
		"--type=textClassification")
	if err := app.Run(context.Background(), args); err != nil {
		check(t, err)
	}
}

func TestTokenClassificationCli(t *testing.T) {
	app := &cli.Command{
		Name:     "hugot",
		Usage:    "Huggingface transformers from the command line - alpha",
		Commands: []*cli.Command{runCommand},
	}
	baseArgs := os.Args[0:1]

	testModel := path.Join("../models", "KnightsAnalytics_distilbert-NER")

	testDataDir := path.Join(os.TempDir(), "hugoTestData")
	err := os.MkdirAll(testDataDir, os.ModePerm)
	check(t, err)
	err = os.WriteFile(path.Join(testDataDir, "test-token-classification.jsonl"), embedded.TokenClassificationData, 0o600)
	check(t, err)
	defer func() {
		err := os.RemoveAll(testDataDir)
		check(t, err)
	}()

	args := append(baseArgs, "run",
		fmt.Sprintf("--input=%s", path.Join(testDataDir, "test-token-classification.jsonl")),
		fmt.Sprintf("--model=%s", testModel),
		"--type=tokenClassification",
		fmt.Sprintf("--output=%s", testDataDir))
	if err := app.Run(context.Background(), args); err != nil {
		check(t, err)
	}
	result, err := os.ReadFile(path.Join(testDataDir, "result-0.jsonl"))
	check(t, err)
	fmt.Println(string(result))
}

func TestFeatureExtractionCli(t *testing.T) {
	app := &cli.Command{
		Name:     "hugot",
		Usage:    "Huggingface transformers from the command line - alpha",
		Commands: []*cli.Command{runCommand},
	}
	baseArgs := os.Args[0:1]
	testModel := path.Join("../models", "KnightsAnalytics_all-MiniLM-L6-v2")
	testDataDir := path.Join(os.TempDir(), "hugoTestData")
	err := os.MkdirAll(testDataDir, os.ModePerm)
	check(t, err)
	err = os.WriteFile(path.Join(testDataDir, "test-feature-extraction.jsonl"), embedded.TokenClassificationData, 0o600)
	check(t, err)
	defer func() {
		err := os.RemoveAll(testDataDir)
		check(t, err)
	}()

	args := append(baseArgs, "run",
		fmt.Sprintf("--input=%s", path.Join(testDataDir, "test-feature-extraction.jsonl")),
		fmt.Sprintf("--model=%s", testModel),
		fmt.Sprintf("--onnxFilename=%s", "model.onnx"),
		"--type=featureExtraction",
		fmt.Sprintf("--output=%s", testDataDir))
	if err := app.Run(context.Background(), args); err != nil {
		check(t, err)
	}
	result, err := os.ReadFile(path.Join(testDataDir, "result-0.jsonl"))
	check(t, err)
	fmt.Println(string(result))
}

func TestModelChain(t *testing.T) {
	app := &cli.Command{
		Name:     "hugot",
		Usage:    "Huggingface transformers from the command line - alpha",
		Commands: []*cli.Command{runCommand},
	}
	baseArgs := os.Args[0:1]

	// write the test data and test recursive reads and processing from folder
	testDataDir := path.Join(os.TempDir(), "hugoTestData")
	recurseDir := path.Join(testDataDir, "cliRecurseTest")
	err := os.MkdirAll(recurseDir, os.ModePerm)
	check(t, err)
	err = os.WriteFile(path.Join(testDataDir, "test-0.jsonl"), embedded.TextClassificationData, 0o600)
	check(t, err)
	defer func() {
		err := os.RemoveAll(testDataDir)
		check(t, err)
	}()

	// wipe the hugo folder
	userFolder, err := os.UserHomeDir()
	check(t, err)
	check(t, fileutil.DeleteFile(fileutil.PathJoinSafe(userFolder, "hugot")))

	// try to download the model to hugo folder and run it
	args := append(baseArgs, "run",
		fmt.Sprintf("--input=%s", testDataDir),
		fmt.Sprintf("--model=%s", "KnightsAnalytics/distilbert-base-uncased-finetuned-sst-2-english"),
		"--type=textClassification")
	if err := app.Run(context.Background(), args); err != nil {
		check(t, err)
	}

	// run it again. This time the model should be read from the hugot folder without re-downloading it.
	args = append(baseArgs, "run", fmt.Sprintf("--input=%s", testDataDir),
		fmt.Sprintf("--model=%s", "KnightsAnalytics/distilbert-base-uncased-finetuned-sst-2-english"),
		"--type=textClassification")
	if err := app.Run(context.Background(), args); err != nil {
		check(t, err)
	}
}

func check(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatalf("%s", err.Error())
	}
}
