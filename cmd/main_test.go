package main

import (
	"context"
	"fmt"
	"os"
	"path"
	"testing"

	_ "embed"

	"github.com/knights-analytics/hugot"
	util "github.com/knights-analytics/hugot/utils"
	"github.com/urfave/cli/v2"
)

//go:embed testData/textClassification.jsonl
var textClassificationData []byte

//go:embed testData/tokenClassification.jsonl
var tokenClassificationData []byte

func TestMain(m *testing.M) {
	// model setup
	if ok, err := util.FileSystem.Exists(context.Background(), "../models"); err == nil {
		if !ok {
			session, err := hugot.NewSession()
			if err != nil {
				panic(err)
			}
			err = os.MkdirAll("../models", os.ModePerm)
			if err != nil {
				panic(err)
			}
			downloadOptions := hugot.NewDownloadOptions()
			for _, modelName := range []string{
				"KnightsAnalytics/all-MiniLM-L6-v2",
				"KnightsAnalytics/distilbert-base-uncased-finetuned-sst-2-english",
				"KnightsAnalytics/distilbert-NER"} {
				_, err := session.DownloadModel(modelName, "../models", downloadOptions)
				if err != nil {
					panic(err)
				}
			}
			err = session.Destroy()
			if err != nil {
				panic(err)
			}
		}
	} else {
		panic(err)
	}
	// run all tests
	code := m.Run()
	os.Exit(code)
}

func TestTextClassificationCli(t *testing.T) {
	app := &cli.App{
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

	testModel := path.Join("../models", "KnightsAnalytics_distilbert-NER")

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

	testModel := path.Join("../models", "KnightsAnalytics_all-MiniLM-L6-v2")

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

func TestModelChain(t *testing.T) {
	app := &cli.App{
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
	err = os.WriteFile(path.Join(testDataDir, "test-0.jsonl"), textClassificationData, os.ModePerm)
	check(t, err)
	defer func() {
		err := os.RemoveAll(testDataDir)
		check(t, err)
	}()

	// wipe the hugo folder
	userFolder, err := os.UserHomeDir()
	check(t, err)
	check(t, util.FileSystem.Delete(context.Background(), util.PathJoinSafe(userFolder, "hugot")))

	// try to download the model to hugo folder and run it
	args := append(baseArgs, "run", fmt.Sprintf("--input=%s", testDataDir), fmt.Sprintf("--model=%s", "KnightsAnalytics/distilbert-base-uncased-finetuned-sst-2-english"), "--type=textClassification")
	if err := app.Run(args); err != nil {
		check(t, err)
	}

	// run it again. This time the model should be read from the hugot folder without re-downloading it.
	args = append(baseArgs, "run", fmt.Sprintf("--input=%s", testDataDir), fmt.Sprintf("--model=%s", "KnightsAnalytics/distilbert-base-uncased-finetuned-sst-2-english"), "--type=textClassification")
	if err := app.Run(args); err != nil {
		check(t, err)
	}
}

func check(t *testing.T, err error) {
	if err != nil {
		t.Fatalf("%s", err.Error())
	}
}
