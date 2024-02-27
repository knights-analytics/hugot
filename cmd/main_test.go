package main

import (
	"os"
	"testing"

	"github.com/urfave/cli/v2"
)

func TestRun(t *testing.T) {
	app := &cli.App{
		Name:     "hugot",
		Usage:    "Huggingface transformers from the command line - alpha",
		Commands: []*cli.Command{runCommand},
	}
	args := os.Args[0:1]
	args = append(args, "run", "--input=./testData", "--model=../models/distilbert-base-uncased-finetuned-sst-2-english", "--type=textClassification")
	if err := app.Run(args); err != nil {
		panic(err)
	}
}
