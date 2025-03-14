//go:build !XLA && !ALL

package datasets

import (
	"bufio"
	"errors"
	"io"

	"github.com/knights-analytics/hugot/pipelineBackends"
	"github.com/knights-analytics/hugot/pipelines"
)

type Dataset interface {
	Validate() error
	SetTokenizationPipeline(pipeline pipelineBackends.Pipeline) error
	SetVerbose(bool)
	Close() error
}

type SemanticSimilarityDataset struct {
	trainingPath     string
	trainingExamples []SemanticSimilarityExample
	preprocessFunc   ExamplePreprocessFunc
	batchSize        int
	pipeline         *pipelines.FeatureExtractionPipeline
	reader           *bufio.Reader
	sourceFile       io.ReadCloser
	batchN           int
	verbose          bool
}

func (s *SemanticSimilarityDataset) Yield() (_ any, _ any, _ any, err error) {
	return nil, nil, nil, errors.New("XLA is not enabled")
}
