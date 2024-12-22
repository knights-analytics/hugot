//go:build !XLA && !ALL

package datasets

import "github.com/knights-analytics/hugot/pipelineBackends"

type Dataset interface {
	SetVerbose(bool)
}

type SemanticSimilarityDataset struct{}

func (s *SemanticSimilarityDataset) SetTokenizationPipeline(_ pipelineBackends.Pipeline) error {
	return nil
}

func (s *SemanticSimilarityDataset) SetVerbose(_ bool) {}
