package datasets

import (
	"github.com/gomlx/gomlx/ml/train"
	"github.com/knights-analytics/hugot/backends"
)

type Dataset interface {
	train.Dataset
	Validate() error
	SetTokenizationPipeline(pipeline backends.Pipeline) error
	SetVerbose(bool)
	Close() error
}
