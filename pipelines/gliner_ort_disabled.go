//go:build !ORT && !ALL

package pipelines

import (
	"github.com/knights-analytics/hugot/backends"
)

func createGLiNERTensorsORT(batch *GLiNERBatch, model *pipelineBackends.Model) error {
	// ORT not available - this will never be called when using GO/XLA runtime
	return nil
}

func runGLiNERSessionOnBatchORT(batch *GLiNERBatch, p *pipelineBackends.BasePipeline) error {
	// ORT not available - this will never be called when using GO/XLA runtime
	return nil
}
