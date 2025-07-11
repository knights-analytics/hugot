//go:build XLA || ALL

package hugot

import (
	"fmt"
	"os"
	"testing"

	"github.com/gomlx/gomlx/types/tensors"
	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/pipelineBackends"
	"github.com/knights-analytics/hugot/pipelines"
)

// FEATURE EXTRACTION

func TestFeatureExtractionPipelineXLA(t *testing.T) {
	session, err := NewXLASession()
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	featureExtractionPipeline(t, session)
}

func TestFeatureExtractionPipelineXLACuda(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	session, err := NewXLASession(options.WithCuda(map[string]string{
		"device_id": "0",
	}))
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	featureExtractionPipeline(t, session)
}

func TestFeatureExtractionPipelineValidationXLA(t *testing.T) {
	session, err := NewXLASession()
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	featureExtractionPipelineValidation(t, session)
}

// Text classification

func TestTextClassificationPipelineXLA(t *testing.T) {
	session, err := NewXLASession()
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	textClassificationPipeline(t, session)
}

func TestTextClassificationPipelineXLACuda(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	session, err := NewXLASession(options.WithCuda(map[string]string{
		"device_id": "0",
	}))
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	textClassificationPipeline(t, session)
}

func TestTextClassificationPipelineMultiXLA(t *testing.T) {
	session, err := NewXLASession()
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	textClassificationPipelineMulti(t, session)
}

func TestTextClassificationPipelineMultiXLACuda(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	session, err := NewXLASession(options.WithCuda(map[string]string{
		"device_id": "0",
	}))
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	textClassificationPipelineMulti(t, session)
}

func TestTextClassificationPipelineValidationXLA(t *testing.T) {
	session, err := NewXLASession()
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	textClassificationPipelineValidation(t, session)
}

// Token classification

func TestTokenClassificationPipelineXLA(t *testing.T) {
	session, err := NewXLASession()
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	tokenClassificationPipeline(t, session)
}

func TestTokenClassificationPipelineXLACuda(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	session, err := NewXLASession(options.WithCuda(map[string]string{
		"device_id": "0",
	}))
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	tokenClassificationPipeline(t, session)
}

func TestTokenClassificationPipelineValidationXLA(t *testing.T) {
	session, err := NewXLASession()
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	tokenClassificationPipelineValidation(t, session)
}

// Zero shot

func TestZeroShotClassificationPipelineXLA(t *testing.T) {
	session, err := NewXLASession()
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	zeroShotClassificationPipeline(t, session)
}

func TestZeroShotClassificationPipelineXLACuda(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	session, err := NewXLASession(options.WithCuda(map[string]string{
		"device_id": "0",
	}))
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	zeroShotClassificationPipeline(t, session)
}

func TestZeroShotClassificationPipelineValidationXLA(t *testing.T) {
	session, err := NewXLASession()
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	zeroShotClassificationPipelineValidation(t, session)
}

// No same name

func TestNoSameNamePipelineXLA(t *testing.T) {
	session, err := NewXLASession()
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	noSameNamePipeline(t, session)
}

func TestDestroyPipelineXLA(t *testing.T) {
	session, err := NewXLASession()
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	destroyPipelines(t, session)
}

// Thread safety

func TestThreadSafetyXLA(t *testing.T) {
	session, err := NewXLASession()
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	threadSafety(t, session, 500)
}

func TestThreadSafetyXLACuda(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.SkipNow()
	}
	session, err := NewXLASession(options.WithCuda(map[string]string{
		"device_id": "0",
	}))
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	threadSafety(t, session, 1000)
}

// TEMP: testing how we can do inference with XLA and gemma

type ModelConfig struct {
	NumKeyValueHeads int   `json:"num_key_value_heads"`
	HeadDim          int   `json:"head_dim"`
	NumHiddenLayers  int   `json:"num_hidden_layers"`
	EosTokenID       []int `json:"eos_token_id"`
}

func CreateCache(batchSize, numLayers, numKeyValueHeads, seqLen, headDim int) []*tensors.Tensor {
	cache := make([]*tensors.Tensor, numLayers*2)

	for layer := 0; layer < numLayers; layer++ {
		keyTensor := tensors.FromScalarAndDimensions(float32(0), batchSize, numKeyValueHeads, seqLen, headDim)
		cache[layer*2] = keyTensor

		valueTensor := tensors.FromScalarAndDimensions(float32(0), batchSize, numKeyValueHeads, seqLen, headDim)
		cache[layer*2+1] = valueTensor
	}
	return cache
}

func check(err error) {
	if err != nil {
		panic(err.Error())
	}
}

func TestHugoPipeline(t *testing.T) {
	session, err := NewXLASession()
	check(err)

	defer func(session *Session) {
		err := session.Destroy()
		check(err)
	}(session)

	config := TextGenerationConfig{
		ModelPath:    "/home/testuser/repositories/onnx_models",
		Name:         "test pipeline",
		OnnxFilename: "gemma_model.onnx",
		Options: []pipelineBackends.PipelineOption[*pipelines.TextGenerationPipeline]{
			pipelines.WithMaxTokens(15),
		},
	}

	gemmaPipeline, err := NewPipeline(session, config)

	check(err)

	batch := []string{"who was the fourty third president of the United States?", "who is the president of the Netherlands?"}
	// batch := []string{"what is the capital of the Netherlands?"}

	batchResult, err := gemmaPipeline.Run(batch)
	check(err)

	fmt.Println(batchResult.GetOutput())
}

func TestTextGenerationPipelineTokensXLA(t *testing.T) {
	session, err := NewXLASession()
	checkT(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkT(t, destroyErr)
	}(session)
	textGenerationPipelineTokens(t, session)
}
