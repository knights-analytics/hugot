package pipelines

import (
	"bytes"
	"errors"
	"fmt"
	"math"
	"sync/atomic"
	"text/template"
	"time"

	"github.com/knights-analytics/hugot/chatTemplates"
	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/pipelineBackends"
)

// TextGenerationPipeline enables generative text inference using ONNX models.
// Supported backends: Go and ORT
//
// Decoding method:
// - Greedy search (default)
//
// Supports chat templating (like Hugging Face’s apply_chat_template) via RunWithTemplate,
// allowing structured roles (system, user, assistant) important for instruction-tuned models.
//
// Templates come from tokenizer_config.json files in Jinja2 format, converted for Go’s text/template.
// Currently includes Gemma and Phi chat templates (./chatTemplates/chatTemplates.go).
//
// Without templates, plain text input is supported via Run, but templated input improves output quality.
//
// Test cases use HuggingFaceTB/SmolLM-135M and compare the outputs with the outputs of Python's onnxruntime library.
// onnx-community/gemma-3-1b-it and microsoft/Phi-3-mini-4k-instruct-onnx were also tested but not included in the testing file

// Example usage:

// session, err := NewORTSession()
// check(err)

// defer func(session *Session) {
// 	err := session.Destroy()
// 	check(err)
// }(session)

// config := TextGenerationConfig{
// 	ModelPath:    "./models/KnightsAnalytics_gemma-3-1b-it",
// 	Name:         "testPipeline",
// 	OnnxFilename: "model_quantized.onnx",
// 	Options: []pipelineBackends.PipelineOption[*pipelines.TextGenerationPipeline]{
// 		pipelines.WithMaxTokens(256),
// 		pipelines.WithGemmaTemplate(),
// 	},
// }

// gemmaPipeline, err := NewPipeline(session, config)
// check(err)

// messages := [][]pipelines.Message{
// 	{
// 		{Role: "system", Content: "you are a helpful assistant."},
// 		{Role: "user", Content: "tell me some facts about the capital of the Netherlands"},
// 	},
// }

// batchResult, err := gemmaPipeline.RunWithTemplate(messages)
// check(err)
// fmt.Println(batchResult.GetOutput())

type TextGenerationPipeline struct {
	*pipelineBackends.BasePipeline
	MaxNewTokens int
	OutputName   string
	Output       pipelineBackends.InputOutputInfo
	Template     *template.Template
	EosToken     string
}

type TextGenerationOutput struct {
	TextGenerationOutputs []string
	GeneratedTokens       [][]uint32
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type TemplateData struct {
	Messages            []Message `json:"messages"`
	AddGenerationPrompt bool      `json:"add_generation_prompt"`
	EosToken            string    `json:"eos_token"`
}

func (t *TextGenerationOutput) GetOutput() []any {
	out := make([]any, len(t.TextGenerationOutputs))
	for i, textOutput := range t.TextGenerationOutputs {
		out[i] = any(textOutput)
	}
	return out
}

// WithMaxTokens allows the user to define the maximum generated tokens
func WithMaxTokens(maxToken int) pipelineBackends.PipelineOption[*TextGenerationPipeline] {
	return func(pipeline *TextGenerationPipeline) error {
		pipeline.MaxNewTokens = maxToken
		return nil
	}
}

// WithGemmaTemplate allows the user to apply the chat template provided in Gemma's tokenizer_config file
func WithGemmaTemplate() pipelineBackends.PipelineOption[*TextGenerationPipeline] {
	return func(pipeline *TextGenerationPipeline) error {
		tmpl, err := template.New("gemma").Funcs(chatTemplates.FuncMap).Parse(chatTemplates.GemmaTemplate)
		if err != nil {
			return errors.New("parsing of gemma template failed")
		}
		pipeline.Template = tmpl
		return nil
	}
}

// WithPhiTemplate allows the user to apply the chat template provided in Phi's tokenizer_config file
func WithPhiTemplate() pipelineBackends.PipelineOption[*TextGenerationPipeline] {
	return func(pipeline *TextGenerationPipeline) error {
		tmpl, err := template.New("phi").Funcs(chatTemplates.FuncMap).Parse(chatTemplates.PhiTemplate)
		if err != nil {
			return errors.New("parsing of gemma template failed")
		}
		pipeline.Template = tmpl
		return nil
	}
}

func WithCustomStopTokens(stopTokens []int64) pipelineBackends.PipelineOption[*TextGenerationPipeline] {
	return func(pipeline *TextGenerationPipeline) error {
		for _, token := range stopTokens {
			pipeline.Model.EosTokenIDs[token] = true
		}

		return nil
	}
}

// NewTextGenerationPipeline initializes a new text generation pipeline
func NewTextGenerationPipeline(config pipelineBackends.PipelineConfig[*TextGenerationPipeline], s *options.Options, model *pipelineBackends.Model) (*TextGenerationPipeline, error) {
	defaultPipeline, err := pipelineBackends.NewBasePipeline(config, s, model)
	if err != nil {
		return nil, err
	}

	pipeline := &TextGenerationPipeline{BasePipeline: defaultPipeline}
	for _, o := range config.Options {
		err = o(pipeline)
		if err != nil {
			return nil, err
		}
	}

	if pipeline.MaxNewTokens <= 0 {
		pipeline.MaxNewTokens = 1028 // Default value if not set as per Python
	}

	err = pipeline.Validate()
	if err != nil {
		return nil, err
	}
	return pipeline, nil
}

// INTERFACE IMPLEMENTATION

func (p *TextGenerationPipeline) GetMetadata() pipelineBackends.PipelineMetadata {
	return pipelineBackends.PipelineMetadata{}
}

func (p *TextGenerationPipeline) GetModel() *pipelineBackends.Model {
	return p.BasePipeline.Model
}

func (p *TextGenerationPipeline) GetStats() []string {
	return []string{
		fmt.Sprintf("Statistics for pipeline: %s", p.PipelineName),
		fmt.Sprintf("Tokenizer: Total time=%s, Execution count=%d, Average query time=%s",
			time.Duration(p.Model.Tokenizer.TokenizerTimings.TotalNS),
			p.Model.Tokenizer.TokenizerTimings.NumCalls,
			time.Duration(float64(p.Model.Tokenizer.TokenizerTimings.TotalNS)/math.Max(1, float64(p.Model.Tokenizer.TokenizerTimings.NumCalls)))),
		fmt.Sprintf("ONNX: Total time=%s, Execution count=%d, Average query time=%s",
			time.Duration(p.PipelineTimings.TotalNS),
			p.PipelineTimings.NumCalls,
			time.Duration(float64(p.PipelineTimings.TotalNS)/math.Max(1, float64(p.PipelineTimings.NumCalls)))),
	}
}

func (p *TextGenerationPipeline) Validate() error {
	var validationErrors []error
	if p.Model.Tokenizer == nil {
		validationErrors = append(validationErrors, fmt.Errorf("text generation pipeline requires a tokenizer"))
	}
	if len(p.Model.EosTokenIDs) == 0 {
		validationErrors = append(validationErrors, errors.New("no EOS Token IDs found"))
	}
	if p.Model.NumHiddenLayers == 0 {
		validationErrors = append(validationErrors, errors.New("num hidden layers cannot be 0"))
	}
	if p.Model.NumKeyValueHeads == 0 {
		validationErrors = append(validationErrors, errors.New("num key value heads cannot be 0"))
	}
	if p.Model.HeadDim == 0 {
		validationErrors = append(validationErrors, errors.New("head dim cannot be 0"))
	}
	return errors.Join(validationErrors...)
}

// Preprocess tokenizes the input strings.
func (p *TextGenerationPipeline) Preprocess(batch *pipelineBackends.PipelineBatch, inputs []string) error {
	start := time.Now()
	pipelineBackends.TokenizeInputs(batch, p.Model.Tokenizer, inputs)
	atomic.AddUint64(&p.Model.Tokenizer.TokenizerTimings.NumCalls, 1)
	atomic.AddUint64(&p.Model.Tokenizer.TokenizerTimings.TotalNS, uint64(time.Since(start)))
	return pipelineBackends.CreateInputTensors(batch, p.Model, p.Runtime)
}

// Forward performs the generation loop
func (p *TextGenerationPipeline) Forward(batch *pipelineBackends.PipelineBatch) error {
	start := time.Now()

	// generation loop
	err := pipelineBackends.RunGenerativeSessionOnBatch(batch, p.BasePipeline)
	if err != nil {
		return err
	}
	atomic.AddUint64(&p.PipelineTimings.NumCalls, 1)
	atomic.AddUint64(&p.PipelineTimings.TotalNS, uint64(time.Since(start)))
	return nil
}

// Postprocess converts the generated tokens back into a readable string.
func (p *TextGenerationPipeline) Postprocess(batch *pipelineBackends.PipelineBatch) (*TextGenerationOutput, error) {
	outputValues := batch.OutputValues
	output := TextGenerationOutput{
		TextGenerationOutputs: make([]string, batch.Size),
		GeneratedTokens:       make([][]uint32, batch.Size),
	}
	for i, val := range outputValues {
		tokenIDs := val.([]int64)
		convertedTokens := make([]uint32, len(tokenIDs))
		for j, tok := range tokenIDs {
			convertedTokens[j] = uint32(tok)
		}
		output.GeneratedTokens[i] = convertedTokens

		decodedString, err := pipelineBackends.Decode(convertedTokens, p.Model.Tokenizer, true)

		if err != nil {
			return nil, errors.New("error in decoding generated tokens")
		}
		output.TextGenerationOutputs[i] = decodedString
	}

	return &output, nil
}

func (p *TextGenerationPipeline) Run(inputs []string) (pipelineBackends.PipelineBatchOutput, error) {
	return p.RunPipeline(inputs)
}

// applyChatTemplate applies the provided chat template to the input sequence
func applyChatTemplate(tmpl *template.Template, data TemplateData) (string, error) {
	var buf bytes.Buffer
	err := tmpl.Execute(&buf, data)
	if err != nil {
		return "", err
	}
	return buf.String(), nil
}

// RunWithTemplate allows the user to apply a chat template to their input sequence
func (p *TextGenerationPipeline) RunWithTemplate(inputs [][]Message) (pipelineBackends.PipelineBatchOutput, error) {
	// apply template to messages, returning []string
	// if template is not compliled, return error
	templatedMessages := make([]string, len(inputs))

	for i, message := range inputs {
		data := TemplateData{
			Messages:            message,
			AddGenerationPrompt: true,
			EosToken:            p.EosToken,
		}
		outputStr, err := applyChatTemplate(p.Template, data)
		if err != nil {
			return nil, err
		}
		templatedMessages[i] = outputStr
	}
	return p.RunPipeline(templatedMessages)
}

func (p *TextGenerationPipeline) RunPipeline(inputs []string) (*TextGenerationOutput, error) {
	var runErrors []error
	batch := pipelineBackends.NewBatch(len(inputs))
	batch.MaxNewTokens = p.MaxNewTokens
	defer func(*pipelineBackends.PipelineBatch) {
		runErrors = append(runErrors, batch.Destroy())
	}(batch)

	runErrors = append(runErrors, p.Preprocess(batch, inputs))
	if e := errors.Join(runErrors...); e != nil {
		return nil, e
	}

	runErrors = append(runErrors, p.Forward(batch))
	if e := errors.Join(runErrors...); e != nil {
		return nil, e
	}

	result, postErr := p.Postprocess(batch)
	runErrors = append(runErrors, postErr)
	return result, errors.Join(runErrors...)
}
