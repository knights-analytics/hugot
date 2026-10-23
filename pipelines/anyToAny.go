package pipelines

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/knights-analytics/hugot/backends"
)

// AnyToAnyPipeline supports Phi-4 multimodal understanding and native Janus/LFM
// media generation. Input and output modalities depend on the model family;
// see PIPELINE_SUPPORT.md for the verified graph contracts.
type AnyToAnyPipeline struct {
	*multimodalGeneration
	VideoDecoder    VideoDecoder
	VideoFrameLimit int
	GenerationMode  string
	MaxNewTokens    int
	native          *nativeMultimodal
}
type AnyToAnyConfig = backends.PipelineConfig[*AnyToAnyPipeline]
type AnyToAnyOption = backends.PipelineOption[*AnyToAnyPipeline]

func (*AnyToAnyPipeline) IsGenerative() bool { return true }

func (p *AnyToAnyPipeline) GetStatistics() backends.PipelineStatistics {
	if p.native != nil {
		return nativeStatistics(p.BasePipeline)
	}
	return p.multimodalGeneration.GetStatistics()
}
func WithAnyToAnyMaxLength(n int) AnyToAnyOption {
	return func(p *AnyToAnyPipeline) error { p.MaxLength = n; return nil }
}
func WithAnyToAnyStreaming() AnyToAnyOption {
	return func(p *AnyToAnyPipeline) error { p.Streaming = true; return nil }
}
func WithAnyToAnySystemPrompt(prompt string) AnyToAnyOption {
	return func(p *AnyToAnyPipeline) error { p.SystemPrompt = prompt; return nil }
}

// WithAnyToAnyTools sets Hermes-style tool definitions for Phi-4's GenAI chat template.
// The pipeline returns generated tool-call text; applications execute tools themselves.
func WithAnyToAnyTools(tools []string) AnyToAnyOption {
	return func(p *AnyToAnyPipeline) error {
		p.Tools = append([]string(nil), tools...)
		return nil
	}
}

// WithAnyToAnyGuidance enables constrained text generation for Phi-4's GenAI session.
func WithAnyToAnyGuidance(guidance *backends.Guidance) AnyToAnyOption {
	return func(p *AnyToAnyPipeline) error {
		if guidance == nil {
			p.Guidance = nil
			return nil
		}
		guidanceCopy := *guidance
		p.Guidance = &guidanceCopy
		return nil
	}
}

// WithAnyToAnyGenerationMode requests text, image, audio or interleaved output.
// Available modes depend on the selected model family.
func WithAnyToAnyGenerationMode(mode string) AnyToAnyOption {
	return func(p *AnyToAnyPipeline) error { p.GenerationMode = mode; return nil }
}

func WithAnyToAnyMaxNewTokens(n int) AnyToAnyOption {
	return func(p *AnyToAnyPipeline) error { p.MaxNewTokens = n; return nil }
}

func NewAnyToAnyPipeline(ctx context.Context, config AnyToAnyConfig, model *backends.Model) (*AnyToAnyPipeline, error) {
	if model == nil {
		return nil, errors.New("any-to-any requires a supported multimodal model")
	}
	p := &AnyToAnyPipeline{multimodalGeneration: &multimodalGeneration{BasePipeline: backends.NewBasePipeline(ctx, config, model), MaxLength: defaultMultimodalMaxLength}, VideoFrameLimit: 16, GenerationMode: "text", MaxNewTokens: 256}
	for _, option := range config.Options {
		if err := option(p); err != nil {
			return nil, err
		}
	}
	if !model.IsGenerative {
		var err error
		p.native, err = newNativeMultimodal(ctx, p)
		if err != nil {
			return nil, err
		}
		return p, p.Validate()
	}
	if err := p.Validate(); err != nil {
		return nil, err
	}
	// Do not infer capabilities from a generic generative flag or silently
	// accept a vision-only bundle as full any-to-any support.
	var cfg struct {
		Model struct {
			Type   string `json:"type"`
			Vision struct {
				Filename       string `json:"filename"`
				ConfigFilename string `json:"config_filename"`
			} `json:"vision"`
			Speech struct {
				Filename       string `json:"filename"`
				ConfigFilename string `json:"config_filename"`
			} `json:"speech"`
		} `json:"model"`
	}
	if err := readNativeConfig(ctx, model.Path, "genai_config.json", &cfg); err != nil {
		return nil, err
	}
	if cfg.Model.Type != "phi4mm" || cfg.Model.Vision.Filename == "" || cfg.Model.Vision.ConfigFilename == "" || cfg.Model.Speech.Filename == "" || cfg.Model.Speech.ConfigFilename == "" {
		return nil, errors.New("any-to-any Stage A requires a phi4mm GenAI bundle with vision and speech processors")
	}
	return p, nil
}

func (p *AnyToAnyPipeline) Validate() error {
	if p == nil {
		return errors.New("any-to-any requires a model")
	}
	if p.native != nil {
		if p.MaxNewTokens <= 0 || p.MaxLength <= 0 || p.VideoFrameLimit <= 0 {
			return errors.New("generation lengths and video frame limit must be positive")
		}
		if len(p.Tools) > 0 || p.Guidance != nil {
			return errors.New("tool calls and guidance require the Phi-4 GenAI backend; native Janus and LFM pipelines do not support them")
		}
		return p.native.validateMode(p.GenerationMode)
	}
	if err := p.multimodalGeneration.Validate(); err != nil {
		return err
	}
	if p.VideoFrameLimit <= 0 {
		return errors.New("video frame limit must be positive")
	}
	if p.GenerationMode != "text" {
		return errors.New("phi4mm produces text only; select a native media-generating bundle for other output modes")
	}
	return p.Model.ValidateMultimodalSession()
}

func (p *AnyToAnyPipeline) Run(ctx context.Context, inputs []string) (backends.PipelineBatchOutput, error) {
	conversations := make([][]backends.Message, len(inputs))
	for i, text := range inputs {
		conversations[i] = []backends.Message{{Role: "user", Content: text}}
	}
	return p.RunMessages(ctx, conversations)
}

// RunMessages accepts conversations with attachments or ordered ContentBlocks.
// Phi-4 produces text; native bundles select media with GenerationMode. Audio
// inputs must use the format accepted by the selected model's processor.
func (p *AnyToAnyPipeline) RunMessages(ctx context.Context, inputs [][]backends.Message) (*AnyToAnyOutput, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	if err := validatePipelineContext(ctx, p.BasePipeline); err != nil {
		return nil, err
	}
	if p.native != nil {
		return p.runNativeMultimodal(ctx, inputs)
	}
	prepared, err := anyToAnyMessages(inputs)
	if err != nil {
		return nil, err
	}
	out, err := p.run(ctx, prepared, p.Tools, p.Guidance)
	if out == nil {
		return nil, err
	}
	return &AnyToAnyOutput{Responses: out.Responses, TokenStream: out.TokenStream, ErrorStream: out.ErrorStream}, err
}

func anyToAnyMessages(inputs [][]backends.Message) ([][]backends.Message, error) {
	if len(inputs) == 0 {
		return nil, errors.New("any-to-any requires at least one conversation")
	}
	result := make([][]backends.Message, len(inputs))
	for i, conversation := range inputs {
		if len(conversation) == 0 {
			return nil, fmt.Errorf("conversation %d is empty", i)
		}
		result[i] = make([]backends.Message, len(conversation))
		audioIndex := 1
		for j, message := range conversation {
			if message.Role == "" {
				message.Role = "user"
			}
			if strings.TrimSpace(message.Content) == "" && len(message.ImageURLs) == 0 && len(message.AudioURLs) == 0 && len(message.ContentBlocks) == 0 {
				return nil, fmt.Errorf("conversation %d message %d is empty", i, j)
			}
			// Both marker families are generated once, with conversation-wide
			// indices. The existing backend adds Phi image markers after content.
			if strings.Contains(message.Content, "<|image_") || strings.Contains(message.Content, "<|audio_") {
				return nil, errors.New("pass media as attachments; any-to-any generates Phi media markers")
			}
			for _, paths := range [][]string{message.ImageURLs, message.AudioURLs} {
				for _, path := range paths {
					if strings.TrimSpace(path) == "" {
						return nil, errors.New("media attachment path is empty")
					}
				}
			}
			var prefix strings.Builder
			for range message.AudioURLs {
				fmt.Fprintf(&prefix, "<|audio_%d|>\n", audioIndex)
				audioIndex++
			}
			message.Content = prefix.String() + message.Content
			message.ImageURLs = append([]string(nil), message.ImageURLs...)
			message.AudioURLs = append([]string(nil), message.AudioURLs...)
			message.ContentBlocks = append([]backends.ContentBlock(nil), message.ContentBlocks...)
			for _, block := range message.ContentBlocks {
				if block.Type == "audio" {
					audioIndex++
				}
			}
			result[i][j] = message
		}
	}
	return result, nil
}
