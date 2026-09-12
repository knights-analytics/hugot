package pipelines

import (
	"context"
	"errors"

	"github.com/knights-analytics/hugot/backends"
)

type Seq2SeqOutput struct {
	TokenStream chan backends.SequenceDelta
	ErrorStream chan error
	Responses   []string
}

func (o *Seq2SeqOutput) GetOutput() []any {
	if o.TokenStream != nil || o.ErrorStream != nil {
		return []any{o.TokenStream, o.ErrorStream}
	}
	out := make([]any, len(o.Responses))
	for i, response := range o.Responses {
		out[i] = response
	}
	return out
}

type seq2SeqPipeline struct {
	*backends.BasePipeline
	Generation backends.Seq2SeqOptions
	Streaming  bool
	Prefix     string
}

func (p *seq2SeqPipeline) validate() error {
	if p.BasePipeline == nil || p.Model == nil || !p.Model.IsGenerative {
		return errors.New("seq2seq pipeline requires a generative encoder-decoder model")
	}
	if p.Generation.MaxLength <= 0 {
		return errors.New("seq2seq maximum length must be greater than zero")
	}
	return nil
}

func (p *seq2SeqPipeline) run(ctx context.Context, inputs []string) (*Seq2SeqOutput, error) {
	batch := backends.NewBatch(len(inputs))
	if err := backends.CreateMessages(batch, p.BasePipeline, inputs, p.Prefix); err != nil {
		return nil, errors.Join(err, batch.Destroy())
	}
	tokenStream, errorStream, err := backends.RunSeq2SeqSessionOnBatch(ctx, batch, p.BasePipeline, p.Generation)
	if err != nil {
		return nil, errors.Join(err, batch.Destroy())
	}
	if p.Streaming {
		return &Seq2SeqOutput{TokenStream: tokenStream, ErrorStream: errorStream}, nil
	}
	responses, responseErr := collectResponses(tokenStream, errorStream, len(inputs))
	return &Seq2SeqOutput{Responses: responses}, errors.Join(responseErr, batch.Destroy())
}

func (p *seq2SeqPipeline) getStatistics() backends.PipelineStatistics {
	if p.Model != nil && p.Model.ORTModel != nil && p.Model.ORTModel.Generative != nil {
		return p.Model.ORTModel.Generative.Statistics()
	}
	return backends.PipelineStatistics{}
}

func (p *seq2SeqPipeline) metadata() backends.PipelineMetadata { return backends.PipelineMetadata{} }

func WithSeq2SeqMaxLength(maxLength int) backends.PipelineOption[*SummarizationPipeline] {
	return func(p *SummarizationPipeline) error { p.Generation.MaxLength = maxLength; return nil }
}

func WithTranslationMaxLength(maxLength int) backends.PipelineOption[*TranslationPipeline] {
	return func(p *TranslationPipeline) error { p.Generation.MaxLength = maxLength; return nil }
}

func WithText2TextMaxLength(maxLength int) backends.PipelineOption[*Text2TextGenerationPipeline] {
	return func(p *Text2TextGenerationPipeline) error { p.Generation.MaxLength = maxLength; return nil }
}

func WithSummarizationStreaming() backends.PipelineOption[*SummarizationPipeline] {
	return func(p *SummarizationPipeline) error { p.Streaming = true; return nil }
}

func WithTranslationStreaming() backends.PipelineOption[*TranslationPipeline] {
	return func(p *TranslationPipeline) error { p.Streaming = true; return nil }
}

func WithText2TextStreaming() backends.PipelineOption[*Text2TextGenerationPipeline] {
	return func(p *Text2TextGenerationPipeline) error { p.Streaming = true; return nil }
}

func WithSummarizationPrefix(prefix string) backends.PipelineOption[*SummarizationPipeline] {
	return func(p *SummarizationPipeline) error { p.Prefix = prefix; return nil }
}

func WithTranslationPrefix(prefix string) backends.PipelineOption[*TranslationPipeline] {
	return func(p *TranslationPipeline) error { p.Prefix = prefix; return nil }
}

func WithText2TextPrefix(prefix string) backends.PipelineOption[*Text2TextGenerationPipeline] {
	return func(p *Text2TextGenerationPipeline) error { p.Prefix = prefix; return nil }
}

func WithTranslationLanguageCode(code string) backends.PipelineOption[*TranslationPipeline] {
	return func(p *TranslationPipeline) error { p.LanguageCode = code; return nil }
}

type (
	SummarizationPipeline struct{ seq2SeqPipeline }
	TranslationPipeline   struct {
		seq2SeqPipeline
		LanguageCode string
	}
)
type Text2TextGenerationPipeline struct{ seq2SeqPipeline }

func NewSummarizationPipeline(ctx context.Context, config backends.PipelineConfig[*SummarizationPipeline], model *backends.Model) (*SummarizationPipeline, error) {
	pipeline := &SummarizationPipeline{BasePipeline: backends.NewBasePipeline(ctx, config, model), Generation: backends.Seq2SeqOptions{MaxLength: 128}, Prefix: "summarize: "}
	for _, option := range config.Options {
		if err := option(pipeline); err != nil {
			return nil, err
		}
	}
	return pipeline, pipeline.validate()
}

func NewTranslationPipeline(ctx context.Context, config backends.PipelineConfig[*TranslationPipeline], model *backends.Model) (*TranslationPipeline, error) {
	pipeline := &TranslationPipeline{BasePipeline: backends.NewBasePipeline(ctx, config, model), Generation: backends.Seq2SeqOptions{MaxLength: 128}}
	for _, option := range config.Options {
		if err := option(pipeline); err != nil {
			return nil, err
		}
	}
	if pipeline.LanguageCode != "" {
		pipeline.Prefix = pipeline.LanguageCode + ": "
	}
	return pipeline, pipeline.validate()
}

func NewText2TextGenerationPipeline(ctx context.Context, config backends.PipelineConfig[*Text2TextGenerationPipeline], model *backends.Model) (*Text2TextGenerationPipeline, error) {
	pipeline := &Text2TextGenerationPipeline{BasePipeline: backends.NewBasePipeline(ctx, config, model), Generation: backends.Seq2SeqOptions{MaxLength: 128}}
	for _, option := range config.Options {
		if err := option(pipeline); err != nil {
			return nil, err
		}
	}
	return pipeline, pipeline.validate()
}

func (p *SummarizationPipeline) IsGenerative() bool                     { return true }
func (p *TranslationPipeline) IsGenerative() bool                       { return true }
func (p *Text2TextGenerationPipeline) IsGenerative() bool               { return true }
func (p *SummarizationPipeline) GetModel() *backends.Model              { return p.Model }
func (p *TranslationPipeline) GetModel() *backends.Model                { return p.Model }
func (p *Text2TextGenerationPipeline) GetModel() *backends.Model        { return p.Model }
func (p *SummarizationPipeline) GetMetadata() backends.PipelineMetadata { return p.metadata() }

func (p *TranslationPipeline) GetMetadata() backends.PipelineMetadata { return p.metadata() }

func (p *Text2TextGenerationPipeline) GetMetadata() backends.PipelineMetadata { return p.metadata() }

func (p *SummarizationPipeline) GetStatistics() backends.PipelineStatistics { return p.getStatistics() }

func (p *TranslationPipeline) GetStatistics() backends.PipelineStatistics { return p.getStatistics() }

func (p *Text2TextGenerationPipeline) GetStatistics() backends.PipelineStatistics {
	return p.getStatistics()
}
func (p *SummarizationPipeline) Validate() error       { return p.validate() }
func (p *TranslationPipeline) Validate() error         { return p.validate() }
func (p *Text2TextGenerationPipeline) Validate() error { return p.validate() }
func (p *SummarizationPipeline) Run(ctx context.Context, inputs []string) (backends.PipelineBatchOutput, error) {
	return p.RunPipeline(ctx, inputs)
}

func (p *TranslationPipeline) Run(ctx context.Context, inputs []string) (backends.PipelineBatchOutput, error) {
	return p.RunPipeline(ctx, inputs)
}

func (p *Text2TextGenerationPipeline) Run(ctx context.Context, inputs []string) (backends.PipelineBatchOutput, error) {
	return p.RunPipeline(ctx, inputs)
}

func (p *SummarizationPipeline) RunPipeline(ctx context.Context, inputs []string) (*Seq2SeqOutput, error) {
	return p.run(ctx, inputs)
}

func (p *TranslationPipeline) RunPipeline(ctx context.Context, inputs []string) (*Seq2SeqOutput, error) {
	return p.run(ctx, inputs)
}

func (p *Text2TextGenerationPipeline) RunPipeline(ctx context.Context, inputs []string) (*Seq2SeqOutput, error) {
	return p.run(ctx, inputs)
}
