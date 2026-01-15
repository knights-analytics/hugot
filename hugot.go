package hugot

import (
	"errors"
	"fmt"
	"maps"

	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/pipelines"
)

// Session allows for the creation of new pipelines and holds the pipeline already created.
type Session struct {
	featureExtractionPipelines      pipelineMap[*pipelines.FeatureExtractionPipeline]
	tokenClassificationPipelines    pipelineMap[*pipelines.TokenClassificationPipeline]
	textClassificationPipelines     pipelineMap[*pipelines.TextClassificationPipeline]
	zeroShotClassificationPipelines pipelineMap[*pipelines.ZeroShotClassificationPipeline]
	crossEncoderPipelines           pipelineMap[*pipelines.CrossEncoderPipeline]
	imageClassificationPipelines    pipelineMap[*pipelines.ImageClassificationPipeline]
	objectDetectionPipelines        pipelineMap[*pipelines.ObjectDetectionPipeline]
	textGenerationPipelines         pipelineMap[*pipelines.TextGenerationPipeline]
	seq2seqPipelines                pipelineMap[*pipelines.Seq2SeqPipeline]
	glinerPipelines                 pipelineMap[*pipelines.GLiNERPipeline]
	models                          map[string]*backends.Model
	options                         *options.Options
	environmentDestroy              func() error
}

func newSession(backend string, opts ...options.WithOption) (*Session, error) {
	parsedOptions := options.Defaults()
	parsedOptions.Backend = backend
	// Collect options into a struct, so they can be applied in the correct order later
	if backend == "XLA" {
		parsedOptions.GoMLXOptions.XLA = true
	}
	for _, option := range opts {
		err := option(parsedOptions)
		if err != nil {
			return nil, err
		}
	}

	session := &Session{
		featureExtractionPipelines:      map[string]*pipelines.FeatureExtractionPipeline{},
		textClassificationPipelines:     map[string]*pipelines.TextClassificationPipeline{},
		tokenClassificationPipelines:    map[string]*pipelines.TokenClassificationPipeline{},
		zeroShotClassificationPipelines: map[string]*pipelines.ZeroShotClassificationPipeline{},
		crossEncoderPipelines:           map[string]*pipelines.CrossEncoderPipeline{},
		imageClassificationPipelines:    map[string]*pipelines.ImageClassificationPipeline{},
		objectDetectionPipelines:        map[string]*pipelines.ObjectDetectionPipeline{},
		textGenerationPipelines:         map[string]*pipelines.TextGenerationPipeline{},
		seq2seqPipelines:                map[string]*pipelines.Seq2SeqPipeline{},
		glinerPipelines:                 map[string]*pipelines.GLiNERPipeline{},
		models:                          map[string]*backends.Model{},
		options:                         parsedOptions,
		environmentDestroy: func() error {
			return nil
		},
	}

	return session, nil
}

type pipelineMap[T backends.Pipeline] map[string]T

func (m pipelineMap[T]) GetStatistics() map[string]backends.PipelineStatistics {
	statistics := map[string]backends.PipelineStatistics{}
	for pipelineName, p := range m {
		statistics[pipelineName] = p.GetStatistics()
	}
	return statistics
}

// FeatureExtractionConfig is the configuration for a feature extraction pipeline.
type FeatureExtractionConfig = backends.PipelineConfig[*pipelines.FeatureExtractionPipeline]

// FeatureExtractionOption is an option for a feature extraction pipeline.
type FeatureExtractionOption = backends.PipelineOption[*pipelines.FeatureExtractionPipeline]

// TextClassificationConfig is the configuration for a text classification pipeline.
type TextClassificationConfig = backends.PipelineConfig[*pipelines.TextClassificationPipeline]

// TextClassificationOption is an option for a text classification pipeline.
type TextClassificationOption = backends.PipelineOption[*pipelines.TextClassificationPipeline]

// ZeroShotClassificationConfig is the configuration for a zero shot classification pipeline.
type ZeroShotClassificationConfig = backends.PipelineConfig[*pipelines.ZeroShotClassificationPipeline]

// ZeroShotClassificationOption is an option for a zero shot classification pipeline.
type ZeroShotClassificationOption = backends.PipelineOption[*pipelines.ZeroShotClassificationPipeline]

// TokenClassificationConfig is the configuration for a token classification pipeline.
type TokenClassificationConfig = backends.PipelineConfig[*pipelines.TokenClassificationPipeline]

// TokenClassificationOption is an option for a token classification pipeline.
type TokenClassificationOption = backends.PipelineOption[*pipelines.TokenClassificationPipeline]

// CrossEncoderConfig is the configuration for a cross encoder pipeline.
type CrossEncoderConfig = backends.PipelineConfig[*pipelines.CrossEncoderPipeline]

// CrossEncoderOption is an option for a cross encoder pipeline.
type CrossEncoderOption = backends.PipelineOption[*pipelines.CrossEncoderPipeline]

// ImageClassificationConfig is the configuration for an image classification pipeline.
type ImageClassificationConfig = backends.PipelineConfig[*pipelines.ImageClassificationPipeline]

// ImageClassificationOption is an option for an image classification pipeline.
type ImageClassificationOption = backends.PipelineOption[*pipelines.ImageClassificationPipeline]

// ObjectDetectionConfig is the configuration for an object detection pipeline.
type ObjectDetectionConfig = backends.PipelineConfig[*pipelines.ObjectDetectionPipeline]

// ObjectDetectionOption is an option for an object detection pipeline.
type ObjectDetectionOption = backends.PipelineOption[*pipelines.ObjectDetectionPipeline]

// TextGenerationConfig is the configuration for a text generation pipeline.
type TextGenerationConfig = backends.PipelineConfig[*pipelines.TextGenerationPipeline]

// TextGenerationOption is an option for a text generation pipeline.
type TextGenerationOption = backends.PipelineOption[*pipelines.TextGenerationPipeline]

// Seq2SeqConfig is the configuration for a seq2seq (T5, doc2query, etc.) pipeline.
type Seq2SeqConfig = backends.PipelineConfig[*pipelines.Seq2SeqPipeline]

// Seq2SeqOption is an option for a seq2seq pipeline.
type Seq2SeqOption = backends.PipelineOption[*pipelines.Seq2SeqPipeline]

// GLiNERConfig is the configuration for a GLiNER (zero-shot NER) pipeline
type GLiNERConfig = backends.PipelineConfig[*pipelines.GLiNERPipeline]

// GLiNEROption is an option for a GLiNER pipeline
type GLiNEROption = backends.PipelineOption[*pipelines.GLiNERPipeline]

// NewPipeline can be used to create a new pipeline of type T. The initialised pipeline will be returned and it
// will also be stored in the session object so that all created pipelines can be destroyed with session.Destroy()
// at once.
func NewPipeline[T backends.Pipeline](s *Session, pipelineConfig backends.PipelineConfig[T]) (T, error) {
	var pipeline T
	if pipelineConfig.Name == "" {
		return pipeline, errors.New("a name for the pipeline is required")
	}

	_, getError := GetPipeline[T](s, pipelineConfig.Name)
	var notFoundError *pipelineNotFoundError
	if getError == nil {
		return pipeline, fmt.Errorf("pipeline %s has already been initialised", pipelineConfig.Name)
	} else if !errors.As(getError, &notFoundError) {
		return pipeline, getError
	}

	var err error
	var name string
	var model *backends.Model

	// Check if this is a Seq2SeqPipeline - it manages its own encoder/decoder models
	// and should not go through the standard model loading path
	_, isSeq2Seq := any(pipeline).(*pipelines.Seq2SeqPipeline)

	if !isSeq2Seq {
		// Load model if it has not been loaded already (for non-Seq2Seq pipelines)
		// Use combined ModelPath:OnnxFilename as key to allow multiple pipelines on same model path
		modelID := pipelineConfig.ModelPath + ":" + pipelineConfig.OnnxFilename
		var ok bool
		model, ok = s.models[modelID]

		if !ok {
			model, err = backends.LoadModel(pipelineConfig.ModelPath, pipelineConfig.OnnxFilename, s.options, pipeline.IsGenerative())
			if err != nil {
				return pipeline, err
			}
			s.models[modelID] = model
		}
	}

	pipeline, name, err = InitializePipeline(pipeline, pipelineConfig, s.options, model)
	if err != nil {
		return pipeline, err
	}

	switch typedPipeline := any(pipeline).(type) {
	case *pipelines.TokenClassificationPipeline:
		s.tokenClassificationPipelines[name] = typedPipeline
	case *pipelines.TextClassificationPipeline:
		s.textClassificationPipelines[name] = typedPipeline
	case *pipelines.FeatureExtractionPipeline:
		s.featureExtractionPipelines[name] = typedPipeline
	case *pipelines.ZeroShotClassificationPipeline:
		s.zeroShotClassificationPipelines[name] = typedPipeline
	case *pipelines.CrossEncoderPipeline:
		s.crossEncoderPipelines[name] = typedPipeline
	case *pipelines.ImageClassificationPipeline:
		s.imageClassificationPipelines[name] = typedPipeline
	case *pipelines.ObjectDetectionPipeline:
		s.objectDetectionPipelines[name] = typedPipeline
	case *pipelines.TextGenerationPipeline:
		s.textGenerationPipelines[name] = typedPipeline
	case *pipelines.Seq2SeqPipeline:
		s.seq2seqPipelines[name] = typedPipeline
	case *pipelines.GLiNERPipeline:
		s.glinerPipelines[name] = typedPipeline
	default:
		return pipeline, fmt.Errorf("pipeline type not supported: %T", typedPipeline)
	}
	return pipeline, nil
}

func InitializePipeline[T backends.Pipeline](p T, pipelineConfig backends.PipelineConfig[T], options *options.Options, model *backends.Model) (T, string, error) {
	var pipeline T
	var name string

	switch any(p).(type) {
	case *pipelines.TokenClassificationPipeline:
		config := any(pipelineConfig).(backends.PipelineConfig[*pipelines.TokenClassificationPipeline])
		pipelineInitialised, err := pipelines.NewTokenClassificationPipeline(config, options, model)
		if err != nil {
			return pipeline, name, err
		}
		pipeline = any(pipelineInitialised).(T)
		name = config.Name
	case *pipelines.TextClassificationPipeline:
		config := any(pipelineConfig).(backends.PipelineConfig[*pipelines.TextClassificationPipeline])
		pipelineInitialised, err := pipelines.NewTextClassificationPipeline(config, options, model)
		if err != nil {
			return pipeline, name, err
		}
		pipeline = any(pipelineInitialised).(T)
		name = config.Name
	case *pipelines.FeatureExtractionPipeline:
		config := any(pipelineConfig).(backends.PipelineConfig[*pipelines.FeatureExtractionPipeline])
		pipelineInitialised, err := pipelines.NewFeatureExtractionPipeline(config, options, model)
		if err != nil {
			return pipeline, name, err
		}
		pipeline = any(pipelineInitialised).(T)
		name = config.Name
	case *pipelines.ZeroShotClassificationPipeline:
		config := any(pipelineConfig).(backends.PipelineConfig[*pipelines.ZeroShotClassificationPipeline])
		pipelineInitialised, err := pipelines.NewZeroShotClassificationPipeline(config, options, model)
		if err != nil {
			return pipeline, name, err
		}
		pipeline = any(pipelineInitialised).(T)
		name = config.Name
	case *pipelines.CrossEncoderPipeline:
		config := any(pipelineConfig).(backends.PipelineConfig[*pipelines.CrossEncoderPipeline])
		pipelineInitialised, err := pipelines.NewCrossEncoderPipeline(config, options, model)
		if err != nil {
			return pipeline, name, err
		}
		pipeline = any(pipelineInitialised).(T)
		name = config.Name
	case *pipelines.ImageClassificationPipeline:
		config := any(pipelineConfig).(backends.PipelineConfig[*pipelines.ImageClassificationPipeline])
		pipelineInitialised, err := pipelines.NewImageClassificationPipeline(config, options, model)
		if err != nil {
			return pipeline, name, err
		}
		pipeline = any(pipelineInitialised).(T)
		name = config.Name
	case *pipelines.ObjectDetectionPipeline:
		config := any(pipelineConfig).(backends.PipelineConfig[*pipelines.ObjectDetectionPipeline])
		pipelineInitialised, err := pipelines.NewObjectDetectionPipeline(config, options, model)
		if err != nil {
			return pipeline, name, err
		}
		pipeline = any(pipelineInitialised).(T)
		name = config.Name
	case *pipelines.TextGenerationPipeline:
		config := any(pipelineConfig).(backends.PipelineConfig[*pipelines.TextGenerationPipeline])
		pipelineInitialised, err := pipelines.NewTextGenerationPipeline(config, options, model)
		if err != nil {
			return pipeline, name, err
		}
		pipeline = any(pipelineInitialised).(T)
		name = config.Name
	case *pipelines.Seq2SeqPipeline:
		// Seq2SeqPipeline is special: it loads its own encoder/decoder models
		// The model parameter is ignored (passed as nil) since it manages its own models
		config := any(pipelineConfig).(backends.PipelineConfig[*pipelines.Seq2SeqPipeline])
		pipelineInitialised, err := pipelines.NewSeq2SeqPipeline(config, options)
		if err != nil {
			return pipeline, name, err
		}
		pipeline = any(pipelineInitialised).(T)
		name = config.Name
		// Don't add to model.Pipelines since Seq2SeqPipeline manages its own models
		return pipeline, name, nil
	case *pipelines.GLiNERPipeline:
		config := any(pipelineConfig).(backends.PipelineConfig[*pipelines.GLiNERPipeline])
		pipelineInitialised, err := pipelines.NewGLiNERPipeline(config, options, model)
		if err != nil {
			return pipeline, name, err
		}
		pipeline = any(pipelineInitialised).(T)
		name = config.Name
	default:
		return pipeline, name, fmt.Errorf("not implemented")
	}

	model.Pipelines[name] = pipeline
	return pipeline, name, nil
}

// GetPipeline can be used to retrieve a pipeline of type T with the given name from the session.
func GetPipeline[T backends.Pipeline](s *Session, name string) (T, error) {
	var pipeline T
	switch any(pipeline).(type) {
	case *pipelines.TokenClassificationPipeline:
		p, ok := s.tokenClassificationPipelines[name]
		if !ok {
			return pipeline, &pipelineNotFoundError{pipelineName: name}
		}
		return any(p).(T), nil
	case *pipelines.TextClassificationPipeline:
		p, ok := s.textClassificationPipelines[name]
		if !ok {
			return pipeline, &pipelineNotFoundError{pipelineName: name}
		}
		return any(p).(T), nil
	case *pipelines.FeatureExtractionPipeline:
		p, ok := s.featureExtractionPipelines[name]
		if !ok {
			return pipeline, &pipelineNotFoundError{pipelineName: name}
		}
		return any(p).(T), nil
	case *pipelines.ZeroShotClassificationPipeline:
		p, ok := s.zeroShotClassificationPipelines[name]
		if !ok {
			return pipeline, &pipelineNotFoundError{pipelineName: name}
		}
		return any(p).(T), nil
	case *pipelines.CrossEncoderPipeline:
		p, ok := s.crossEncoderPipelines[name]
		if !ok {
			return pipeline, &pipelineNotFoundError{pipelineName: name}
		}
		return any(p).(T), nil
	case *pipelines.ImageClassificationPipeline:
		p, ok := s.imageClassificationPipelines[name]
		if !ok {
			return pipeline, &pipelineNotFoundError{pipelineName: name}
		}
		return any(p).(T), nil
	case *pipelines.ObjectDetectionPipeline:
		p, ok := s.objectDetectionPipelines[name]
		if !ok {
			return pipeline, &pipelineNotFoundError{pipelineName: name}
		}
		return any(p).(T), nil
	case *pipelines.TextGenerationPipeline:
		p, ok := s.textGenerationPipelines[name]
		if !ok {
			return pipeline, &pipelineNotFoundError{pipelineName: name}
		}
		return any(p).(T), nil
	case *pipelines.Seq2SeqPipeline:
		p, ok := s.seq2seqPipelines[name]
		if !ok {
			return pipeline, &pipelineNotFoundError{pipelineName: name}
		}
		return any(p).(T), nil
	case *pipelines.GLiNERPipeline:
		p, ok := s.glinerPipelines[name]
		if !ok {
			return pipeline, &pipelineNotFoundError{pipelineName: name}
		}
		return any(p).(T), nil
	default:
		return pipeline, errors.New("pipeline type not supported")
	}
}

func ClosePipeline[T backends.Pipeline](s *Session, name string) error {
	var pipeline T
	switch any(pipeline).(type) {
	case *pipelines.TokenClassificationPipeline:
		p, ok := s.tokenClassificationPipelines[name]
		if ok {
			model := p.Model
			delete(s.tokenClassificationPipelines, name)
			delete(model.Pipelines, name)
			if len(model.Pipelines) == 0 {
				delete(s.models, model.ID)
				return model.Destroy()
			}
		}
	case *pipelines.TextClassificationPipeline:
		p, ok := s.textClassificationPipelines[name]
		if ok {
			model := p.Model
			delete(s.textClassificationPipelines, name)
			delete(model.Pipelines, name)
			if len(model.Pipelines) == 0 {
				delete(s.models, model.ID)
				return model.Destroy()
			}
		}
	case *pipelines.FeatureExtractionPipeline:
		p, ok := s.featureExtractionPipelines[name]
		if ok {
			model := p.Model
			delete(s.featureExtractionPipelines, name)
			delete(model.Pipelines, name)
			if len(model.Pipelines) == 0 {
				delete(s.models, model.ID)
				return model.Destroy()
			}
		}
	case *pipelines.ZeroShotClassificationPipeline:
		p, ok := s.zeroShotClassificationPipelines[name]
		if ok {
			model := p.Model
			delete(s.zeroShotClassificationPipelines, name)
			delete(model.Pipelines, name)
			if len(model.Pipelines) == 0 {
				delete(s.models, model.ID)
				return model.Destroy()
			}
		}
	case *pipelines.CrossEncoderPipeline:
		p, ok := s.crossEncoderPipelines[name]
		if ok {
			model := p.Model
			delete(s.crossEncoderPipelines, name)
			delete(model.Pipelines, name)
			if len(model.Pipelines) == 0 {
				delete(s.models, model.ID)
				return model.Destroy()
			}
		}
	case *pipelines.ImageClassificationPipeline:
		p, ok := s.imageClassificationPipelines[name]
		if ok {
			model := p.Model
			delete(s.imageClassificationPipelines, name)
			delete(model.Pipelines, name)
			if len(model.Pipelines) == 0 {
				delete(s.models, model.ID)
				return model.Destroy()
			}
		}
	case *pipelines.ObjectDetectionPipeline:
		p, ok := s.objectDetectionPipelines[name]
		if ok {
			model := p.Model
			delete(s.objectDetectionPipelines, name)
			delete(model.Pipelines, name)
			if len(model.Pipelines) == 0 {
				delete(s.models, model.ID)
				return model.Destroy()
			}
		}
	case *pipelines.TextGenerationPipeline:
		p, ok := s.textGenerationPipelines[name]
		if ok {
			model := p.Model
			delete(s.textGenerationPipelines, name)
			delete(model.Pipelines, name)
			if len(model.Pipelines) == 0 {
				delete(s.models, model.ID)
				return model.Destroy()
			}
		}
	case *pipelines.Seq2SeqPipeline:
		// Seq2SeqPipeline manages its own models, so we just destroy the pipeline
		p, ok := s.seq2seqPipelines[name]
		if ok {
			delete(s.seq2seqPipelines, name)
			return p.Destroy()
		}
	case *pipelines.GLiNERPipeline:
		p, ok := s.glinerPipelines[name]
		if ok {
			model := p.Model
			delete(s.glinerPipelines, name)
			delete(model.Pipelines, name)
			if len(model.Pipelines) == 0 {
				delete(s.models, model.Path)
				return model.Destroy()
			}
		}
	default:
		return errors.New("pipeline type not supported")
	}
	return nil
}

type pipelineNotFoundError struct {
	pipelineName string
}

func (e *pipelineNotFoundError) Error() string {
	return fmt.Sprintf("Pipeline with name %s not found", e.pipelineName)
}

// GetStatistics returns runtime statistics for all initialized pipelines for profiling purposes. We currently record for each pipeline:
// the total runtime of the tokenization step
// the number of batch calls to the tokenization step
// the average time per tokenization batch call
// the total runtime of the inference (i.e. onnxruntime) step
// the average time per onnxruntime inference batch call.
func (s *Session) GetStatistics() map[string]backends.PipelineStatistics {
	statistics := map[string]backends.PipelineStatistics{}
	maps.Copy(statistics, s.tokenClassificationPipelines.GetStatistics())
	maps.Copy(statistics, s.textClassificationPipelines.GetStatistics())
	maps.Copy(statistics, s.featureExtractionPipelines.GetStatistics())
	maps.Copy(statistics, s.imageClassificationPipelines.GetStatistics())
	maps.Copy(statistics, s.zeroShotClassificationPipelines.GetStatistics())
	maps.Copy(statistics, s.crossEncoderPipelines.GetStatistics())
	maps.Copy(statistics, s.textGenerationPipelines.GetStatistics())
	maps.Copy(statistics, s.seq2seqPipelines.GetStatistics())
	maps.Copy(statistics, s.glinerPipelines.GetStatistics())
	return statistics
}

// PrintStatistics prints runtime statistics for all initialized pipelines to stdout.
func (s *Session) PrintStatistics() {
	statistics := s.GetStatistics()
	for pipelineName, v := range statistics {
		fmt.Printf("Statistics for pipeline %s:\n", pipelineName)
		v.Print()
	}
}

// Destroy deletes the hugot session and onnxruntime environment and all initialized pipelines, freeing memory.
// A hugot session should be destroyed when not neeeded any more, preferably with a defer() call.
func (s *Session) Destroy() error {
	var err error
	for _, model := range s.models {
		err = errors.Join(err, model.Destroy())
	}
	// Seq2SeqPipelines manage their own models, destroy them separately
	for _, pipeline := range s.seq2seqPipelines {
		err = errors.Join(err, pipeline.Destroy())
	}
	s.models = nil
	s.featureExtractionPipelines = nil
	s.tokenClassificationPipelines = nil
	s.textClassificationPipelines = nil
	s.imageClassificationPipelines = nil
	s.zeroShotClassificationPipelines = nil
	s.textGenerationPipelines = nil
	s.crossEncoderPipelines = nil
	s.seq2seqPipelines = nil
	s.glinerPipelines = nil

	if s.options != nil {
		err = errors.Join(err, s.options.Destroy())
		s.options = nil
	}

	err = errors.Join(err, s.environmentDestroy())
	return err
}
