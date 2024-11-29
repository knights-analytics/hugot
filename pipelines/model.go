package pipelines

type Model struct {
	Path         string
	OnnxFilename string
	OnnxBytes    []byte
	ORTModel     *ORTModel
	XLAModel     *XLAModel
	Tokenizer    *Tokenizer
	InputsMeta   []InputOutputInfo
	OutputsMeta  []InputOutputInfo
	Destroy      func() error
}
