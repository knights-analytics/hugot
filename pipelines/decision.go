package pipelines

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/gomlx/go-huggingface/tokenizers/api"
	"github.com/gomlx/go-huggingface/tokenizers/hftokenizer"

	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/util/fileutil"
	"github.com/knights-analytics/hugot/util/safeconv"
)

// DecisionQuestionType is one of the three decision primitives.
type DecisionQuestionType string

const (
	// DecisionNoul is a yes/no question: is the statement true of the state?
	DecisionNoul DecisionQuestionType = "noul"
	// DecisionChoice selects exactly one of N named options.
	DecisionChoice DecisionQuestionType = "choice"
	// DecisionScore rates the state on an ordered scale of 2 to 10 levels.
	DecisionScore DecisionQuestionType = "score"
)

// DecisionCriterion is one option of a question.
//
// For a choice question, Name is the option's label and Description is optional.
// For a score question, Description is the level's rubric; Name is ignored, since
// levels are numbered 0..L-1 lowest first. For a noul question, criteria are optional
// and override the default description of "false" and/or "true" by Name.
type DecisionCriterion struct {
	Name        string
	Description string
}

// DecisionQuestion is one question asked of a state.
type DecisionQuestion struct {
	Type         DecisionQuestionType
	Instructions string
	Criteria     []DecisionCriterion
}

// DecisionInput is a state and the questions to ask of it. Use RenderDecisionState
// to turn a structured state into text.
type DecisionInput struct {
	State     string
	Questions map[string]DecisionQuestion
}

// DecisionAnswer is the model's answer to one question.
type DecisionAnswer struct {
	Type DecisionQuestionType
	// Probabilities over the options, keyed by option name: "false"/"true" for noul,
	// the option names for choice, the level index ("0", "1", ...) for score.
	Probabilities map[string]float64
	// Noul is P(true), set for noul questions.
	Noul float64
	// Choice is the most probable option, set for choice questions.
	Choice string
	// Score is the expected level index, set for score questions.
	Score float64
	// Confidence is 0 for a uniform and 1 for a one-hot distribution (choice), or how
	// tightly the mass clusters on the scale (score). Noul has none: P(true) is it.
	Confidence float64
}

// DecisionResult holds one input's answers, keyed by question name.
type DecisionResult struct {
	Answers map[string]DecisionAnswer
}

// DecisionOutput holds one result per input, in input order.
type DecisionOutput struct {
	Results []DecisionResult
}

func (o *DecisionOutput) GetOutput() []any {
	out := make([]any, len(o.Results))
	for i, r := range o.Results {
		out[i] = r
	}
	return out
}

// DecisionReadoutConfig holds the readout settings a decision model ships with: the
// fitted temperature per question type, the ordinal smoothing the score confidence
// corrects for, and the context length. Strands models ship it as strands_decider_config.json.
type DecisionReadoutConfig struct {
	Temperature       float64            `json:"temperature"`
	TemperatureByKind map[string]float64 `json:"temperature_by_kind"`
	OrdinalSmoothing  float64            `json:"ordinal_smoothing"`
	MaxLength         int                `json:"max_length"`
	ForceBOS          bool               `json:"force_bos"`
	HeadType          string             `json:"head_type"`
}

// RenderDecisionState renders a structured state as decision models take a JSON state:
// indented JSON with non-ASCII characters left as they are. Object keys
// follow Go's encoding (struct field order, or sorted map keys), so use a struct when
// key order matters.
func RenderDecisionState(v any) (string, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	if err := enc.Encode(v); err != nil {
		return "", err
	}
	return strings.TrimSuffix(buf.String(), "\n"), nil
}

const (
	decisionDefaultMaxBatchRows   = 8
	decisionDefaultMaxBatchTokens = 16384

	decisionMaxChoiceOptions = 255
	decisionMinScoreLevels   = 2
	decisionMaxScoreLevels   = 10
)

// DecisionPipeline runs decision models ("system one" models, as served behind the Jev
// API): instead of generating text, the model answers typed questions about a state (a
// text, or JSON). The question types are noul (yes/no), choice (one of N named options)
// and score (a level on an ordered scale). Options are given at request time, so any
// label set works without retraining.
//
// The ONNX graph takes input_ids, attention_mask and option_index ([batch, options],
// the position of each option's last token) and returns per-option logits
// ([batch, options]); temperature, softmax and confidence are applied here. Prompt
// rendering follows the model's format; the Strands format (the strands* declarations at
// the end of this file) is the one supported today.
type DecisionPipeline struct {
	*backends.BasePipeline
	// Questions are asked of every input passed to Run.
	Questions map[string]DecisionQuestion
	// MaxBatchRows caps how many (state, question) rows run in one forward pass.
	MaxBatchRows int
	// MaxBatchTokens caps a forward pass's padded size (rows x longest row), so a long
	// state is not batched with short ones it would pad to its own length. A row longer
	// than the budget still runs, alone.
	MaxBatchTokens int
	Config         DecisionReadoutConfig

	stateTokenizer    *hftokenizer.Tokenizer
	questionTokenizer *hftokenizer.Tokenizer
	bosID, padID      int
}

// WithDecisionQuestions sets the questions Run asks of every input.
func WithDecisionQuestions(questions map[string]DecisionQuestion) backends.PipelineOption[*DecisionPipeline] {
	return func(p *DecisionPipeline) error {
		p.Questions = questions
		return nil
	}
}

// WithDecisionMaxBatchRows caps the (state, question) rows per forward pass.
func WithDecisionMaxBatchRows(n int) backends.PipelineOption[*DecisionPipeline] {
	return func(p *DecisionPipeline) error {
		if n <= 0 {
			return errors.New("decision max batch rows must be positive")
		}
		p.MaxBatchRows = n
		return nil
	}
}

// WithDecisionMaxBatchTokens caps a forward pass's padded size, rows times longest row.
func WithDecisionMaxBatchTokens(n int) backends.PipelineOption[*DecisionPipeline] {
	return func(p *DecisionPipeline) error {
		if n <= 0 {
			return errors.New("decision max batch tokens must be positive")
		}
		p.MaxBatchTokens = n
		return nil
	}
}

func NewDecisionPipeline(ctx context.Context, config backends.PipelineConfig[*DecisionPipeline], model *backends.Model) (*DecisionPipeline, error) {
	if model == nil {
		return nil, errors.New("decision pipeline requires a model")
	}
	if model.ORTModel == nil {
		return nil, errors.New("decision models currently require the ORT backend")
	}
	for _, name := range []string{"input_ids", "attention_mask", "option_index"} {
		if !tensorPresent(model.InputsMeta, name) {
			return nil, fmt.Errorf("decision model is missing input %q", name)
		}
	}
	if !tensorPresent(model.OutputsMeta, "logits") {
		return nil, errors.New("decision model is missing output \"logits\"")
	}
	p := &DecisionPipeline{BasePipeline: backends.NewBasePipeline(ctx, config, model), MaxBatchRows: decisionDefaultMaxBatchRows, MaxBatchTokens: decisionDefaultMaxBatchTokens}
	for _, option := range config.Options {
		if err := option(p); err != nil {
			return nil, err
		}
	}

	raw, err := fileutil.ReadFileBytes(ctx, fileutil.PathJoinSafe(model.Path, strandsConfigFilename))
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", strandsConfigFilename, err)
	}
	if err := json.Unmarshal(raw, &p.Config); err != nil {
		return nil, fmt.Errorf("parse %s: %w", strandsConfigFilename, err)
	}
	if p.Config.HeadType != "" && p.Config.HeadType != "pointer" {
		return nil, fmt.Errorf("decision head type %q is not supported; only the pointer head is", p.Config.HeadType)
	}

	tk, err := fileutil.ReadFileBytes(ctx, fileutil.PathJoinSafe(model.Path, "tokenizer.json"))
	if err != nil {
		return nil, fmt.Errorf("read tokenizer.json: %w", err)
	}
	// The state is encoded with special tokens (BOS), each question without, and the
	// ids are concatenated: never the prompt as one string, which tokenizes differently.
	if p.stateTokenizer, err = newDecisionTokenizer(tk, api.EncodeOptions{AddSpecialTokens: true}); err != nil {
		return nil, err
	}
	if p.questionTokenizer, err = newDecisionTokenizer(tk, api.EncodeOptions{IncludeSpans: true}); err != nil {
		return nil, err
	}
	vocab := p.stateTokenizer.GetVocab()
	p.bosID = -1
	if id, ok := vocab["<bos>"]; ok {
		p.bosID = id
	}
	if p.Config.ForceBOS && p.bosID < 0 {
		return nil, errors.New("decision config sets force_bos but the tokenizer has no <bos> token")
	}
	if id, ok := vocab["<pad>"]; ok {
		p.padID = id
	}
	if err := p.Validate(); err != nil {
		return nil, err
	}
	return p, nil
}

func newDecisionTokenizer(content []byte, opts api.EncodeOptions) (*hftokenizer.Tokenizer, error) {
	tk, err := hftokenizer.NewFromContent(nil, content)
	if err != nil {
		return nil, fmt.Errorf("load tokenizer: %w", err)
	}
	if err := tk.With(opts); err != nil {
		return nil, err
	}
	return tk, nil
}

func (*DecisionPipeline) IsGenerative() bool          { return false }
func (p *DecisionPipeline) GetModel() *backends.Model { return p.Model }
func (p *DecisionPipeline) GetMetadata() backends.PipelineMetadata {
	return nativeMetadata(p.Model)
}
func (p *DecisionPipeline) GetStatistics() backends.PipelineStatistics {
	return nativeStatistics(p.BasePipeline)
}

func (p *DecisionPipeline) Validate() error {
	if p == nil || p.BasePipeline == nil || p.Model == nil {
		return errors.New("decision pipeline requires a model")
	}
	if p.stateTokenizer == nil || p.questionTokenizer == nil {
		return errors.New("decision pipeline tokenizer is not loaded")
	}
	if p.MaxBatchRows <= 0 || p.MaxBatchTokens <= 0 {
		return errors.New("decision batch limits must be positive")
	}
	for name, q := range p.Questions {
		if err := q.validate(); err != nil {
			return fmt.Errorf("question %q: %w", name, err)
		}
	}
	return nil
}

// Run asks the pipeline's configured questions (WithDecisionQuestions) of each input.
func (p *DecisionPipeline) Run(ctx context.Context, inputs []string) (backends.PipelineBatchOutput, error) {
	if len(p.Questions) == 0 {
		return nil, errors.New("decision Run needs questions: set them with WithDecisionQuestions, or call RunPipeline")
	}
	in := make([]DecisionInput, len(inputs))
	for i, s := range inputs {
		in[i] = DecisionInput{State: s, Questions: p.Questions}
	}
	return p.RunPipeline(ctx, in)
}

// RunPipeline answers each input's questions. Every (state, question) pair is one row;
// rows of similar length are batched together.
func (p *DecisionPipeline) RunPipeline(ctx context.Context, inputs []DecisionInput) (*DecisionOutput, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	rows, err := p.encodeRows(inputs)
	if err != nil {
		return nil, err
	}
	out := &DecisionOutput{Results: make([]DecisionResult, len(inputs))}
	for i := range out.Results {
		out.Results[i].Answers = make(map[string]DecisionAnswer, len(inputs[i].Questions))
	}
	// Ascending length: each chunk's last row is its longest, so the padded size of a
	// chunk grown by one row is (rows + 1) x that row's length.
	sort.SliceStable(rows, func(a, b int) bool { return len(rows[a].ids) < len(rows[b].ids) })
	for start := 0; start < len(rows); {
		end := start + 1
		for end < len(rows) && end-start < p.MaxBatchRows && (end-start+1)*len(rows[end].ids) <= p.MaxBatchTokens {
			end++
		}
		if err := p.runChunk(ctx, rows[start:end], out); err != nil {
			return nil, err
		}
		start = end
	}
	return out, nil
}

type decisionRow struct {
	input    int
	question string
	ids      []int64
	options  []int64
	rq       renderedQuestion
}

func (p *DecisionPipeline) encodeRows(inputs []DecisionInput) ([]decisionRow, error) {
	if p.BasePipeline != nil && p.TokenizerTimings != nil {
		start := time.Now()
		defer func() {
			atomic.AddUint64(&p.TokenizerTimings.NumCalls, 1)
			atomic.AddUint64(&p.TokenizerTimings.TotalNS, safeconv.DurationToU64(time.Since(start)))
		}()
	}
	type encodedQuestion struct {
		ids     []int64
		options []int
		rq      renderedQuestion
	}
	questions := map[string]encodedQuestion{} // by rendered text: shared across inputs
	var rows []decisionRow
	for i, in := range inputs {
		if len(in.Questions) == 0 {
			return nil, fmt.Errorf("input %d has no questions", i)
		}
		state := p.encodeState(in.State)
		names := make([]string, 0, len(in.Questions))
		for name := range in.Questions {
			names = append(names, name)
		}
		slices.Sort(names)
		for _, name := range names {
			rq, err := renderStrandsQuestion(in.Questions[name])
			if err != nil {
				return nil, fmt.Errorf("input %d, question %q: %w", i, name, err)
			}
			eq, ok := questions[rq.text]
			if !ok {
				enc := p.questionTokenizer.EncodeWithAnnotations(rq.text)
				offsets := make([][2]int, len(enc.Spans))
				for j, s := range enc.Spans {
					offsets[j] = [2]int{s.Start, s.End}
				}
				opts, err := strandsOptionTokenIndex(offsets, rq.spans)
				if err != nil {
					return nil, fmt.Errorf("input %d, question %q: %w", i, name, err)
				}
				eq = encodedQuestion{ids: toInt64(enc.IDs), options: opts, rq: rq}
				questions[rq.text] = eq
			}
			n := len(state) + len(eq.ids)
			if p.Config.MaxLength > 0 && n > p.Config.MaxLength {
				return nil, fmt.Errorf("input %d, question %q: prompt of %d tokens (%d state + %d question) exceeds the model's %d",
					i, name, n, len(state), len(eq.ids), p.Config.MaxLength)
			}
			ids := make([]int64, 0, n)
			ids = append(append(ids, state...), eq.ids...)
			options := make([]int64, len(eq.options))
			for k, o := range eq.options {
				options[k] = int64(len(state) + o)
			}
			rows = append(rows, decisionRow{input: i, question: name, ids: ids, options: options, rq: eq.rq})
		}
	}
	return rows, nil
}

func (p *DecisionPipeline) encodeState(state string) []int64 {
	ids := p.stateTokenizer.Encode(renderStrandsState(state))
	if p.Config.ForceBOS && (len(ids) == 0 || ids[0] != p.bosID) {
		ids = append([]int{p.bosID}, ids...)
	}
	return toInt64(ids)
}

// runChunk runs one right-padded batch. Padded option slots point at position 0; their
// logits are never read.
func (p *DecisionPipeline) runChunk(ctx context.Context, chunk []decisionRow, out *DecisionOutput) error {
	seq, opts := 0, 0
	for _, r := range chunk {
		seq = max(seq, len(r.ids))
		opts = max(opts, len(r.options))
	}
	b := len(chunk)
	ids := make([]int64, b*seq)
	mask := make([]int64, b*seq)
	optIdx := make([]int64, b*opts)
	for i, r := range chunk {
		for j := range seq {
			if j < len(r.ids) {
				ids[i*seq+j] = r.ids[j]
				mask[i*seq+j] = 1
			} else {
				ids[i*seq+j] = int64(p.padID)
			}
		}
		copy(optIdx[i*opts:], r.options)
	}
	res, err := runNativeTensors(ctx, p.BasePipeline, p.Model, map[string]backends.Tensor{
		"input_ids":      {Shape: []int64{int64(b), int64(seq)}, Data: ids},
		"attention_mask": {Shape: []int64{int64(b), int64(seq)}, Data: mask},
		"option_index":   {Shape: []int64{int64(b), int64(opts)}, Data: optIdx},
	})
	if err != nil {
		return err
	}
	logits, ok := res["logits"].Data.([]float32)
	if !ok || len(logits) != b*opts {
		return fmt.Errorf("decision logits: expected %d float32 values, got %T of shape %v", b*opts, res["logits"].Data, res["logits"].Shape)
	}
	for i, r := range chunk {
		ans, err := readDecision(logits[i*opts:(i+1)*opts], r.rq, p.Config)
		if err != nil {
			return fmt.Errorf("input %d, question %q: %w", r.input, r.question, err)
		}
		out.Results[r.input].Answers[r.question] = ans
	}
	return nil
}

func toInt64(ids []int) []int64 {
	out := make([]int64, len(ids))
	for i, id := range ids {
		out[i] = int64(id)
	}
	return out
}

func (q DecisionQuestion) validate() error {
	switch q.Type {
	case DecisionNoul:
		seen := map[string]bool{}
		for _, c := range q.Criteria {
			if c.Name != "true" && c.Name != "false" {
				return fmt.Errorf("noul criteria names must be \"true\" or \"false\", got %q", c.Name)
			}
			if seen[c.Name] {
				return fmt.Errorf("noul criterion %q given twice", c.Name)
			}
			seen[c.Name] = true
		}
	case DecisionChoice:
		if len(q.Criteria) < 2 || len(q.Criteria) > decisionMaxChoiceOptions {
			return fmt.Errorf("choice needs 2 to %d options, got %d", decisionMaxChoiceOptions, len(q.Criteria))
		}
		seen := map[string]bool{}
		for _, c := range q.Criteria {
			name := strings.TrimSpace(c.Name)
			if name == "" || strings.ContainsAny(c.Name, "\n\r") {
				return fmt.Errorf("choice option names must be non-empty single lines, got %q", c.Name)
			}
			if seen[c.Name] {
				return fmt.Errorf("choice option %q given twice", c.Name)
			}
			seen[c.Name] = true
		}
	case DecisionScore:
		if len(q.Criteria) < decisionMinScoreLevels || len(q.Criteria) > decisionMaxScoreLevels {
			return fmt.Errorf("score needs %d to %d levels, got %d", decisionMinScoreLevels, decisionMaxScoreLevels, len(q.Criteria))
		}
	default:
		return fmt.Errorf("unknown question type %q", q.Type)
	}
	return nil
}

func (c DecisionReadoutConfig) temperature(kind DecisionQuestionType) float64 {
	if t, ok := c.TemperatureByKind[string(kind)]; ok && t > 0 {
		return t
	}
	if c.Temperature > 0 {
		return c.Temperature
	}
	return 1
}

// readDecision turns one row's raw head logits into an answer.
func readDecision(logits []float32, rq renderedQuestion, cfg DecisionReadoutConfig) (DecisionAnswer, error) {
	if len(logits) < len(rq.labels) {
		return DecisionAnswer{}, errors.New("fewer logits than options")
	}
	p := softmaxTemperature(logits[:len(rq.labels)], cfg.temperature(rq.kind))
	ans := DecisionAnswer{Type: rq.kind, Probabilities: make(map[string]float64, len(p))}
	for i, label := range rq.labels {
		ans.Probabilities[label] = p[i]
	}
	switch rq.kind {
	case DecisionNoul:
		ans.Noul = ans.Probabilities["true"]
	case DecisionChoice:
		best := 0
		for i := range p {
			if p[i] > p[best] {
				best = i
			}
		}
		ans.Choice = rq.labels[best]
		ans.Confidence = choiceConfidence(p)
	case DecisionScore:
		for i, x := range p {
			ans.Score += float64(i) * x
		}
		ans.Confidence = scoreConfidence(p, cfg.OrdinalSmoothing)
	}
	return ans, nil
}

func softmaxTemperature(logits []float32, t float64) []float64 {
	p := make([]float64, len(logits))
	maxZ := math.Inf(-1)
	for i, l := range logits {
		p[i] = float64(l) / t
		maxZ = math.Max(maxZ, p[i])
	}
	sum := 0.0
	for i := range p {
		p[i] = math.Exp(p[i] - maxZ)
		sum += p[i]
	}
	for i := range p {
		p[i] /= sum
	}
	return p
}

func choiceConfidence(p []float64) float64 {
	n := float64(len(p))
	if n <= 1 {
		return 1
	}
	pmax := 0.0
	for _, x := range p {
		pmax = math.Max(pmax, x)
	}
	return clamp01((n*pmax - 1) / (n - 1))
}

func scoreConfidence(p []float64, smoothing float64) float64 {
	n := len(p)
	if n <= 1 {
		return 1
	}
	mean := 0.0
	for i, x := range p {
		mean += float64(i) * x
	}
	variance := 0.0
	for i, x := range p {
		variance += x * (float64(i) - mean) * (float64(i) - mean)
	}
	sigma := math.Sqrt(variance)
	sigmaMax := float64(n-1) / 2
	sigmaFloor := 0.0
	if smoothing > 0 {
		sigmaFloor = math.Sqrt(smoothing)
	}
	if sigmaMax <= sigmaFloor {
		if sigma <= sigmaFloor {
			return 1
		}
		return 0
	}
	return clamp01((sigmaMax - sigma) / (sigmaMax - sigmaFloor))
}

func clamp01(x float64) float64 { return math.Max(0, math.Min(1, x)) }

// The Strands format for decision models (github.com/strands-labs/strands-decider,
// src/strands_decider/prompting.py): how a state and a question become token ids, and
// where each option's last token sits. The pointer head binds each option to the hidden
// state at the end of its numbered line, so the text must match byte for byte what the
// model was trained on. The state is tokenized with special tokens (BOS), each question
// without, and the ids are concatenated.

// strandsConfigFilename holds a Strands model's readout settings (DecisionReadoutConfig).
const strandsConfigFilename = "strands_decider_config.json"

var strandsHeaders = map[DecisionQuestionType]string{
	DecisionNoul:   "Decide whether the statement is true of the state.",
	DecisionChoice: "Select exactly one option.",
	DecisionScore:  "Rate the state against the ordered levels below (lowest first).",
}

var strandsNoulDefaults = [2]DecisionCriterion{
	{Name: "false", Description: "the statement does not hold for this state"},
	{Name: "true", Description: "the statement holds for this state"},
}

// renderedQuestion is a question's prompt text, the byte span of each option line
// within it, and the label bound to each slot.
type renderedQuestion struct {
	text   string
	spans  [][2]int
	labels []string
	kind   DecisionQuestionType
}

func renderStrandsState(state string) string {
	return "<state>\n" + strings.TrimSpace(state) + "\n</state>\n"
}

func renderStrandsQuestion(q DecisionQuestion) (renderedQuestion, error) {
	if err := q.validate(); err != nil {
		return renderedQuestion{}, err
	}
	var pairs []DecisionCriterion
	switch q.Type {
	case DecisionNoul:
		pairs = append([]DecisionCriterion(nil), strandsNoulDefaults[:]...)
		for _, c := range q.Criteria {
			for i := range pairs {
				if pairs[i].Name == c.Name {
					pairs[i].Description = strings.TrimSpace(c.Description)
				}
			}
		}
	case DecisionChoice:
		for _, c := range q.Criteria {
			pairs = append(pairs, DecisionCriterion{Name: c.Name, Description: strings.TrimSpace(c.Description)})
		}
	case DecisionScore:
		for i, c := range q.Criteria {
			pairs = append(pairs, DecisionCriterion{Name: strconv.Itoa(i), Description: c.Description})
		}
	}

	prefix := "<question type=\"" + string(q.Type) + "\">\n" + strandsHeaders[q.Type] + "\n" +
		strings.TrimSpace(q.Instructions) + "\n<options>\n"
	var block strings.Builder
	spans := make([][2]int, len(pairs))
	labels := make([]string, len(pairs))
	for i, c := range pairs {
		if i > 0 {
			block.WriteByte('\n')
		}
		line := strconv.Itoa(i+1) + ". " + c.Name
		if desc := collapseWhitespace(c.Description); desc != "" {
			line += " — " + desc
		}
		start := len(prefix) + block.Len()
		block.WriteString(line)
		spans[i] = [2]int{start, start + len(line)}
		labels[i] = c.Name
	}
	text := prefix + block.String() + "\n</options>\n</question>\n<answer>"
	return renderedQuestion{text: text, spans: spans, labels: labels, kind: q.Type}, nil
}

func collapseWhitespace(s string) string {
	return strings.Join(strings.Fields(s), " ")
}

// strandsOptionTokenIndex returns, for each option span (byte offsets into the question text),
// the index of the last token lying wholly inside it: the token that has just read the
// whole option under causal attention.
func strandsOptionTokenIndex(offsets [][2]int, spans [][2]int) ([]int, error) {
	out := make([]int, len(spans))
	for k, span := range spans {
		last := -1
		for j, off := range offsets {
			if off[1] <= off[0] {
				continue
			}
			if off[0] >= span[0] && off[1] <= span[1] {
				last = j
			}
		}
		if last < 0 {
			return nil, fmt.Errorf("option %d has no token of its own", k+1)
		}
		out[k] = last
	}
	return out, nil
}
