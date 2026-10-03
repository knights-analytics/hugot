package pipelines

import (
	"context"
	"errors"
	"fmt"
	"math"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"

	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/util/fileutil"
)

// TableQuestionAnsweringPipeline runs TAPAS cell selection and aggregation.
type TableQuestionAnsweringPipeline struct {
	*backends.BasePipeline
	MaxLength     int
	CellThreshold float32
	// Embedding limits include the reserved zero row/column ID.
	MaxRows           int
	MaxColumns        int
	vocabulary        map[string]int64
	aggregationLabels map[int]string
}

type TableQuestionAnsweringResult struct {
	Answer      string
	Coordinates [][2]int
	Cells       []string
	Aggregator  string
}
type TableQuestionAnsweringOutput struct {
	Results []TableQuestionAnsweringResult
}

func (o *TableQuestionAnsweringOutput) GetOutput() []any {
	out := make([]any, len(o.Results))
	for i, result := range o.Results {
		out[i] = result
	}
	return out
}

func NewTableQuestionAnsweringPipeline(ctx context.Context, config TableQuestionAnsweringConfig, model *backends.Model) (*TableQuestionAnsweringPipeline, error) {
	if model == nil {
		return nil, errors.New("table QA requires a TAPAS cell-selection model")
	}
	p := &TableQuestionAnsweringPipeline{BasePipeline: backends.NewBasePipeline(ctx, config, model), MaxLength: 512, CellThreshold: 0.5, MaxRows: 64, MaxColumns: 32, aggregationLabels: map[int]string{0: "NONE", 1: "SUM", 2: "AVERAGE", 3: "COUNT"}}
	for _, option := range config.Options {
		if err := option(p); err != nil {
			return nil, err
		}
	}
	if err := p.Validate(); err != nil {
		return nil, err
	}
	data, err := fileutil.ReadFileBytes(ctx, fileutil.PathJoinSafe(model.Path, "vocab.txt"))
	if err != nil {
		return nil, err
	}
	p.vocabulary = make(map[string]int64)
	for i, word := range strings.Split(strings.TrimRight(string(data), "\r\n"), "\n") {
		p.vocabulary[strings.TrimSuffix(word, "\r")] = int64(i)
	}
	for _, special := range []string{"[CLS]", "[SEP]", "[PAD]", "[UNK]"} {
		if _, ok := p.vocabulary[special]; !ok {
			return nil, fmt.Errorf("TAPAS vocabulary lacks %s", special)
		}
	}
	var cfg struct {
		Labels     map[int]string `json:"aggregation_labels"`
		MaxRows    int            `json:"max_num_rows"`
		MaxColumns int            `json:"max_num_columns"`
	}
	if err := readNativeConfig(ctx, model.Path, "config.json", &cfg); err != nil {
		return nil, err
	}
	if len(cfg.Labels) > 0 {
		p.aggregationLabels = cfg.Labels
	}
	if cfg.MaxRows != 0 {
		p.MaxRows = cfg.MaxRows
	}
	if cfg.MaxColumns != 0 {
		p.MaxColumns = cfg.MaxColumns
	}
	if err := p.Validate(); err != nil {
		return nil, err
	}
	return p, nil
}
func (*TableQuestionAnsweringPipeline) IsGenerative() bool          { return false }
func (p *TableQuestionAnsweringPipeline) GetModel() *backends.Model { return p.Model }
func (p *TableQuestionAnsweringPipeline) GetMetadata() backends.PipelineMetadata {
	return nativeMetadata(p.Model)
}
func (p *TableQuestionAnsweringPipeline) GetStatistics() backends.PipelineStatistics {
	return nativeStatistics(p.BasePipeline)
}
func (p *TableQuestionAnsweringPipeline) Validate() error {
	if p == nil || p.BasePipeline == nil || p.Model == nil {
		return errors.New("table QA requires a TAPAS model")
	}
	if err := requireTensorNames(p.Model, []string{"input_ids", "attention_mask", "token_type_ids"}, []string{"logits"}); err != nil {
		return err
	}
	for _, info := range p.Model.InputsMeta {
		if info.Name == "token_type_ids" && (len(info.Dimensions) != 3 || info.Dimensions[2] != 7) {
			return errors.New("TAPAS token_type_ids must have seven table feature channels")
		}
	}
	if p.MaxLength <= 0 || p.MaxLength > 512 || math.IsNaN(float64(p.CellThreshold)) || p.CellThreshold < 0 || p.CellThreshold > 1 {
		return errors.New("invalid TAPAS sequence length or cell threshold")
	}
	if p.MaxRows <= 0 || p.MaxColumns <= 0 {
		return errors.New("invalid TAPAS row or column embedding limits")
	}
	return p.validateAggregationLabels()
}

func (p *TableQuestionAnsweringPipeline) validateAggregationLabels() error {
	if len(p.aggregationLabels) == 0 {
		return errors.New("TAPAS aggregation labels are missing")
	}
	for i := 0; i < len(p.aggregationLabels); i++ {
		if strings.TrimSpace(p.aggregationLabels[i]) == "" {
			return errors.New("TAPAS aggregation labels must be nonempty and indexed contiguously from zero")
		}
	}
	return nil
}

type tapasEncoding struct {
	IDs, Attention, Types []int64
	Cells                 [][2]int
}

func (p *TableQuestionAnsweringPipeline) RunPipeline(ctx context.Context, inputs []TableQuestionAnsweringInput) (*TableQuestionAnsweringOutput, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	return p.runTableQuestions(ctx, inputs, false, p.inferTable)
}

// RunSequential answers questions in order, using only the preceding answer's
// selected cells as TAPAS prev_labels. State is local to this call.
func (p *TableQuestionAnsweringPipeline) RunSequential(ctx context.Context, table [][]string, questions []string) (*TableQuestionAnsweringOutput, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	inputs := make([]TableQuestionAnsweringInput, len(questions))
	for i, question := range questions {
		inputs[i] = TableQuestionAnsweringInput{Table: table, Question: question}
	}
	return p.runTableQuestions(ctx, inputs, true, p.inferTable)
}

type tapasInference func(context.Context, map[string]backends.Tensor) (map[string]backends.Tensor, error)

func (p *TableQuestionAnsweringPipeline) inferTable(ctx context.Context, inputs map[string]backends.Tensor) (map[string]backends.Tensor, error) {
	return runNativeTensors(ctx, p.BasePipeline, p.Model, inputs)
}

func (p *TableQuestionAnsweringPipeline) runTableQuestions(ctx context.Context, inputs []TableQuestionAnsweringInput, sequential bool, infer tapasInference) (*TableQuestionAnsweringOutput, error) {
	if len(inputs) == 0 {
		return nil, errors.New("table QA requires at least one table question")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(p.vocabulary) == 0 {
		return nil, errors.New("TAPAS vocabulary is not loaded")
	}
	if infer == nil {
		return nil, errors.New("TAPAS inference is not configured")
	}
	encodings := make([]tapasEncoding, len(inputs))
	for i, input := range inputs {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if len(input.History) > 0 || (input.Role != "" && input.Role != "user") {
			return nil, errors.New("native table QA does not accept conversation history or roles")
		}
		encoding, err := p.encodeTable(input)
		if err != nil {
			return nil, err
		}
		encodings[i] = encoding
	}
	out := &TableQuestionAnsweringOutput{Results: make([]TableQuestionAnsweringResult, len(inputs))}
	previous := map[[2]int]bool{}
	for i, input := range inputs {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		encoding := encodings[i]
		if sequential {
			for j, cell := range encoding.Cells {
				if previous[cell] {
					encoding.Types[j*7+3] = 1
				}
			}
		}
		n := int64(len(encoding.IDs))
		decoded, err := infer(ctx, map[string]backends.Tensor{
			"input_ids": {Shape: []int64{1, n}, Data: encoding.IDs}, "attention_mask": {Shape: []int64{1, n}, Data: encoding.Attention}, "token_type_ids": {Shape: []int64{1, n, 7}, Data: encoding.Types},
		})
		if err != nil {
			return nil, err
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		result, err := p.decodeTable(input.Table, encoding, decoded)
		if err != nil {
			return nil, err
		}
		out.Results[i] = result
		if sequential {
			previous = make(map[[2]int]bool, len(result.Coordinates))
			for _, cell := range result.Coordinates {
				previous[cell] = true
			}
		}
	}
	return out, nil
}

func (p *TableQuestionAnsweringPipeline) encodeTable(input TableQuestionAnsweringInput) (tapasEncoding, error) {
	question := input.Question
	if question == "" {
		question = input.Query
	}
	if strings.TrimSpace(question) == "" || len(input.Table) < 2 || len(input.Table[0]) == 0 {
		return tapasEncoding{}, errors.New("TAPAS requires a question, header, and data rows")
	}
	columns := len(input.Table[0])
	if columns >= p.MaxColumns || len(input.Table)-1 >= p.MaxRows {
		return tapasEncoding{}, errors.New("TAPAS table exceeds row or column embeddings")
	}
	for _, row := range input.Table {
		if len(row) != columns {
			return tapasEncoding{}, errors.New("TAPAS table rows must have the same number of columns")
		}
	}
	encoding := tapasEncoding{}
	appendToken := func(id int64, features [7]int64, cell [2]int) {
		encoding.IDs = append(encoding.IDs, id)
		encoding.Attention = append(encoding.Attention, 1)
		encoding.Types = append(encoding.Types, features[:]...)
		encoding.Cells = append(encoding.Cells, cell)
	}
	none := [2]int{-1, -1}
	appendToken(p.vocabulary["[CLS]"], [7]int64{}, none)
	for _, id := range p.wordPieces(question) {
		appendToken(id, [7]int64{}, none)
	}
	appendToken(p.vocabulary["[SEP]"], [7]int64{}, none)
	questionNumbers := tapasQuestionNumbers(question)
	numericColumns := make([]bool, columns)
	for c := range numericColumns {
		count := 0
		for _, row := range input.Table[1:] {
			if _, ok := tapasNumber(row[c]); ok {
				count++
			}
		}
		numericColumns[c] = float64(count) >= 0.7*float64(len(input.Table)-1)
	}
	for r, row := range input.Table {
		for c, text := range row {
			features := [7]int64{1, int64(c + 1), int64(r), 0, 0, 0, 0}
			cell := none
			if r > 0 {
				cell = [2]int{r - 1, c}
				if value, ok := tapasNumber(text); ok && numericColumns[c] {
					rank, inv := tapasColumnRank(input.Table, c, value)
					features[4], features[5] = int64(rank), int64(inv)
					for _, queryNumber := range questionNumbers {
						if queryNumber == value {
							features[6] |= 1
						}
						if queryNumber < value {
							features[6] |= 2
						}
						if queryNumber > value {
							features[6] |= 4
						}
					}
				}
			}
			for _, id := range p.wordPieces(text) {
				appendToken(id, features, cell)
			}
		}
	}
	if len(encoding.IDs) > p.MaxLength {
		return tapasEncoding{}, fmt.Errorf("TAPAS table requires %d tokens, exceeding max length %d; refusing to drop answer cells", len(encoding.IDs), p.MaxLength)
	}
	return encoding, nil
}

func (p *TableQuestionAnsweringPipeline) wordPieces(text string) []int64 {
	// TAPAS uses uncased BERT basic tokenization followed by greedy WordPiece.
	var cleaned strings.Builder
	for _, r := range norm.NFD.String(strings.ToLower(text)) {
		if unicode.Is(unicode.Mn, r) || r == 0 || r == unicode.ReplacementChar || (unicode.IsControl(r) && !unicode.IsSpace(r)) {
			continue
		}
		if unicode.IsSpace(r) {
			cleaned.WriteByte(' ')
			continue
		}
		if unicode.IsPunct(r) || (r >= 33 && r <= 47) || (r >= 58 && r <= 64) || (r >= 91 && r <= 96) || (r >= 123 && r <= 126) || (r >= 0x4e00 && r <= 0x9fff) {
			cleaned.WriteByte(' ')
			cleaned.WriteRune(r)
			cleaned.WriteByte(' ')
			continue
		}
		cleaned.WriteRune(r)
	}
	var ids []int64
	for _, word := range strings.Fields(cleaned.String()) {
		runes := []rune(word)
		var pieces []int64
		failed := len(runes) > 100
		for start := 0; start < len(runes) && !failed; {
			found := false
			for end := len(runes); end > start; end-- {
				piece := string(runes[start:end])
				if start > 0 {
					piece = "##" + piece
				}
				if id, ok := p.vocabulary[piece]; ok {
					pieces = append(pieces, id)
					start = end
					found = true
					break
				}
			}
			if !found {
				failed = true
			}
		}
		if failed {
			ids = append(ids, p.vocabulary["[UNK]"])
		} else {
			ids = append(ids, pieces...)
		}
	}
	return ids
}

var tapasNumericPattern = regexp.MustCompile(`[-+]?(?:\d+(?:,\d{3})*(?:\.\d*)?|\.\d+)`)

func tapasNumber(text string) (float64, bool) {
	text = strings.TrimSpace(text)
	text = strings.TrimPrefix(text, "$")
	text = strings.ReplaceAll(text, ",", "")
	value, err := strconv.ParseFloat(text, 64)
	return value, err == nil && !math.IsInf(value, 0) && !math.IsNaN(value)
}
func tapasQuestionNumbers(question string) []float64 {
	var out []float64
	for _, text := range tapasNumericPattern.FindAllString(question, -1) {
		if value, ok := tapasNumber(text); ok {
			out = append(out, value)
		}
	}
	return out
}
func tapasColumnRank(table [][]string, column int, value float64) (int, int) {
	var numbers []float64
	for _, row := range table[1:] {
		if number, ok := tapasNumber(row[column]); ok {
			numbers = append(numbers, number)
		}
	}
	sort.Float64s(numbers)
	var unique []float64
	for _, number := range numbers {
		if len(unique) == 0 || unique[len(unique)-1] != number {
			unique = append(unique, number)
		}
	}
	index := sort.SearchFloat64s(unique, value)
	return index + 1, len(unique) - index
}

func (p *TableQuestionAnsweringPipeline) decodeTable(table [][]string, encoding tapasEncoding, outputs map[string]backends.Tensor) (TableQuestionAnsweringResult, error) {
	tensor := outputs["logits"]
	logits, ok := tensor.Data.([]float32)
	if !ok || len(logits) == 0 || len(logits) != len(encoding.IDs) || len(encoding.Cells) != len(logits) || len(tensor.Shape) != 2 || tensor.Shape[0] != 1 || tensor.Shape[1] != int64(len(logits)) {
		return TableQuestionAnsweringResult{}, errors.New("TAPAS logits must have shape [1, sequence_length] matching the table token sequence")
	}
	type cellScore struct {
		sum   float64
		count int
	}
	scores := map[[2]int]cellScore{}
	for i, cell := range encoding.Cells {
		if math.IsNaN(float64(logits[i])) || math.IsInf(float64(logits[i]), 0) {
			return TableQuestionAnsweringResult{}, errors.New("TAPAS logits are nonfinite")
		}
		if cell[0] < 0 {
			continue
		}
		score := scores[cell]
		score.sum += 1 / (1 + math.Exp(-float64(logits[i])))
		score.count++
		scores[cell] = score
	}
	result := TableQuestionAnsweringResult{Aggregator: "NONE"}
	for r, row := range table[1:] {
		for c, text := range row {
			cell := [2]int{r, c}
			score := scores[cell]
			if score.count > 0 && score.sum/float64(score.count) > float64(p.CellThreshold) {
				result.Coordinates = append(result.Coordinates, cell)
				result.Cells = append(result.Cells, text)
			}
		}
	}
	if aggregation, exists := outputs["logits_aggregation"]; exists {
		if err := p.validateAggregationLabels(); err != nil {
			return result, err
		}
		values, ok := aggregation.Data.([]float32)
		if !ok || len(values) != len(p.aggregationLabels) || len(aggregation.Shape) != 2 || aggregation.Shape[0] != 1 || aggregation.Shape[1] != int64(len(values)) {
			return result, errors.New("invalid TAPAS aggregation logits")
		}
		best := 0
		for i, value := range values {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return result, errors.New("nonfinite aggregation logits")
			}
			if value > values[best] {
				best = i
			}
		}
		label, ok := p.aggregationLabels[best]
		if !ok {
			return result, errors.New("unknown TAPAS aggregation label")
		}
		result.Aggregator = label
	}
	result.Answer = strings.Join(result.Cells, ", ")
	if result.Aggregator != "NONE" && result.Aggregator != "" {
		result.Answer = result.Aggregator + " > " + result.Answer
	}
	return result, nil
}
