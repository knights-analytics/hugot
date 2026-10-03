package pipelines

import (
	"context"
	"image"
	"image/color"
	"math"
	"testing"

	"github.com/knights-analytics/hugot/backends"
	"github.com/stretchr/testify/require"
)

func TestNativeQARejectsGenericGenerativeModel(t *testing.T) {
	model := &backends.Model{IsGenerative: true}
	_, err := NewDocumentQuestionAnsweringPipeline(context.Background(), DocumentQuestionAnsweringConfig{}, model)
	require.Error(t, err)
	_, err = NewTableQuestionAnsweringPipeline(context.Background(), TableQuestionAnsweringConfig{}, model)
	require.Error(t, err)
}

func TestDonutAnswerAndDecoderContract(t *testing.T) {
	answer, err := donutAnswer(" us-001</s_answer>")
	require.NoError(t, err)
	require.Equal(t, "us-001", answer)
	_, err = donutAnswer("<s_question>cat</s_question>")
	require.Error(t, err)
	_, err = donutAnswer("")
	require.Error(t, err)
	token, err := lastTokenArgmax(backends.Tensor{Shape: []int64{1, 2, 3}, Data: []float32{100, 1, 1, 0, 2, 9}})
	require.NoError(t, err)
	require.Equal(t, int64(2), token)
	_, err = lastTokenArgmax(backends.Tensor{Shape: []int64{1, 1, 2}, Data: []float32{0, float32(math.NaN())}})
	require.Error(t, err)
}

func TestDonutProcessorPreservesLayout(t *testing.T) {
	p := donutProcessor{Mean: []float32{0.5, 0.5, 0.5}, Std: []float32{0.5, 0.5, 0.5}}
	p.Size.Width, p.Size.Height = 8, 8
	img := image.NewRGBA(image.Rect(10, 20, 18, 24))
	for y := 20; y < 24; y++ {
		for x := 10; x < 18; x++ {
			img.Set(x, y, color.White)
		}
	}
	pixels, err := p.pixels(img)
	require.NoError(t, err)
	require.Equal(t, []int64{1, 3, 8, 8}, pixels.Shape)
	data := pixels.Data.([]float32)
	require.Zero(t, data[0])
	require.Equal(t, float32(1), data[2*8])
	require.Zero(t, data[7*8])
}

func tapasTestPipeline() *TableQuestionAnsweringPipeline {
	return &TableQuestionAnsweringPipeline{MaxLength: 512, CellThreshold: 0.5, MaxRows: 64, MaxColumns: 32,
		vocabulary:        map[string]int64{"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "how": 4, "old": 5, "is": 6, "bob": 7, "?": 8, "name": 9, "age": 10, "alice": 11, "30": 12, "42": 13, "play": 14, "##ing": 15},
		aggregationLabels: map[int]string{0: "NONE", 1: "SUM", 2: "AVERAGE", 3: "COUNT"}}
}

func TestTAPASEncodingAndCellSelection(t *testing.T) {
	p := tapasTestPipeline()
	table := [][]string{{"Name", "Age"}, {"Alice", "30"}, {"Bob", "42"}}
	encoding, err := p.encodeTable(TableQuestionAnsweringInput{Table: table, Question: "How old is Bob?"})
	require.NoError(t, err)
	require.Equal(t, []int64{2, 4, 5, 6, 7, 8, 3, 9, 10, 11, 12, 7, 13}, encoding.IDs)
	require.Len(t, encoding.Types, len(encoding.IDs)*7)
	require.Equal(t, []int64{1, 2, 2, 0, 2, 1, 0}, encoding.Types[12*7:13*7])
	logits := make([]float32, len(encoding.IDs))
	for i := range logits {
		logits[i] = -10
	}
	logits[12] = 10
	result, err := p.decodeTable(table, encoding, map[string]backends.Tensor{"logits": {Shape: []int64{1, int64(len(logits))}, Data: logits}, "logits_aggregation": {Shape: []int64{1, 4}, Data: []float32{5, 0, 0, 0}}})
	require.NoError(t, err)
	require.Equal(t, "42", result.Answer)
	require.Equal(t, [][2]int{{1, 1}}, result.Coordinates)
	require.Equal(t, []string{"42"}, result.Cells)
	require.Equal(t, "NONE", result.Aggregator)
	require.Equal(t, []int64{14, 15}, p.wordPieces("Playing"))
	logits[12] = float32(math.NaN())
	_, err = p.decodeTable(table, encoding, map[string]backends.Tensor{"logits": {Shape: []int64{1, int64(len(logits))}, Data: logits}})
	require.Error(t, err)
}

func TestTAPASNumericRelationsAndInputErrors(t *testing.T) {
	p := tapasTestPipeline()
	table := [][]string{{"Age"}, {"30"}, {"42"}}
	encoding, err := p.encodeTable(TableQuestionAnsweringInput{Table: table, Query: "42"})
	require.NoError(t, err)
	require.Equal(t, int64(4), encoding.Types[4*7+6])
	require.Equal(t, int64(1), encoding.Types[5*7+6])
	_, err = p.encodeTable(TableQuestionAnsweringInput{Table: [][]string{{"Name", "Age"}, {"Bob"}}, Question: "age?"})
	require.Error(t, err)
	p.MaxLength = 3
	_, err = p.encodeTable(TableQuestionAnsweringInput{Table: table, Question: "How old is Bob?"})
	require.Error(t, err)
}
