package pipelines

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/knights-analytics/hugot/backends"
	"github.com/knights-analytics/hugot/util/fileutil"
	"github.com/stretchr/testify/require"
)

func TestTAPASStrictOutputShapes(t *testing.T) {
	p := tapasTestPipeline()
	table := [][]string{{"Age"}, {"42"}}
	encoding, err := p.encodeTable(TableQuestionAnsweringInput{Table: table, Question: "age?"})
	require.NoError(t, err)
	n := int64(len(encoding.IDs))
	for _, shape := range [][]int64{nil, {n}, {n, 1}, {2, n}, {1, n, 1}} {
		_, err = p.decodeTable(table, encoding, map[string]backends.Tensor{"logits": {Shape: shape, Data: make([]float32, n)}})
		require.Error(t, err, "shape %v", shape)
	}
	for _, tensor := range []backends.Tensor{
		{Shape: []int64{1, n}, Data: make([]float32, n-1)},
		{Shape: []int64{1, n}, Data: make([]int64, n)},
		{Shape: []int64{1, n + 1}, Data: make([]float32, n)},
	} {
		_, err = p.decodeTable(table, encoding, map[string]backends.Tensor{"logits": tensor})
		require.Error(t, err)
	}
}

func TestTAPASNumericConsolidation(t *testing.T) {
	p := tapasTestPipeline()
	encoding, err := p.encodeTable(TableQuestionAnsweringInput{Table: [][]string{{"Age"}, {"42"}, {"unknown"}, {"unknown"}}, Question: "42"})
	require.NoError(t, err)
	for i, cell := range encoding.Cells {
		if cell[0] >= 0 {
			require.Equal(t, []int64{0, 0, 0}, encoding.Types[i*7+4:i*7+7])
		}
	}
}

func TestTAPASSequentialStateAndIsolation(t *testing.T) {
	p := tapasTestPipeline()
	table := [][]string{{"Name"}, {"Playing Bob"}, {"Alice"}}
	inputs := []TableQuestionAnsweringInput{
		{Table: table, Question: "Bob?"}, {Table: table, Question: "Alice?"},
		{Table: table, Question: "Name?"}, {Table: table, Question: "Name?"},
	}
	for _, sequential := range []bool{true, false} {
		for run := 0; run < 2; run++ {
			t.Run(fmt.Sprintf("sequential=%t/run=%d", sequential, run), func(t *testing.T) {
				t.Parallel()
				call := 0
				infer := func(ctx context.Context, tensors map[string]backends.Tensor) (map[string]backends.Tensor, error) {
					require.NoError(t, ctx.Err())
					n := tensors["input_ids"].Shape[1]
					require.Equal(t, []int64{1, n}, tensors["attention_mask"].Shape)
					require.Equal(t, []int64{1, n, 7}, tensors["token_type_ids"].Shape)
					types := tensors["token_type_ids"].Data.([]int64)
					logits := make([]float32, n)
					for i := range logits {
						features := types[i*7 : (i+1)*7]
						want := int64(0)
						if sequential && ((call == 1 && features[2] == 1) || (call == 2 && features[2] == 2)) {
							want = 1
						}
						require.Equal(t, want, features[3], "call %d token %d", call, i)
						logits[i] = -10
						if (call == 0 && features[2] == 1) || (call == 1 && features[2] == 2) {
							logits[i] = 10
						}
					}
					call++
					return map[string]backends.Tensor{"logits": {Shape: []int64{1, n}, Data: logits}}, nil
				}
				out, err := p.runTableQuestions(context.Background(), inputs, sequential, infer)
				require.NoError(t, err)
				require.Equal(t, 4, call)
				require.Equal(t, []string{"Playing Bob", "Alice", "", ""}, []string{out.Results[0].Answer, out.Results[1].Answer, out.Results[2].Answer, out.Results[3].Answer})
				require.Equal(t, [][2]int{{0, 0}}, out.Results[0].Coordinates)
				require.Equal(t, [][2]int{{1, 0}}, out.Results[1].Coordinates)
			})
		}
	}
}

func TestTAPASRunnerInvalidAndCancellation(t *testing.T) {
	p := tapasTestPipeline()
	valid := TableQuestionAnsweringInput{Table: [][]string{{"Age"}, {"42"}}, Question: "age?"}
	for _, inputs := range [][]TableQuestionAnsweringInput{
		nil, {{}}, {valid, {Table: valid.Table, Question: " "}},
		{{Table: [][]string{{"Age"}}, Question: "age?"}},
		{{Table: [][]string{{"Name", "Age"}, {"Bob"}}, Question: "age?"}},
		{{Table: valid.Table, Question: "age?", Role: "assistant"}},
	} {
		out, err := p.runTableQuestions(context.Background(), inputs, true, func(context.Context, map[string]backends.Tensor) (map[string]backends.Tensor, error) {
			t.Fatal("invalid inputs must not reach inference")
			return nil, nil
		})
		require.Error(t, err)
		require.Nil(t, out)
	}
	_, err := p.runTableQuestions(context.Background(), []TableQuestionAnsweringInput{valid}, true, nil)
	require.Error(t, err)
	for _, cancelBefore := range []bool{true, false} {
		ctx, cancel := context.WithCancel(context.Background())
		if cancelBefore {
			cancel()
		}
		calls := 0
		out, err := p.runTableQuestions(ctx, []TableQuestionAnsweringInput{valid, valid}, true, func(context.Context, map[string]backends.Tensor) (map[string]backends.Tensor, error) {
			calls++
			cancel()
			return nil, nil
		})
		cancel()
		require.ErrorIs(t, err, context.Canceled)
		require.Nil(t, out)
		if cancelBefore {
			require.Zero(t, calls)
		} else {
			require.Equal(t, 1, calls)
		}
	}
	failure := errors.New("inference failed")
	out, err := p.runTableQuestions(context.Background(), []TableQuestionAnsweringInput{valid, valid}, true, func(context.Context, map[string]backends.Tensor) (map[string]backends.Tensor, error) {
		return nil, failure
	})
	require.ErrorIs(t, err, failure)
	require.Nil(t, out)
}

func TestTAPASAggregation(t *testing.T) {
	p := tapasTestPipeline()
	table := [][]string{{"Age"}, {"30"}, {"42"}}
	encoding, err := p.encodeTable(TableQuestionAnsweringInput{Table: table, Question: "age?"})
	require.NoError(t, err)
	logits := make([]float32, len(encoding.IDs))
	for i := range logits {
		logits[i] = 10
	}
	outputs := map[string]backends.Tensor{"logits": {Shape: []int64{1, int64(len(logits))}, Data: logits}}
	for i, label := range []string{"NONE", "SUM", "AVERAGE", "COUNT"} {
		values := make([]float32, 4)
		values[i] = 5
		outputs["logits_aggregation"] = backends.Tensor{Shape: []int64{1, 4}, Data: values}
		result, err := p.decodeTable(table, encoding, outputs)
		require.NoError(t, err)
		require.Equal(t, label, result.Aggregator)
		want := "30, 42"
		if label != "NONE" {
			want = label + " > " + want
		}
		require.Equal(t, want, result.Answer)
	}
	for _, tensor := range []backends.Tensor{
		{Data: []float32{0, 1, 0, 0}},
		{Shape: []int64{4}, Data: []float32{0, 1, 0, 0}},
		{Shape: []int64{2, 2}, Data: []float32{0, 1, 0, 0}},
		{Shape: []int64{1, 3}, Data: []float32{0, 1, 0}},
		{Shape: []int64{1, 4}, Data: []float32{}},
		{Shape: []int64{1, 4}, Data: []int64{0, 1, 0, 0}},
		{Shape: []int64{1, 4}, Data: []float32{0, float32(math.NaN()), 0, 0}},
		{Shape: []int64{1, 4}, Data: []float32{0, 1, float32(math.Inf(1)), 0}},
	} {
		outputs["logits_aggregation"] = tensor
		_, err = p.decodeTable(table, encoding, outputs)
		require.Error(t, err)
	}
	outputs["logits_aggregation"] = backends.Tensor{Shape: []int64{1, 4}, Data: []float32{0, 1, 0, 0}}
	for _, labels := range []map[int]string{nil, {0: "NONE", 2: "SUM"}, {0: "NONE", 1: " "}, {0: "NONE", 1: "SUM", 2: "AVERAGE"}} {
		p.aggregationLabels = labels
		_, err = p.decodeTable(table, encoding, outputs)
		require.Error(t, err)
	}
}

func TestTAPASTableEmbeddingLimits(t *testing.T) {
	p := tapasTestPipeline()
	table := make([][]string, 64)
	for i := range table {
		table[i] = []string{"Age"}
	}
	_, err := p.encodeTable(TableQuestionAnsweringInput{Table: table, Question: "age?"})
	require.NoError(t, err)
	_, err = p.encodeTable(TableQuestionAnsweringInput{Table: append(table, []string{"42"}), Question: "age?"})
	require.Error(t, err)
	p.MaxRows, p.MaxColumns = 3, 3
	_, err = p.encodeTable(TableQuestionAnsweringInput{Table: [][]string{{"Name", "Age"}, {"Bob", "42"}, {"Alice", "30"}}, Question: "age?"})
	require.NoError(t, err)
	_, err = p.encodeTable(TableQuestionAnsweringInput{Table: [][]string{{"Age"}, {"42"}, {"30"}, {"30"}}, Question: "age?"})
	require.Error(t, err)
	_, err = p.encodeTable(TableQuestionAnsweringInput{Table: [][]string{{"Age", "Age", "Age"}, {"42", "30", "30"}}, Question: "age?"})
	require.Error(t, err)
}

func TestTAPASConfiguredLimitsAndLabels(t *testing.T) {
	for _, config := range []struct {
		json    string
		invalid bool
	}{
		{json: `{"max_num_rows":3,"max_num_columns":3,"aggregation_labels":{"0":"NONE","1":"SUM"}}`},
		{json: `{"max_num_rows":-1}`, invalid: true},
		{json: `{"max_num_columns":-1}`, invalid: true},
		{json: `{"aggregation_labels":{"0":"NONE","2":"SUM"}}`, invalid: true},
		{json: `{"aggregation_labels":{"0":"NONE","1":" "}}`, invalid: true},
	} {
		t.Run(config.json, func(t *testing.T) {
			path := t.TempDir()
			require.NoError(t, os.WriteFile(filepath.Join(path, "vocab.txt"), []byte("[PAD]\n[UNK]\n[CLS]\n[SEP]\n"), 0600))
			require.NoError(t, os.WriteFile(filepath.Join(path, "config.json"), []byte(config.json), 0600))
			model := &backends.Model{Path: path,
				InputsMeta: []backends.InputOutputInfo{
					{Name: "input_ids", Dimensions: []int64{-1, -1}},
					{Name: "attention_mask", Dimensions: []int64{-1, -1}},
					{Name: "token_type_ids", Dimensions: []int64{-1, -1, 7}},
				},
				OutputsMeta: []backends.InputOutputInfo{{Name: "logits", Dimensions: []int64{-1, -1}}},
			}
			p, err := NewTableQuestionAnsweringPipeline(fileutil.WithFileSystem(context.Background(), nil), TableQuestionAnsweringConfig{}, model)
			if config.invalid {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			require.Equal(t, 3, p.MaxRows)
			require.Equal(t, 3, p.MaxColumns)
			require.Equal(t, map[int]string{0: "NONE", 1: "SUM"}, p.aggregationLabels)
			table := [][]string{{"Age"}, {"42"}, {"30"}, {"30"}}
			_, err = p.encodeTable(TableQuestionAnsweringInput{Table: table, Question: "age?"})
			require.Error(t, err)
			_, err = p.RunSequential(context.Background(), table[:3], nil)
			require.Error(t, err)
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			_, err = p.RunSequential(ctx, table[:3], []string{"age?"})
			require.ErrorIs(t, err, context.Canceled)
			_, err = p.RunPipeline(ctx, []TableQuestionAnsweringInput{{Table: table[:3], Question: "age?"}})
			require.ErrorIs(t, err, context.Canceled)
		})
	}
}

func TestTAPASNumericDecimalRelationsAndTiedRanks(t *testing.T) {
	require.Equal(t, []float64{-0.5, 1250, 42}, tapasQuestionNumbers("-.5 1,250 42."))
	p := tapasTestPipeline()
	table := [][]string{{"Age"}, {".5"}, {"30"}, {"30"}, {"42"}}
	encoding, err := p.encodeTable(TableQuestionAnsweringInput{Table: table, Question: "30 42 .5"})
	require.NoError(t, err)
	want := [][3]int64{{1, 3, 5}, {2, 2, 7}, {2, 2, 7}, {3, 1, 3}}
	for i, cell := range encoding.Cells {
		if cell[0] >= 0 {
			require.Equal(t, want[cell[0]][:], encoding.Types[i*7+4:i*7+7])
		}
	}
}
