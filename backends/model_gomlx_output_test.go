package backends

import (
	"testing"

	"github.com/gomlx/gomlx/core/tensors"
	"github.com/stretchr/testify/require"
)

func TestReshapeGoMLXOutputPreservesIntegerLabels(t *testing.T) {
	tensor := tensors.FromValue([]int64{9007199254740993, 42, -1, -1})
	defer tensor.FinalizeAll()
	batch := &PipelineBatch{Size: 2, PaddedBatchSize: 4}
	output, err := reshapeGoMLXOutput[int64](tensor, InputOutputInfo{Dimensions: Shape{-1}}, batch)
	require.NoError(t, err)
	require.Equal(t, []int64{9007199254740993, 42}, output)
	// Output must own its data after the runtime tensor is released.
	require.NoError(t, tensor.FinalizeAll())
	require.Equal(t, []int64{9007199254740993, 42}, output)
}

func TestReshapeGoMLXOutputTrimsPaddedRows(t *testing.T) {
	tensor := tensors.FromValue([][]float32{{1, 2, 3}, {4, 5, 6}, {0, 0, 0}, {0, 0, 0}})
	defer tensor.FinalizeAll()
	output, err := reshapeGoMLXOutput[float32](tensor, InputOutputInfo{Dimensions: Shape{-1, -1}}, &PipelineBatch{Size: 2, PaddedBatchSize: 4})
	require.NoError(t, err)
	require.Equal(t, [][]float32{{1, 2, 3}, {4, 5, 6}}, output)
}
