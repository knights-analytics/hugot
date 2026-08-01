//go:build cgo && (ORT || ALL)

package backends

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	ort "github.com/yalue/onnxruntime_go"
)

func initORTLibrary(t *testing.T) {
	t.Helper()
	ort.SetSharedLibraryPath("/usr/local/lib/libonnxruntime.dylib")
	err := ort.InitializeEnvironment()
	if err == nil {
		return
	}
	if strings.Contains(strings.ToLower(err.Error()), "already been initialized") {
		return
	}
	require.NoError(t, err)
}

func TestConvertRankTwoInt8TensorCopiesSignedBytes(t *testing.T) {
	initORTLibrary(t)

	data := []int8{-128, -127, 0, 1, 127, 42}
	tensor, err := ort.NewTensor(ort.NewShape(2, 3), data)
	require.NoError(t, err)

	got, err := convertRankTwoInt8Tensor(tensor, "pooler_output_int8", 2)
	require.NoError(t, err)
	require.Equal(t, [][]int8{{-128, -127, 0}, {1, 127, 42}}, got)

	require.NoError(t, tensor.Destroy())
	assert.Equal(t, [][]int8{{-128, -127, 0}, {1, 127, 42}}, got)
}

func TestConvertRankTwoInt8TensorValidation(t *testing.T) {
	initORTLibrary(t)

	rank1, err := ort.NewTensor(ort.NewShape(4), []int8{1, 2, 3, 4})
	require.NoError(t, err)
	defer func() { require.NoError(t, rank1.Destroy()) }()
	_, err = convertRankTwoInt8Tensor(rank1, "out", 4)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "rank")

	rank3, err := ort.NewTensor(ort.NewShape(1, 2, 2), []int8{1, 2, 3, 4})
	require.NoError(t, err)
	defer func() { require.NoError(t, rank3.Destroy()) }()
	_, err = convertRankTwoInt8Tensor(rank3, "out", 1)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "rank")

	batchMismatch, err := ort.NewTensor(ort.NewShape(2, 2), []int8{1, 2, 3, 4})
	require.NoError(t, err)
	defer func() { require.NoError(t, batchMismatch.Destroy()) }()
	_, err = convertRankTwoInt8Tensor(batchMismatch, "out", 3)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "batch size")

	_, err = convertRankTwoInt8Tensor(nil, "out", 1)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "nil")
}

func TestDestroyORTOutputsClearsSlots(t *testing.T) {
	initORTLibrary(t)

	a, err := ort.NewTensor(ort.NewShape(1, 2), []int8{9, 8})
	require.NoError(t, err)
	b, err := ort.NewTensor(ort.NewShape(1, 2), []float32{1, 2})
	require.NoError(t, err)
	slots := []ort.Value{a, nil, b}
	require.NoError(t, destroyORTOutputs(slots))
	assert.Nil(t, slots[0])
	assert.Nil(t, slots[1])
	assert.Nil(t, slots[2])
	require.NoError(t, destroyORTOutputs(slots))
}
