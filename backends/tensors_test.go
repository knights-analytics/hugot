package backends

import (
	"context"
	"errors"
	"testing"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/stretchr/testify/require"
)

func TestValidateNamedTensorsAcceptsStrings(t *testing.T) {
	metadata := []InputOutputInfo{{Name: "text", Dimensions: NewShape(-1, 2)}}

	require.NoError(t, validateNamedTensors(metadata, map[string]Tensor{
		"text": {Shape: []int64{2, 2}, Data: []string{"a", "b", "", "ünïcode"}},
	}))
	require.EqualError(t, validateNamedTensors(metadata, map[string]Tensor{
		"text": {Shape: []int64{2, 2}, Data: []string{"a", "b", "c"}},
	}), `input "text" has 3 elements, expected 4 for shape [2 2]`)
}

func TestValidateNamedTensorsChecksDataType(t *testing.T) {
	metadata := []InputOutputInfo{{Name: "ids", Dimensions: NewShape(-1), DataType: DataTypeInt64}}

	require.NoError(t, validateNamedTensors(metadata, map[string]Tensor{
		"ids": {Shape: []int64{2}, Data: []int64{1, 2}},
	}))
	require.EqualError(t, validateNamedTensors(metadata, map[string]Tensor{
		"ids": {Shape: []int64{2}, Data: []string{"1", "2"}},
	}), `input "ids" expects int64 data, got string`)

	// An unknown expected type leaves the check to the backend.
	metadata[0].DataType = DataTypeUnknown
	require.NoError(t, validateNamedTensors(metadata, map[string]Tensor{
		"ids": {Shape: []int64{2}, Data: []string{"1", "2"}},
	}))
}

func TestDataTypeFromGoMLX(t *testing.T) {
	require.Equal(t, DataTypeFloat32, dataTypeFromGoMLX(shapes.Make(dtypes.Float32, 2, 3)))
	require.Equal(t, DataTypeInt64, dataTypeFromGoMLX(shapes.Make(dtypes.Int64, 2)))
	require.Equal(t, DataTypeBool, dataTypeFromGoMLX(shapes.Make(dtypes.Bool)))
	require.Equal(t, DataTypeUnknown, dataTypeFromGoMLX(shapes.Make(dtypes.Complex64, 1)))
}

func TestRunNamedTensorsGoMLXRejectsStrings(t *testing.T) {
	model := &Model{InputsMeta: []InputOutputInfo{{Name: "text", Dimensions: NewShape(-1)}}}

	_, err := runNamedTensorsGoMLX(context.Background(), model, map[string]Tensor{
		"text": {Shape: []int64{1}, Data: []string{"a"}},
	})

	var unavailable *backendUnavailableError
	require.True(t, errors.As(err, &unavailable), "expected backendUnavailableError, got %v", err)
	require.EqualError(t, err, `GoMLX backend does not support string tensors (input "text")`)
}
