//go:build cgo && (ORT || ALL)

package backends

import (
	"context"
	"testing"

	ort "github.com/microsoft/onnxruntime/go/onnxruntime"
	"github.com/stretchr/testify/require"
)

func TestConvertCoreORTIO(t *testing.T) {
	converted := convertCoreORTIO([]ort.IOInfo{{
		Name:     "input_ids",
		DataType: ort.TensorElementDataTypeInt64,
		Shape:    []int64{-1, 128},
	}})

	require.Equal(t, []InputOutputInfo{{Name: "input_ids", Dimensions: Shape{-1, 128}}}, converted)
}

func TestCoreORTSessionRejectsUninitializedSession(t *testing.T) {
	_, err := (&coreORTSession{}).Run(context.Background(), nil)
	require.EqualError(t, err, "ORT session is not initialized")
}

func TestCoreORTTensorCloseIsNilSafe(t *testing.T) {
	require.NoError(t, (*coreORTTensor)(nil).Close())
}
