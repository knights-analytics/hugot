package backends

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestModelIdentityEmptySelectionMatchesLegacy(t *testing.T) {
	t.Parallel()

	assert.Equal(t, "/models/foo:", ModelIdentity("/models/foo", "", nil))
	assert.Equal(t, "/models/foo:model.onnx", ModelIdentity("/models/foo", "model.onnx", nil))
	assert.Equal(t, "/models/foo:model.onnx", ModelIdentity("/models/foo", "model.onnx", []string{}))
}

func TestModelIdentityPreservesSelectionOrder(t *testing.T) {
	t.Parallel()

	sameOrderA := ModelIdentity("/models/foo", "model.onnx", []string{"pooler_output_int8", "last_hidden_state"})
	sameOrderB := ModelIdentity("/models/foo", "model.onnx", []string{"pooler_output_int8", "last_hidden_state"})
	reversed := ModelIdentity("/models/foo", "model.onnx", []string{"last_hidden_state", "pooler_output_int8"})
	single := ModelIdentity("/models/foo", "model.onnx", []string{"pooler_output_int8"})

	assert.Equal(t, sameOrderA, sameOrderB)
	assert.NotEqual(t, sameOrderA, reversed)
	assert.NotEqual(t, sameOrderA, single)
	assert.Equal(t, "/models/foo:model.onnx#o2/18:pooler_output_int8/17:last_hidden_state", sameOrderA)
}

func TestModelIdentityRejectsDelimiterCollisions(t *testing.T) {
	t.Parallel()

	commaJoined := ModelIdentity("/models/foo", "model.onnx", []string{"a,b"})
	splitNames := ModelIdentity("/models/foo", "model.onnx", []string{"a", "b"})
	assert.NotEqual(t, commaJoined, splitNames)
	assert.Equal(t, "/models/foo:model.onnx#o1/3:a,b", commaJoined)
	assert.Equal(t, "/models/foo:model.onnx#o2/1:a/1:b", splitNames)

	colonName := ModelIdentity("/models/foo", "model.onnx", []string{"a:b"})
	twoParts := ModelIdentity("/models/foo", "model.onnx", []string{"a", "b"})
	assert.NotEqual(t, colonName, twoParts)
}

func TestCopyOnnxOutputNamesIsDefensive(t *testing.T) {
	t.Parallel()

	assert.Nil(t, CopyOnnxOutputNames(nil))

	original := []string{"last_hidden_state"}
	copied := CopyOnnxOutputNames(original)
	require.Equal(t, original, copied)

	original[0] = "mutated"
	assert.Equal(t, []string{"last_hidden_state"}, copied)
}

func TestValidateOnnxOutputNameList(t *testing.T) {
	t.Parallel()

	assert.NoError(t, ValidateOnnxOutputNameList(nil))
	assert.NoError(t, ValidateOnnxOutputNameList([]string{}))
	assert.NoError(t, ValidateOnnxOutputNameList([]string{"a", "b"}))

	err := ValidateOnnxOutputNameList([]string{"a", ""})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "blank")

	err = ValidateOnnxOutputNameList([]string{"a", "a"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "duplicate")
}

func TestSelectOnnxOutputsPreservesOrderAndListsAvailable(t *testing.T) {
	t.Parallel()

	available := []InputOutputInfo{
		{Name: "last_hidden_state"},
		{Name: "pooler_output"},
		{Name: "pooler_output_int8"},
	}

	allMeta, allNames, err := SelectOnnxOutputs(available, nil)
	require.NoError(t, err)
	assert.Equal(t, available, allMeta)
	assert.Equal(t, []string{"last_hidden_state", "pooler_output", "pooler_output_int8"}, allNames)

	selected, names, err := SelectOnnxOutputs(available, []string{"pooler_output_int8", "last_hidden_state"})
	require.NoError(t, err)
	assert.Equal(t, []string{"pooler_output_int8", "last_hidden_state"}, names)
	require.Len(t, selected, 2)
	assert.Equal(t, "pooler_output_int8", selected[0].Name)
	assert.Equal(t, "last_hidden_state", selected[1].Name)

	_, _, err = SelectOnnxOutputs(available, []string{"missing"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "unknown ONNX output")
	assert.True(t, strings.Contains(err.Error(), "last_hidden_state"))
	assert.True(t, strings.Contains(err.Error(), "pooler_output_int8"))
}
