package pipelines

import (
	"fmt"
	"github.com/knights-analytics/hugot/backends"
)

func tensorContract(meta backends.InputOutputInfo, want []int64) error {
	if len(meta.Dimensions) != len(want) {
		return fmt.Errorf("tensor %q: expected rank %d, got %v", meta.Name, len(want), meta.Dimensions)
	}
	for i, dim := range want {
		if dim >= 0 && meta.Dimensions[i] >= 0 && meta.Dimensions[i] != dim {
			return fmt.Errorf("tensor %q: axis %d is %d, expected %d", meta.Name, i, meta.Dimensions[i], dim)
		}
	}
	return nil
}
func namedTensorContract(meta []backends.InputOutputInfo, name string, want []int64) error {
	for _, info := range meta {
		if info.Name == name {
			return tensorContract(info, want)
		}
	}
	return fmt.Errorf("missing tensor %q", name)
}
