package util

import (
	"context"
	"errors"
	"github.com/viant/afs"
	_ "github.com/viant/afsc/s3"
	"io"
	"path/filepath"
	"strings"
)

var FileSystem = afs.New()

func ReadFileBytes(filename string) ([]byte, error) {
	file, err := FileSystem.OpenURL(context.Background(), filename)
	if err != nil {
		return nil, err
	}
	defer func(file io.Closer) {
		err = errors.Join(err, CloseFile(file))
	}(file)

	outBytes, readErr := io.ReadAll(file)
	if readErr != nil {
		return nil, readErr
	}
	return outBytes, err
}

func CloseFile(file io.Closer) error {
	return file.Close()
}

func GetPathType(path string) string {
	if strings.HasPrefix(path, "s3://") {
		return "S3"
	}
	return "os"
}

// PathJoinSafe wrapper around filepath.Join to ensure that paths are correctly constructed
// if the path is a normal OS path, just use filepath.Join
// if the path is S3, trim any trailing slashes and construct it manually from the components
// so that double slashes (e.g. s3://) are preserved.
func PathJoinSafe(elem ...string) string {
	var path string

	switch GetPathType(elem[0]) {
	case "S3":
		basePath := strings.TrimSuffix(elem[0], "/")
		path = basePath + string(filepath.Separator) + filepath.Join(elem[1:]...)
	default:
		path = filepath.Join(elem...)
	}
	return path
}
