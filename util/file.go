package util

import (
	"bufio"
	"context"
	"errors"
	"io"
	"path/filepath"
	"strings"

	"github.com/viant/afs"
	_ "github.com/viant/afsc/s3"
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

func OpenFile(filename string) (io.ReadCloser, error) {
	return FileSystem.OpenURL(context.Background(), filename)
}

// ReadLine returns a single line (without the ending \n)
// from the input buffered reader.
// An error is returned if there is an error with the
// buffered reader.
// This function is needed to avoid the 65K char line limit
func ReadLine(r *bufio.Reader) ([]byte, error) {
	var (
		isPrefix       = true
		err      error = nil
		line, ln []byte
	)
	for isPrefix && err == nil {
		line, isPrefix, err = r.ReadLine()
		ln = append(ln, line...)
	}
	return ln, err
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
