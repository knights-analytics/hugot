package fileutil

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/viant/afs"
	"github.com/viant/afs/option"
	"github.com/viant/afs/option/content"
	"github.com/viant/afs/storage"
)

var fileSystem = afs.New()

const partSize = 64 * 1024 * 1024

func ReadFileBytes(filename string) ([]byte, error) {
	file, err := fileSystem.OpenURL(context.Background(), filename)
	if err != nil {
		return nil, err
	}
	defer func(file io.Closer) {
		err = errors.Join(err, CloseFile(file))
	}(file)

	buf := &bytes.Buffer{}
	_, readErr := io.Copy(buf, file)
	if readErr != nil {
		return nil, readErr
	}
	return buf.Bytes(), err
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
	return fileSystem.OpenURL(context.Background(), filename)
}

// ReadLine returns a single line (without the ending \n)
// from the input buffered reader.
// An error is returned if there is an error with the
// buffered reader.
// This function is needed to avoid the 65K char line limit.
func ReadLine(r *bufio.Reader) ([]byte, error) {
	var (
		isPrefix = true
		err      error
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

func CopyFile(ctx context.Context, from string, to string) error {
	return fileSystem.Copy(ctx, from, to, option.NewSource(option.NewStream(partSize, 0)), option.NewDest(option.NewSkipChecksum(true)))
}

func WalkDir() func(ctx context.Context, URL string, handler storage.OnVisit, options ...storage.Option) error {
	return fileSystem.Walk
}

func DeleteFile(filename string) error {
	return fileSystem.Delete(context.Background(), filename)
}

func CreateFile(fileName string, isDir bool) error {
	return fileSystem.Create(context.Background(), fileName, os.ModePerm, isDir)
}

func FileExists(filename string) (bool, error) {
	return fileSystem.Exists(context.Background(), filename)
}

func NewFileWriter(filename string, contentType string) (io.WriteCloser, error) {
	exists, err := FileExists(filename)
	if err != nil {
		return nil, err
	}
	if exists {
		err = fileSystem.Delete(context.Background(), filename)
		if err != nil {
			return nil, err
		}
	}
	if contentType != "" {
		return fileSystem.NewWriter(context.Background(), filename, 0o644, content.NewMeta(content.Type, contentType), option.NewSkipChecksum(true))
	}
	return fileSystem.NewWriter(context.Background(), filename, 0o644, option.NewSkipChecksum(true))
}
