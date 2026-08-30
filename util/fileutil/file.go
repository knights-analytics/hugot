package fileutil

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// OnVisit is called for each file or directory encountered by WalkDir.
// parent is relative to the URL passed to WalkDir and reader is provided for files.
type OnVisit func(ctx context.Context, URL string, parent string, info os.FileInfo, reader io.Reader) (toContinue bool, err error)

// FileSystem is the storage contract used by Hugot. The default implementation
// uses the local operating system, while applications can provide an adapter for
// object storage or another filesystem.
type FileSystem interface {
	OpenFile(ctx context.Context, filename string) (io.ReadCloser, error)
	CopyFile(ctx context.Context, from string, to string) error
	Walk(ctx context.Context, URL string, handler OnVisit) error
	DeleteFile(ctx context.Context, filename string) error
	FileExists(ctx context.Context, filename string) (bool, error)
	FileStats(ctx context.Context, filename string) (os.FileInfo, error)
	NewFileWriter(ctx context.Context, filename string, contentType string) (io.WriteCloser, error)
}
type fileSystemContextKey struct{}

// WithFileSystem binds a filesystem to a context. This is the preferred
// session-scoped injection mechanism.
func WithFileSystem(ctx context.Context, system FileSystem) context.Context {
	if system == nil {
		system = osFileSystem{}
	}
	return context.WithValue(ctx, fileSystemContextKey{}, system)
}

func fileSystemFor(ctx context.Context) (FileSystem, error) {
	if system, ok := ctx.Value(fileSystemContextKey{}).(FileSystem); ok && system != nil {
		return system, nil
	}
	return nil, fmt.Errorf("no filesystem bound to context")
}

func ReadFileBytes(ctx context.Context, filename string) ([]byte, error) {
	fs, fsErr := fileSystemFor(ctx)
	if fsErr != nil {
		return nil, fsErr
	}

	file, err := fs.OpenFile(ctx, filename)
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

type PathType uint8

const (
	PathTypeLocal PathType = iota
	PathTypeS3
	PathTypeGCP
	PathTypeAzureBlob
)

func getPathType(path string) PathType {
	switch {
	case strings.HasPrefix(path, "s3://"):
		return PathTypeS3
	case strings.HasPrefix(path, "gs://"), strings.HasPrefix(path, "gcs://"):
		return PathTypeGCP
	case strings.HasPrefix(path, "az://"), strings.HasPrefix(path, "azblob://"), strings.HasPrefix(path, "azure://"), strings.Contains(path, ".blob.core.windows.net"):
		return PathTypeAzureBlob
	}
	return PathTypeLocal
}

func OpenFile(ctx context.Context, filename string) (io.ReadCloser, error) {
	fs, fsErr := fileSystemFor(ctx)
	if fsErr != nil {
		return nil, fsErr
	}
	return fs.OpenFile(ctx, filename)
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
// if the path is an object storage URI, trim any trailing slashes and construct it manually from the components
// so that double slashes (e.g. s3://) are preserved.
func PathJoinSafe(elem ...string) string {
	if len(elem) == 0 {
		return ""
	}
	var path string

	switch getPathType(elem[0]) {
	case PathTypeS3, PathTypeGCP, PathTypeAzureBlob:
		basePath := strings.TrimSuffix(elem[0], "/")
		parts := make([]string, 0, len(elem))
		for _, value := range elem[1:] {
			value = strings.Trim(value, "/\\")
			if value != "" {
				parts = append(parts, value)
			}
		}
		path = basePath
		if len(parts) > 0 {
			path += "/" + strings.Join(parts, "/")
		}
	default:
		path = filepath.Join(elem...)
	}
	return path
}

func CopyFile(ctx context.Context, from string, to string) error {
	fs, fsErr := fileSystemFor(ctx)
	if fsErr != nil {
		return fsErr
	}
	return fs.CopyFile(ctx, from, to)
}

func WalkDir(ctx context.Context, URL string, handler OnVisit) error {
	fs, fsErr := fileSystemFor(ctx)
	if fsErr != nil {
		return fsErr
	}
	return fs.Walk(ctx, URL, handler)
}

func DeleteFile(ctx context.Context, filename string) error {
	fs, fsErr := fileSystemFor(ctx)
	if fsErr != nil {
		return fsErr
	}
	return fs.DeleteFile(ctx, filename)
}

func FileExists(ctx context.Context, filename string) (bool, error) {
	fs, fsErr := fileSystemFor(ctx)
	if fsErr != nil {
		return false, fsErr
	}
	return fs.FileExists(ctx, filename)
}

func FileStats(ctx context.Context, filename string) (os.FileInfo, error) {
	fs, fsErr := fileSystemFor(ctx)
	if fsErr != nil {
		return nil, fsErr
	}
	return fs.FileStats(ctx, filename)
}

func NewFileWriter(ctx context.Context, filename string, contentType string) (io.WriteCloser, error) {
	fs, fsErr := fileSystemFor(ctx)
	if fsErr != nil {
		return nil, fsErr
	}
	return fs.NewFileWriter(ctx, filename, contentType)
}
