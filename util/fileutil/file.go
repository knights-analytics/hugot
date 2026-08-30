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
	"sync"
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

var (
	fileSystemMu sync.RWMutex
	fileSystem   FileSystem = osFileSystem{}
)

// SetFileSystem replaces the filesystem used by the package. It is intended to
// be called during application initialization, before starting concurrent work.
func SetFileSystem(system FileSystem) {
	fileSystemMu.Lock()
	if system == nil {
		system = osFileSystem{}
	}
	fileSystem = system
	fileSystemMu.Unlock()
}

func currentFileSystem() FileSystem {
	fileSystemMu.RLock()
	defer fileSystemMu.RUnlock()
	return fileSystem
}

func ReadFileBytes(ctx context.Context, filename string) ([]byte, error) {
	file, err := currentFileSystem().OpenFile(ctx, filename)
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

func OpenFile(ctx context.Context, filename string) (io.ReadCloser, error) {
	return currentFileSystem().OpenFile(ctx, filename)
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
	return currentFileSystem().CopyFile(ctx, from, to)
}

func WalkDir() func(ctx context.Context, URL string, handler OnVisit) error {
	return currentFileSystem().Walk
}

func DeleteFile(ctx context.Context, filename string) error {
	return currentFileSystem().DeleteFile(ctx, filename)
}

func FileExists(ctx context.Context, filename string) (bool, error) {
	return currentFileSystem().FileExists(ctx, filename)
}

func FileStats(ctx context.Context, filename string) (os.FileInfo, error) {
	return currentFileSystem().FileStats(ctx, filename)
}

func NewFileWriter(ctx context.Context, filename string, contentType string) (io.WriteCloser, error) {
	exists, err := FileExists(ctx, filename)
	if err != nil {
		return nil, err
	}
	if exists {
		err = currentFileSystem().DeleteFile(ctx, filename)
		if err != nil {
			return nil, err
		}
	}
	return currentFileSystem().NewFileWriter(ctx, filename, contentType)
}
