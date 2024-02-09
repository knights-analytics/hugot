package util

import (
	"context"
	"io"
	"io/fs"
	"path/filepath"
	"strings"
	"time"

	"github.com/viant/afs"
	_ "github.com/viant/afsc/s3"
)

var FileSystem = afs.New()

// AFS FS abstraction

type AfsFS struct {
}

type AfsFile struct {
	readCloser io.ReadCloser
	path       string
}

type AfsFileInfo struct {
	fileName    string
	fileSize    int64
	fileMode    fs.FileMode
	fileModTime time.Time
	fileIsDir   bool
	fileSys     any
}

func (fileInfo AfsFileInfo) Name() string {
	return fileInfo.fileName
}

func (fileInfo AfsFileInfo) Size() int64 {
	return fileInfo.fileSize
}

func (fileInfo AfsFileInfo) Mode() fs.FileMode {
	return fileInfo.fileMode
}

func (fileInfo AfsFileInfo) ModTime() time.Time {
	return fileInfo.fileModTime
}

func (fileInfo AfsFileInfo) IsDir() bool {
	return fileInfo.fileIsDir
}

func (fileInfo AfsFileInfo) Sys() any {
	return nil
}

func (file *AfsFile) Stat() (fs.FileInfo, error) {
	object, err := FileSystem.Object(context.Background(), file.path)
	fileInfo := AfsFileInfo{
		fileName:    object.Name(),
		fileSize:    object.Size(),
		fileMode:    object.Mode(),
		fileModTime: object.ModTime(),
		fileIsDir:   object.IsDir(),
		fileSys:     object.Sys(),
	}
	return fileInfo, err
}

func (file *AfsFile) Read(p []byte) (int, error) {
	return file.readCloser.Read(p)
}

func (file *AfsFile) Close() error {
	return file.readCloser.Close()
}

func (afs AfsFS) Open(name string) (fs.File, error) {
	f, err := FileSystem.OpenURL(context.Background(), name)
	return &AfsFile{
		readCloser: f,
		path:       name,
	}, err
}

// end AFS FS abstraction

func FileExists(filename string) (bool, error) {
	exists, err := FileSystem.Exists(context.Background(), filename)
	return exists, err
}

func ReadFileBytes(filename string) ([]byte, error) {
	file, err := FileSystem.OpenURL(context.Background(), filename)
	if err != nil {
		return nil, err
	}
	defer CloseFile(file)

	outBytes, readErr := io.ReadAll(file)
	if readErr != nil {
		return nil, readErr
	}
	return outBytes, nil
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
