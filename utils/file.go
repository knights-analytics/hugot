package util

import (
	"bytes"
	"context"
	"io"
	"io/fs"
	"os/user"
	"path/filepath"
	"strings"
	"time"

	"github.com/Knights-Analytics/HuGo/utils/checks"
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

func FileExists(filename string) bool {
	exists, err := FileSystem.Exists(context.Background(), filename)
	checks.Check(err)
	return exists
}

func OpenFile(filename string) io.ReadCloser {
	file, err := FileSystem.OpenURL(context.Background(), filename)
	checks.Check(err)
	return file
}

func DeleteFile(filename string) {
	checks.Check(FileSystem.Delete(context.Background(), filename))
}

func CopyFile(from string, to string) {
	checks.Check(FileSystem.Copy(context.Background(), from, to))
}

func MoveFile(from string, to string) {
	checks.Check(FileSystem.Move(context.Background(), from, to))
}

func ReadFileString(filename string) string {
	return string(ReadFileBytes(filename))
}

func WriteFileString(filename string, data string) {
	checks.Check(FileSystem.Upload(context.Background(), filename, 0664, strings.NewReader(data)))
}

func ReadFileBytes(filename string) []byte {
	file, err := FileSystem.OpenURL(context.Background(), filename)
	checks.Check(err)
	defer CloseFile(file)

	outBytes, readErr := io.ReadAll(file)
	checks.Check(readErr)

	return outBytes
}

func WriteFileBytes(filename string, data []byte) {
	checks.Check(FileSystem.Upload(context.Background(), filename, 0664, bytes.NewReader(data)))
}

func CloseFile(file io.Closer) {
	err := file.Close()
	checks.Check(err)
}

func NewFileWriter(filename string) io.WriteCloser {
	if FileExists(filename) {
		checks.Check(FileSystem.Delete(context.Background(), filename))
	}
	writer, err := FileSystem.NewWriter(context.Background(), filename, 0644)
	checks.Check(err)
	return writer
}

func ReplaceHomeDir(path string) string {
	usr, _ := user.Current()
	dir := usr.HomeDir

	if path == "~" || path == "$HOME" {
		// In case of "~", which won't be caught by the "else if"
		path = dir
	} else if strings.HasPrefix(path, "~/") || strings.HasPrefix(path, "$HOME/") {
		// Use strings.HasPrefix so we don't match paths like
		// "/something/~/something/"
		path = PathJoinSafe(dir, path[2:])
	}
	return path
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
