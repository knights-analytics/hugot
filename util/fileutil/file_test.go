package fileutil

import (
	"context"
	"io"
	"path/filepath"
	"strings"
	"testing"
)

type recordingFileSystem struct {
	FileSystem
	opened bool
}

func (f *recordingFileSystem) OpenFile(context.Context, string) (io.ReadCloser, error) {
	f.opened = true
	return io.NopCloser(strings.NewReader("injected")), nil
}

func TestDefaultFileSystemUsesOS(t *testing.T) {
	SetFileSystem(nil)
	directory := t.TempDir()
	filename := filepath.Join(directory, "file.txt")

	writer, err := NewFileWriter(context.Background(), filename, "text/plain")
	if err != nil {
		t.Fatal(err)
	}
	if _, err = writer.Write([]byte("hello")); err != nil {
		t.Fatal(err)
	}
	if err = writer.Close(); err != nil {
		t.Fatal(err)
	}

	contents, err := ReadFileBytes(context.Background(), filename)
	if err != nil {
		t.Fatal(err)
	}
	if string(contents) != "hello" {
		t.Fatalf("got %q, want %q", contents, "hello")
	}
}

func TestSetFileSystem(t *testing.T) {
	t.Cleanup(func() { SetFileSystem(nil) })
	filesystem := &recordingFileSystem{}
	SetFileSystem(filesystem)

	if _, err := ReadFileBytes(context.Background(), "ignored"); err != nil {
		t.Fatal(err)
	}
	if !filesystem.opened {
		t.Fatal("configured filesystem was not used")
	}
}
