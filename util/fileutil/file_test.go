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
	ctx := WithFileSystem(context.Background(), nil)
	directory := t.TempDir()
	filename := filepath.Join(directory, "file.txt")

	writer, err := NewFileWriter(ctx, filename, "text/plain")
	if err != nil {
		t.Fatal(err)
	}
	if _, err = writer.Write([]byte("hello")); err != nil {
		t.Fatal(err)
	}
	if err = writer.Close(); err != nil {
		t.Fatal(err)
	}

	contents, err := ReadFileBytes(ctx, filename)
	if err != nil {
		t.Fatal(err)
	}
	if string(contents) != "hello" {
		t.Fatalf("got %q, want %q", contents, "hello")
	}
}

func TestContextFileSystemIsScoped(t *testing.T) {
	filesystem := &recordingFileSystem{}
	ctx := WithFileSystem(context.Background(), filesystem)

	if _, err := ReadFileBytes(ctx, "ignored"); err != nil {
		t.Fatal(err)
	}
	if !filesystem.opened {
		t.Fatal("context filesystem was not used")
	}

	if _, err := ReadFileBytes(context.Background(), filepath.Join(t.TempDir(), "missing")); err == nil {
		t.Fatal("unbound context unexpectedly used the injected filesystem")
	}
}

func TestPathJoinSafeHandlesEmptyAndObjectStoragePaths(t *testing.T) {
	if got := PathJoinSafe(); got != "" {
		t.Fatalf("empty path got %q", got)
	}
	tests := []struct {
		name string
		path []string
		want string
	}{
		{name: "S3", path: []string{"s3://bucket/", "/folder/", "file.json"}, want: "s3://bucket/folder/file.json"},
		{name: "GCP", path: []string{"gs://bucket/", "/folder/", "file.json"}, want: "gs://bucket/folder/file.json"},
		{name: "GCS alias", path: []string{"gcs://bucket/", "/folder/", "file.json"}, want: "gcs://bucket/folder/file.json"},
		{name: "Azure alias", path: []string{"az://container/", "/folder/", "file.json"}, want: "az://container/folder/file.json"},
		{name: "Azure Blob", path: []string{"azblob://container/", "/folder/", "file.json"}, want: "azblob://container/folder/file.json"},
		{name: "Azure Blob HTTPS", path: []string{"https://account.blob.core.windows.net/container/", "/folder/", "file.json"}, want: "https://account.blob.core.windows.net/container/folder/file.json"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := PathJoinSafe(test.path...); got != test.want {
				t.Fatalf("path got %q, want %q", got, test.want)
			}
		})
	}
}
