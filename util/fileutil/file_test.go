package fileutil

import (
	"context"
	"io"
	"os"
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
	if !IsDefaultOSFileSystem(ctx) {
		t.Fatal("nil filesystem did not select the default OS filesystem")
	}
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

func TestCopyFileCreatesDestinationDirectories(t *testing.T) {
	ctx := WithFileSystem(context.Background(), nil)
	directory := t.TempDir()
	source := filepath.Join(directory, "source.txt")
	destination := filepath.Join(directory, "nested", "model", "file.txt")

	if err := os.WriteFile(source, []byte("hello"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := CopyFile(ctx, source, destination); err != nil {
		t.Fatal(err)
	}

	contents, err := os.ReadFile(destination)
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
	if IsDefaultOSFileSystem(ctx) {
		t.Fatal("custom filesystem was identified as the default OS filesystem")
	}

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

func TestWithFileSystemNilPreservesExistingFileSystem(t *testing.T) {
	filesystem := &recordingFileSystem{}
	ctx := WithFileSystem(context.Background(), filesystem)
	ctx = WithFileSystem(ctx, nil)

	if _, err := ReadFileBytes(ctx, "ignored"); err != nil {
		t.Fatal(err)
	}
	if !filesystem.opened {
		t.Fatal("existing context filesystem was replaced")
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

func TestIsLocalPath(t *testing.T) {
	t.Parallel()

	t.Run("default OS filesystem", func(t *testing.T) {
		t.Parallel()

		ctx := WithFileSystem(context.Background(), nil)
		isLocal, err := IsLocalPath(ctx, filepath.Join(t.TempDir(), "model"))
		if err != nil {
			t.Fatalf("IsLocalPath() error = %v", err)
		}
		if !isLocal {
			t.Fatal("IsLocalPath() = false, want true")
		}
	})

	t.Run("custom filesystem without local path capability", func(t *testing.T) {
		t.Parallel()

		ctx := WithFileSystem(context.Background(), &recordingFileSystem{})
		isLocal, err := IsLocalPath(ctx, "model")
		if err != nil {
			t.Fatalf("IsLocalPath() error = %v", err)
		}
		if isLocal {
			t.Fatal("IsLocalPath() = true, want false")
		}
	})

	t.Run("custom filesystem reports local path", func(t *testing.T) {
		t.Parallel()

		ctx := WithFileSystem(context.Background(), localPathTestFileSystem{local: true})
		isLocal, err := IsLocalPath(ctx, "model")
		if err != nil {
			t.Fatalf("IsLocalPath() error = %v", err)
		}
		if !isLocal {
			t.Fatal("IsLocalPath() = false, want true")
		}
	})

	t.Run("custom filesystem reports non-local path", func(t *testing.T) {
		t.Parallel()

		ctx := WithFileSystem(context.Background(), localPathTestFileSystem{local: false})
		isLocal, err := IsLocalPath(ctx, "s3://bucket/model")
		if err != nil {
			t.Fatalf("IsLocalPath() error = %v", err)
		}
		if isLocal {
			t.Fatal("IsLocalPath() = true, want false")
		}
	})
}

type localPathTestFileSystem struct {
	FileSystem
	local bool
}

func (f localPathTestFileSystem) IsLocalPath(context.Context, string) (bool, error) {
	return f.local, nil
}
