package fileutil

import (
	"context"
	"errors"
	"io"
	"io/fs"
	"os"
	"path/filepath"
)

type osFileSystem struct{}

func (osFileSystem) OpenFile(ctx context.Context, filename string) (io.ReadCloser, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return os.Open(filename)
}

func (osFileSystem) CopyFile(ctx context.Context, from string, to string) error {
	source, err := osFileSystem{}.OpenFile(ctx, from)
	if err != nil {
		return err
	}

	destination, err := os.OpenFile(to, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o644)
	if err != nil {
		_ = source.Close()
		return err
	}
	_, copyErr := io.Copy(destination, source)
	return errors.Join(copyErr, source.Close(), destination.Close())
}

func (osFileSystem) Walk(ctx context.Context, URL string, handler OnVisit) error {
	root, err := os.OpenRoot(URL)
	if err != nil {
		return err
	}

	walkErr := filepath.Walk(URL, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if err = ctx.Err(); err != nil {
			return err
		}
		parent, parentErr := filepath.Rel(URL, filepath.Dir(path))
		if parentErr != nil {
			return parentErr
		}
		if parent == "." {
			parent = ""
		}
		var reader io.Reader
		if !info.IsDir() {
			relativePath, relErr := filepath.Rel(URL, path)
			if relErr != nil {
				return relErr
			}
			file, openErr := root.Open(relativePath)
			if openErr != nil {
				return openErr
			}
			reader = file
			toContinue, visitErr := handler(ctx, URL, parent, info, reader)
			closeErr := file.Close()
			if visitErr != nil || closeErr != nil {
				return errors.Join(visitErr, closeErr)
			}
			if !toContinue {
				return fs.SkipDir
			}
			return nil
		}
		toContinue, visitErr := handler(ctx, URL, parent, info, reader)
		if visitErr != nil {
			return visitErr
		}
		if !toContinue {
			return fs.SkipDir
		}
		return nil
	})

	return errors.Join(walkErr, root.Close())
}

func (osFileSystem) DeleteFile(ctx context.Context, filename string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	return os.Remove(filename)
}

func (osFileSystem) FileExists(ctx context.Context, filename string) (bool, error) {
	if err := ctx.Err(); err != nil {
		return false, err
	}
	_, err := os.Stat(filename)
	if errors.Is(err, os.ErrNotExist) {
		return false, nil
	}
	return err == nil, err
}

func (osFileSystem) FileStats(ctx context.Context, filename string) (os.FileInfo, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return os.Stat(filename)
}

func (osFileSystem) NewFileWriter(ctx context.Context, filename string, _ string) (io.WriteCloser, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return os.OpenFile(filename, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o644)
}
