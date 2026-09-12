package main

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestDownloadFilePublishesCompleteResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("image data"))
	}))
	defer server.Close()

	destination := filepath.Join(t.TempDir(), "cat.jpg")
	if err := downloadFile(t.Context(), server.URL, destination); err != nil {
		t.Fatal(err)
	}
	content, err := os.ReadFile(destination)
	if err != nil {
		t.Fatal(err)
	}
	if string(content) != "image data" {
		t.Fatalf("unexpected downloaded content %q", content)
	}
}

func TestDownloadFileDoesNotPublishFailedResponse(t *testing.T) {
	for _, tc := range []struct {
		name    string
		handler http.HandlerFunc
	}{
		{
			name: "http error",
			handler: func(w http.ResponseWriter, _ *http.Request) {
				http.Error(w, "unavailable", http.StatusServiceUnavailable)
			},
		},
		{
			name: "truncated body",
			handler: func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Length", "100")
				_, _ = w.Write([]byte("partial"))
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(tc.handler)
			defer server.Close()

			destination := filepath.Join(t.TempDir(), "cat.jpg")
			if err := downloadFile(t.Context(), server.URL, destination); err == nil {
				t.Fatal("expected download failure")
			}
			if _, err := os.Stat(destination); !os.IsNotExist(err) {
				t.Fatalf("failed download left destination behind: %v", err)
			}
			files, err := filepath.Glob(filepath.Join(filepath.Dir(destination), ".download-*"))
			if err != nil {
				t.Fatal(err)
			}
			if len(files) != 0 {
				t.Fatalf("failed download left temporary files: %v", files)
			}
		})
	}
}
