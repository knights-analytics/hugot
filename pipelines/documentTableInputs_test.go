package pipelines

import (
	"github.com/knights-analytics/hugot/backends"
	"image"
	"testing"
)

func TestDocumentQuestionAnsweringRunWithImagesFillsDocumentPath(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 2, 2))
	values, cleanup, err := documentQuestionAnsweringRunWithImages([]backends.ImageTextInput{
		{Image: img, Text: "q"},
		{ImagePath: "/img/c.jpg", Text: "q2"},
	})
	if err != nil {
		t.Fatalf("adapter error = %v", err)
	}
	defer cleanup()
	if len(values) != 2 {
		t.Fatalf("want 2 inputs, got %d", len(values))
	}
	if values[0].DocumentPath == "" || values[1].DocumentPath != "/img/c.jpg" {
		t.Fatalf("DocumentPath mapping = %q %q", values[0].DocumentPath, values[1].DocumentPath)
	}
}
