//go:build cgo && (ORT || ALL) && !TRAINING

package hugot

import (
	"testing"

	"github.com/knights-analytics/hugot/options"
)

func TestNewOptions(t *testing.T) {
	opts := []options.WithOption{
		options.WithEnvLoggingLevel(options.LoggingLevelWarning),
		options.WithLogSeverityLevel(options.LoggingLevelInfo),
		options.WithGraphOptimizationLevel(options.GraphOptimizationLevelEnableBasic),
		options.WithExtraExecutionProvider("CPUExecutionProvider", map[string]string{}),
	}
	session, err := NewORTSession(opts...)
	if err != nil {
		t.Fatalf("failed to create session with new options: %v", err)
	}
	defer session.Destroy()

	// If it didn't panic or return error, the options were accepted and passed to ORT
}
