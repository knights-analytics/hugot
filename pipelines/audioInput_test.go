package pipelines

import (
	"context"
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/knights-analytics/hugot/backends"
)

func TestAudioWaveformRateValidation(t *testing.T) {
	for _, tc := range []struct {
		name   string
		inputs []AudioWaveform
		want   string
	}{
		{"mismatch", []AudioWaveform{{Samples: []float32{0.1}, SampleRate: 8000}}, "model expects 16000 Hz"},
		{"missing", []AudioWaveform{{Samples: []float32{0.1}}}, "invalid sample rate"},
		{"empty", []AudioWaveform{{SampleRate: 16000}}, "waveform 0 is empty"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := validateAudioWaveforms(tc.inputs, 16000); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("expected %q error, got %v", tc.want, err)
			}
		})
	}
}

func TestAudioPipelinesRejectMismatchedRatesBeforeTensorCreation(t *testing.T) {
	model := &backends.Model{}
	base := &backends.BasePipeline{Model: model}
	inputs := []AudioWaveform{{Samples: []float32{0.1}, SampleRate: 8000}}
	for _, tc := range []struct {
		name string
		run  func() error
	}{
		{"classification", func() error {
			_, err := (&AudioClassificationPipeline{BasePipeline: base, SampleRate: 16000}).RunWaveforms(context.Background(), inputs)
			return err
		}},
		{"ASR", func() error {
			_, err := (&AutomaticSpeechRecognitionPipeline{BasePipeline: base, SampleRate: 16000}).RunWaveforms(context.Background(), inputs)
			return err
		}},
		{"zero-shot", func() error {
			_, err := (&ZeroShotAudioClassificationPipeline{BasePipeline: base, SampleRate: 16000}).RunWaveforms(context.Background(), inputs)
			return err
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.run(); err == nil || !strings.Contains(err.Error(), "model expects 16000 Hz") {
				t.Fatalf("expected rate mismatch, got %v", err)
			}
		})
	}
}

func TestLoadWAVRejectsInvalidPCMBitDepth(t *testing.T) {
	data := make([]byte, 46)
	copy(data, "RIFF")
	binary.LittleEndian.PutUint32(data[4:8], 38)
	copy(data[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(data[16:20], 16)
	binary.LittleEndian.PutUint16(data[20:22], 1)
	binary.LittleEndian.PutUint16(data[22:24], 1)
	binary.LittleEndian.PutUint32(data[24:28], 16000)
	binary.LittleEndian.PutUint16(data[34:36], 12)
	copy(data[36:], "data")
	binary.LittleEndian.PutUint32(data[40:44], 2)
	path := filepath.Join(t.TempDir(), "invalid.wav")
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := loadWAV(path); err == nil || !strings.Contains(err.Error(), "unsupported PCM bit depth 12") {
		t.Fatalf("expected invalid bit depth error, got %v", err)
	}
}
