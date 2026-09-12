package pipelines

import (
	"encoding/binary"
	"errors"
	"fmt"
	"os"

	"github.com/knights-analytics/hugot/backends"
)

const defaultAudioSampleRate = 16000

func loadAudioFiles(paths []string) ([]AudioWaveform, error) {
	waveforms := make([]AudioWaveform, len(paths))
	for i, path := range paths {
		waveform, err := loadWAV(path)
		if err != nil {
			return nil, fmt.Errorf("failed to load audio %q: %w", path, err)
		}
		waveforms[i] = waveform
	}
	return waveforms, nil
}

func audioSamples(inputs []AudioWaveform, expectedSampleRate int) ([][]float32, error) {
	if err := validateAudioWaveforms(inputs, expectedSampleRate); err != nil {
		return nil, err
	}
	samples := make([][]float32, len(inputs))
	for i := range inputs {
		samples[i] = inputs[i].Samples
	}
	return samples, nil
}

func createAudioInputTensors(batch *backends.PipelineBatch, model *backends.Model, inputs [][]float32) error {
	for i, samples := range inputs {
		if len(samples) == 0 {
			return fmt.Errorf("waveform %d is empty", i)
		}
	}
	if model.Backend == nil {
		return errors.New("audio tensor creation is unavailable for the configured backend")
	}
	return model.Backend.CreateAudioTensors(batch, model, inputs)
}

func validateAudioWaveforms(inputs []AudioWaveform, expectedSampleRate int) error {
	if expectedSampleRate <= 0 {
		return fmt.Errorf("audio model sample rate must be greater than zero, got %d", expectedSampleRate)
	}
	for i, input := range inputs {
		if len(input.Samples) == 0 {
			return fmt.Errorf("waveform %d is empty", i)
		}
		if input.SampleRate <= 0 {
			return fmt.Errorf("waveform %d has invalid sample rate %d", i, input.SampleRate)
		}
		if input.SampleRate != expectedSampleRate {
			return fmt.Errorf("waveform %d has sample rate %d Hz; model expects %d Hz; resample the waveform or configure the model sample rate", i, input.SampleRate, expectedSampleRate)
		}
	}
	return nil
}

func loadWAV(path string) (AudioWaveform, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return AudioWaveform{}, err
	}
	if len(data) < 12 || string(data[:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return AudioWaveform{}, errors.New("unsupported audio file: expected RIFF/WAVE")
	}
	var format, channels, bits uint16
	var sampleRate uint32
	var audio []byte
	for offset := 12; offset < len(data); {
		if offset+8 > len(data) {
			return AudioWaveform{}, errors.New("invalid WAV chunk header")
		}
		size := uint64(binary.LittleEndian.Uint32(data[offset+4 : offset+8]))
		start := uint64(offset + 8)
		end := start + size
		if end > uint64(len(data)) {
			return AudioWaveform{}, errors.New("invalid WAV chunk size")
		}
		switch string(data[offset : offset+4]) {
		case "fmt ":
			if size < 16 {
				return AudioWaveform{}, errors.New("invalid WAV format chunk")
			}
			format = binary.LittleEndian.Uint16(data[start : start+2])
			channels = binary.LittleEndian.Uint16(data[start+2 : start+4])
			sampleRate = binary.LittleEndian.Uint32(data[start+4 : start+8])
			bits = binary.LittleEndian.Uint16(data[start+14 : start+16])
		case "data":
			audio = data[start:end]
		}
		next := end + size%2
		if next > uint64(len(data)) {
			return AudioWaveform{}, errors.New("invalid WAV chunk padding")
		}
		offset = int(next)
	}
	if format != 1 || channels != 1 || len(audio) == 0 {
		return AudioWaveform{}, errors.New("unsupported WAV layout: only mono PCM is supported")
	}
	if sampleRate == 0 {
		return AudioWaveform{}, errors.New("invalid WAV sample rate 0")
	}
	bytesPerSample := int(bits / 8)
	if bits%8 != 0 || (bytesPerSample != 1 && bytesPerSample != 2 && bytesPerSample != 3 && bytesPerSample != 4) {
		return AudioWaveform{}, fmt.Errorf("unsupported PCM bit depth %d", bits)
	}
	if len(audio)%bytesPerSample != 0 {
		return AudioWaveform{}, errors.New("invalid WAV data chunk size: incomplete PCM sample")
	}
	samples := make([]float32, len(audio)/bytesPerSample)
	for i := range samples {
		chunk := audio[i*bytesPerSample : (i+1)*bytesPerSample]
		switch bytesPerSample {
		case 1:
			samples[i] = (float32(chunk[0]) - 128) / 128
		case 2:
			value := int32(binary.LittleEndian.Uint16(chunk))
			if value >= 1<<15 {
				value -= 1 << 16
			}
			samples[i] = float32(value) / 32768
		case 3:
			value := int32(chunk[0]) | int32(chunk[1])<<8 | int32(chunk[2])<<16
			if value&0x800000 != 0 {
				value |= ^0xffffff
			}
			samples[i] = float32(value) / 8388608
		case 4:
			value := int64(binary.LittleEndian.Uint32(chunk))
			if value >= 1<<31 {
				value -= 1 << 32
			}
			samples[i] = float32(value) / 2147483648
		}
	}
	return AudioWaveform{Samples: samples, SampleRate: int(sampleRate)}, nil
}