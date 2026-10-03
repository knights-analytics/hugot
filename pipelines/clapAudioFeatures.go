package pipelines

import (
	"errors"
	"math"
	"sync"
)

const (
	clapSampleRate = 48000
	clapMaxSamples = 10 * clapSampleRate
	clapFFTSize    = 1024
	clapHopLength  = 480
	clapMelBins    = 64
)

var (
	clapMelOnce sync.Once
	clapMelBank [][]float64
)

func clapAudioFeatures(samples []float32, sampleRate int) ([][][]float32, bool, error) {
	if len(samples) == 0 {
		return nil, false, errors.New("waveform is empty")
	}
	if sampleRate <= 0 {
		return nil, false, errors.New("audio sample rate must be greater than zero")
	}
	if sampleRate != clapSampleRate {
		samples = resampleAudio(samples, sampleRate, clapSampleRate)
	}
	longer := len(samples) > clapMaxSamples
	if longer {
		start := (len(samples) - clapMaxSamples) / 2
		samples = samples[start : start+clapMaxSamples]
	} else if len(samples) < clapMaxSamples {
		padded := make([]float32, clapMaxSamples)
		repeats := clapMaxSamples / len(samples)
		for i := range repeats {
			copy(padded[i*len(samples):], samples)
		}
		samples = padded
	}

	clapMelOnce.Do(func() { clapMelBank = createCLAPMelBank() })
	frames := clapMaxSamples/clapHopLength + 1
	features := make([][]float32, frames)
	window := make([]float64, clapFFTSize)
	for i := range window {
		window[i] = 0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(clapFFTSize))
	}
	for frame := range frames {
		fftInput := make([]complex128, clapFFTSize)
		frameStart := frame*clapHopLength - clapFFTSize/2
		for i := range clapFFTSize {
			sampleIndex := reflectAudioIndex(frameStart+i, len(samples))
			fftInput[i] = complex(float64(samples[sampleIndex])*window[i], 0)
		}
		fftInPlace(fftInput)
		power := make([]float64, clapFFTSize/2+1)
		for i := range power {
			power[i] = real(fftInput[i])*real(fftInput[i]) + imag(fftInput[i])*imag(fftInput[i])
		}
		features[frame] = make([]float32, clapMelBins)
		for melIndex, filter := range clapMelBank {
			energy := 0.0
			for bin, weight := range filter {
				energy += power[bin] * weight
			}
			features[frame][melIndex] = float32(10 * math.Log10(max(energy, 1e-10)))
		}
	}
	return [][][]float32{features}, longer, nil
}

func reflectAudioIndex(index, length int) int {
	for index < 0 || index >= length {
		if index < 0 {
			index = -index
		} else {
			index = 2*length - 2 - index
		}
	}
	return index
}

func fftInPlace(values []complex128) {
	n := len(values)
	for i, j := 1, 0; i < n; i++ {
		bit := n >> 1
		for ; j&bit != 0; bit >>= 1 {
			j ^= bit
		}
		j ^= bit
		if i < j {
			values[i], values[j] = values[j], values[i]
		}
	}
	for size := 2; size <= n; size <<= 1 {
		angle := -2 * math.Pi / float64(size)
		root := complex(math.Cos(angle), math.Sin(angle))
		for start := 0; start < n; start += size {
			factor := complex(1.0, 0)
			half := size / 2
			for offset := range half {
				even := values[start+offset]
				odd := values[start+offset+half] * factor
				values[start+offset] = even + odd
				values[start+offset+half] = even - odd
				factor *= root
			}
		}
	}
}

func createCLAPMelBank() [][]float64 {
	minMel := hzToSlaneyMel(50)
	maxMel := hzToSlaneyMel(14000)
	melPoints := make([]float64, clapMelBins+2)
	for i := range melPoints {
		melPoints[i] = slaneyMelToHz(minMel + (maxMel-minMel)*float64(i)/float64(clapMelBins+1))
	}
	filters := make([][]float64, clapMelBins)
	for melIndex := range filters {
		filters[melIndex] = make([]float64, clapFFTSize/2+1)
		left, center, right := melPoints[melIndex], melPoints[melIndex+1], melPoints[melIndex+2]
		for bin := range filters[melIndex] {
			frequency := float64(bin) * clapSampleRate / clapFFTSize
			lower := (frequency - left) / (center - left)
			upper := (right - frequency) / (right - center)
			filters[melIndex][bin] = max(0, min(lower, upper)) * 2 / (right - left)
		}
	}
	return filters
}

func hzToSlaneyMel(hz float64) float64 {
	const minLogHz = 1000.0
	const minLogMel = 15.0
	const logStep = 0.06875177742094912
	if hz < minLogHz {
		return hz / (200.0 / 3.0)
	}
	return minLogMel + math.Log(hz/minLogHz)/logStep
}

func slaneyMelToHz(mel float64) float64 {
	const minLogHz = 1000.0
	const minLogMel = 15.0
	const logStep = 0.06875177742094912
	if mel < minLogMel {
		return mel * (200.0 / 3.0)
	}
	return minLogHz * math.Exp(logStep*(mel-minLogMel))
}

func resampleAudio(samples []float32, sourceRate, targetRate int) []float32 {
	if len(samples) == 0 || sourceRate <= 0 || targetRate <= 0 {
		return nil
	}
	if sourceRate == targetRate {
		return append([]float32(nil), samples...)
	}
	outputLength := int(math.Round(float64(len(samples)) * float64(targetRate) / float64(sourceRate)))
	if outputLength < 1 {
		return nil
	}
	output := make([]float32, outputLength)
	for i := range output {
		position := float64(i) * float64(sourceRate) / float64(targetRate)
		left := int(position)
		if left >= len(samples)-1 {
			output[i] = samples[len(samples)-1]
			continue
		}
		fraction := float32(position - float64(left))
		output[i] = samples[left]*(1-fraction) + samples[left+1]*fraction
	}
	return output
}
