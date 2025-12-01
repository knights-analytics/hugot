package safeconv

import (
	"math"
	"time"
)

// IntSliceToUint32Slice converts a slice of int to uint32 with clamping to avoid overflow/underflow.
func IntSliceToUint32Slice(input []int) []uint32 {
	out := make([]uint32, len(input))
	for i, v := range input {
		if v < 0 {
			out[i] = 0
		} else if v > math.MaxUint32 {
			out[i] = math.MaxUint32
		} else {
			out[i] = uint32(v)
		}
	}
	return out
}

// Uint32SliceToIntSlice converts a slice of uint32 to int with clamping to MaxInt when necessary.
func Uint32SliceToIntSlice(input []uint32) []int {
	out := make([]int, len(input))
	for i, v := range input {
		if int(v) < 0 { // extremely unlikely on typical platforms, but keep safe
			out[i] = 0
		} else if int(v) > math.MaxInt {
			out[i] = math.MaxInt
		} else {
			out[i] = int(v)
		}
	}
	return out
}

// IntOffsetsToUintPairs converts tokenizer offsets from [][]int to [][2]uint
// with clamping of negative values to 0.
func IntOffsetsToUintPairs(input [][]int) [][2]uint {
	out := make([][2]uint, len(input))
	for i, pair := range input {
		var a, b int
		if len(pair) > 0 {
			a = pair[0]
		}
		if len(pair) > 1 {
			b = pair[1]
		}
		if a < 0 {
			a = 0
		}
		if b < 0 {
			b = 0
		}
		out[i] = [2]uint{uint(a), uint(b)} // #nosec G115 both a and b are clamped to be non-negative above, so int->uint is safe here.
	}
	return out
}

// Int64ToUint32 converts int64 to uint32 with clamping into [0, MaxUint32].
func Int64ToUint32(v int64) uint32 {
	if v < 0 {
		return 0
	}
	if v > math.MaxUint32 {
		return math.MaxUint32
	}
	return uint32(v)
}

// DurationToU64 converts a duration to an unsigned nanoseconds counter safely.
// Negative durations are mapped to 0.
func DurationToU64(d time.Duration) uint64 {
	if d <= 0 {
		return 0
	}
	// Conversion from time.Duration (int64) to uint64 is safe here because negatives are handled above.
	return uint64(d) // #nosec G115
}

// U64ToDuration converts an unsigned nanoseconds count to time.Duration safely.
// Values larger than MaxInt64 are clamped to time.Duration(math.MaxInt64).
func U64ToDuration(u uint64) time.Duration {
	if u > math.MaxInt64 {
		return time.Duration(math.MaxInt64)
	}
	return time.Duration(int64(u))
}
