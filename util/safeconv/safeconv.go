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
