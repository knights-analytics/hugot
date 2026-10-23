package pipelines

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"image/png"
	"io"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// FFmpegVideoDecoder is an optional file decoder. FFmpegPath and FFprobePath
// default to ffmpeg and ffprobe on PATH. No codec library is linked into Hugot.
type FFmpegVideoDecoder struct {
	FFmpegPath  string
	FFprobePath string
}

func (d FFmpegVideoDecoder) Decode(ctx context.Context, path string, maxFrames int) (VideoInput, error) {
	if err := ctx.Err(); err != nil {
		return VideoInput{}, err
	}
	if strings.TrimSpace(path) == "" || maxFrames <= 0 {
		return VideoInput{}, errors.New("video decoder requires a path and positive frame limit")
	}
	ffmpeg, ffprobe := d.FFmpegPath, d.FFprobePath
	if ffmpeg == "" {
		ffmpeg = "ffmpeg"
	}
	if ffprobe == "" {
		ffprobe = "ffprobe"
	}
	// Count decoded frames rather than seconds, preserving variable frame rates.
	probe := exec.CommandContext(ctx, ffprobe, "-v", "error", "-select_streams", "v:0", "-read_intervals", fmt.Sprintf("%%+#%d", maxFrames), "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", path)
	var stderr bytes.Buffer
	probe.Stderr = &stderr
	metadata, err := probe.Output()
	if err != nil {
		if ctx.Err() != nil {
			return VideoInput{}, ctx.Err()
		}
		return VideoInput{}, fmt.Errorf("ffprobe: %w: %s", err, stderr.String())
	}
	var info struct {
		Frames []struct {
			Timestamp string `json:"best_effort_timestamp_time"`
		} `json:"frames"`
	}
	if err := json.Unmarshal(metadata, &info); err != nil {
		return VideoInput{}, fmt.Errorf("ffprobe metadata: %w", err)
	}
	cmd := exec.CommandContext(ctx, ffmpeg, "-v", "error", "-nostdin", "-i", path, "-map", "0:v:0", "-frames:v", strconv.Itoa(maxFrames), "-fps_mode", "passthrough", "-f", "image2pipe", "-vcodec", "png", "-")
	stderr.Reset()
	cmd.Stderr = &stderr
	pipe, err := cmd.StdoutPipe()
	if err != nil {
		return VideoInput{}, err
	}
	if err := cmd.Start(); err != nil {
		return VideoInput{}, fmt.Errorf("ffmpeg: %w", err)
	}
	reader := bufio.NewReader(pipe)
	out := VideoInput{}
	var decodeErr error
	for len(out.Frames) < maxFrames {
		if _, err := reader.Peek(1); err != nil {
			if !errors.Is(err, io.EOF) {
				decodeErr = err
			}
			break
		}
		frame, err := png.Decode(reader)
		if err != nil {
			decodeErr = fmt.Errorf("ffmpeg PNG output: %w", err)
			break
		}
		out.Frames = append(out.Frames, frame)
	}
	if decodeErr != nil {
		_ = cmd.Process.Kill()
	}
	waitErr := cmd.Wait()
	if ctx.Err() != nil {
		return VideoInput{}, ctx.Err()
	}
	if decodeErr != nil {
		return VideoInput{}, decodeErr
	}
	if waitErr != nil {
		return VideoInput{}, fmt.Errorf("ffmpeg: %w: %s", waitErr, stderr.String())
	}
	if len(out.Frames) == 0 {
		return VideoInput{}, errors.New("video contains no decoded frames")
	}
	// Some containers do not supply timestamps. Preserve them only when ffprobe
	// provides a complete sequence; sampling itself never depends on timestamps.
	if len(info.Frames) >= len(out.Frames) {
		for _, f := range info.Frames[:len(out.Frames)] {
			seconds, err := strconv.ParseFloat(f.Timestamp, 64)
			if err != nil || seconds < 0 || seconds > float64(time.Duration(1<<63-1))/float64(time.Second) {
				out.Timestamps = nil
				break
			}
			out.Timestamps = append(out.Timestamps, time.Duration(seconds*float64(time.Second)))
		}
	}
	return out, nil
}
