#!/bin/bash

set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 <mirror-url> <onnxruntime-version> <genai-version> [--cuda]" >&2
  exit 1
fi

mirror_url="$1"
onnxruntime_version="$2"
genai_version="$3"
cuda="${4:-}"
suffix=""
genai_suffix=""
if [[ "$cuda" == "--cuda" ]]; then
  suffix="-gpu_cuda13"
  genai_suffix="-cuda"
elif [[ -n "$cuda" ]]; then
  echo "Invalid argument: $cuda" >&2
  exit 1
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT
archive="$tmp_dir/onnxruntime-mirror.zip"
curl -fL "$mirror_url" -o "$archive"

ort_dir="onnxruntime-linux-x64${suffix}-${onnxruntime_version}"
genai_dir="onnxruntime-genai-${genai_version}-linux-x64${genai_suffix}"
unzip -q "$archive" "$ort_dir/lib/*" "$genai_dir/lib/*" -d "$tmp_dir"
cp "$tmp_dir/$ort_dir"/lib/libonnxruntime* /usr/lib/
cp "$tmp_dir/$genai_dir"/lib/libonnxruntime* /usr/lib/

# The mirror contains libonnxruntime.so.<version> without libonnxruntime.so.
# The Go binding and hugot's default path use the unversioned name.
if [[ ! -e /usr/lib/libonnxruntime.so ]]; then
  ln -s "libonnxruntime.so.${onnxruntime_version}" /usr/lib/libonnxruntime.so
fi

test -e /usr/lib/libonnxruntime.so
test -e /usr/lib/libonnxruntime-genai.so
