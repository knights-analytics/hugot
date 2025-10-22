#!/bin/bash

set -e

# Directory of *this* script
this_dir="$( cd "$( dirname "$0" )" && pwd )"
src_dir="$(realpath "${this_dir}/..")"
export src_dir

os="$1"
arch="$2"
onnxruntime_version="$3"

if [[ -z $onnxruntime_version ]]; then
    echo "Could not extract version for onnxruntime"
    exit 1
fi

if [[ $arch == "arm64" ]]; then
  arch="aarch64"
elif [[ $arch == "amd64" ]]; then
  arch="x64"
fi

name="onnxruntime-${os}-${arch}-${onnxruntime_version}"
url="https://github.com/microsoft/onnxruntime/releases/download/v${onnxruntime_version}/$name.tgz"

echo Downloading version "$onnxruntime_version" \(cpu\) from "${url} into $(pwd)"

function cleanup() {
    rm -r "$name.tgz" "$name" || true
}

trap cleanup EXIT

curl -LO "$url" && tar -xzf "./$name.tgz" && mv "./$name/lib/libonnxruntime.so.${onnxruntime_version}" /usr/lib64/onnxruntime.so
