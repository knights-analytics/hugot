#!/bin/bash

set -e

# Directory of *this* script
this_dir="$( cd "$( dirname "$0" )" && pwd )"
src_dir="$(realpath "${this_dir}/..")"
export src_dir

onnxruntime_version="$1"

if [[ -z $onnxruntime_version ]]; then
    echo version is required
    exit 1
fi

name="onnxruntime-linux-x64-${onnxruntime_version}"
url="https://github.com/microsoft/onnxruntime/releases/download/v${onnxruntime_version}/$name.tgz"
url_gpu=https://github.com/microsoft/onnxruntime/releases/download/v${onnxruntime_version}/onnxruntime-linux-x64-gpu-${onnxruntime_version}.tgz && \

echo Downloading version "$onnxruntime_version" \(cpu and gpu\) from "${url} into $(pwd)"

function cleanup() {
    rm -r "$name.tgz" "$name" "onnxruntime-linux-x64-gpu-${onnxruntime_version}.tgz" || true
}

trap cleanup EXIT

curl -LO "$url" && tar -xzf "./$name.tgz" && mv "./$name/lib/libonnxruntime.so.${onnxruntime_version}" /usr/lib64/onnxruntime.so && \
curl -LO "$url_gpu" && tar -xzf "./onnxruntime-linux-x64-gpu-${onnxruntime_version}.tgz" && mv "onnxruntime-linux-x64-gpu-${onnxruntime_version}/lib" /usr/lib64/onnxruntime-gpu
