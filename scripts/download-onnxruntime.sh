#!/bin/bash

set -e

# Directory of *this* script
this_dir="$( cd "$( dirname "$0" )" && pwd )"
src_dir="$(realpath "${this_dir}/..")"
export src_dir

onnxruntime_version="$1"
gpu="$2"

if [[ -z $onnxruntime_version ]]; then
    echo version is required
    exit 1
fi

version=""
if [[ -n $gpu ]]; then
    version="-gpu"
fi

url="https://github.com/microsoft/onnxruntime/releases/download/v${onnxruntime_version}/onnxruntime-linux-x64${version}-${onnxruntime_version}.tgz"

echo Downloading version "$onnxruntime_version${version}" from "${url} into $(pwd)"

function cleanup() {
    rm -r "onnxruntime-linux-x64${version}-${onnxruntime_version}.tgz" "onnxruntime-linux-x64${version}-${onnxruntime_version}" || true
}

trap cleanup EXIT

curl -LO "$url" && tar -xzf "./onnxruntime-linux-x64${version}-${onnxruntime_version}.tgz" && \
    cp onnxruntime-linux-x64"${version}-${onnxruntime_version}"/lib/libonnxruntime* /usr/lib