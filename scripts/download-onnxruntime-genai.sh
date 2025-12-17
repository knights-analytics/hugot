#!/bin/bash

set -e

# Directory of *this* script
this_dir="$( cd "$( dirname "$0" )" && pwd )"
src_dir="$(realpath "${this_dir}/..")"
export src_dir

onnxruntime_genai_version="$1"
gpu="$2"

if [[ -z $onnxruntime_genai_version ]]; then
    echo version is required
    exit 1
fi

version=""
if [[ -n $gpu ]]; then
    version="-cuda"
fi

url="https://github.com/microsoft/onnxruntime-genai/releases/download/v${onnxruntime_genai_version}/onnxruntime-genai-${onnxruntime_genai_version}-linux-x64${version}.tar.gz"
echo Downloading version "${onnxruntime_genai_version}${version}" from "${url} into $(pwd)"

function cleanup() {
    rm -r "onnxruntime-genai-${onnxruntime_genai_version}-linux-x64${version}.tar.gz" "onnxruntime-genai-${onnxruntime_genai_version}-linux-x64${version}" || true
}

trap cleanup EXIT

curl -LO "$url" && tar -xzf "./onnxruntime-genai-${onnxruntime_genai_version}-linux-x64${version}.tar.gz" && \
    cp onnxruntime-genai-"${onnxruntime_genai_version}"-linux-x64${version}/lib/libonnxruntime* /usr/lib