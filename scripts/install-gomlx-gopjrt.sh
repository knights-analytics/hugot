#!/bin/bash

set -e

# Directory of *this* script
this_dir="$( cd "$( dirname "$0" )" && pwd )"
src_dir="$(realpath "${this_dir}/..")"
export src_dir

os="$1"
arch="$2"

gopjrt_version=$(grep 'github.com/gomlx/gopjrt' /go.mod | awk '{print $2}' | sed 's/^v//')

if [[ -z gopjrt_version ]]; then
    echo "Could not extract version for gopjrt"
    exit 1
fi

if [[ $os == "linux" ]]; then
  os="darwin"
fi

url="https://raw.githubusercontent.com/gomlx/gopjrt/main/cmd/install_${os}_${arch}.sh"
echo "Installing gomlx/gopjrt from '$url', version: $gopjrt_version"

export GOPJRT_VERSION="$gopjrt_version"
curl -sSf "$url" | bash