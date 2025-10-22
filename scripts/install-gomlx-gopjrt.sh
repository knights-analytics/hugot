#!/bin/bash

set -e

gopjrt_version=$(grep 'github.com/gomlx/gopjrt' /go.mod | awk '{print $2}' | sed 's/^v//')

if [[ -z gopjrt_version ]]; then
    echo "Could not extract version for gopjrt"
    exit 1
fi

GOPROXY=direct go run github.com/gomlx/gopjrt/cmd/gopjrt_installer@latest -plugin=amazonlinux -version=v${gopjrt_version} -path=/usr/local

export GOPJRT_VERSION="$gopjrt_version"
