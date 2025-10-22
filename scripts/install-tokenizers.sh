#!/bin/bash

set -e

# Directory of *this* script
this_dir="$( cd "$( dirname "$0" )" && pwd )"
src_dir="$(realpath "${this_dir}/..")"
export src_dir

os="$1"
arch="$2"

tokenizer_version=$(grep 'github.com/daulet/tokenizers' /go.mod | awk '{print $2}')
tokenizer_version=$(echo $tokenizer_version | awk -F'-' '{print $NF}')

url="https://github.com/daulet/tokenizers/releases/download/${tokenizer_version}/libtokenizers.${os}-${arch}.tar.gz"
echo "Installing tokenizers version: $tokenizer_version from $url"

curl -LO "$url"
tar -C /usr/lib -xzf libtokenizers.${os}-${arch}.tar.gz
rm libtokenizers.${os}-${arch}.tar.gz