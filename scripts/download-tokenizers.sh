#!/bin/bash

set -e

# Directory of *this* script
this_dir="$( cd "$( dirname "$0" )" && pwd )"
src_dir="$(realpath "${this_dir}/..")"
export src_dir

# tokenizers
tokenizer_version=$(grep 'github.com/daulet/tokenizers' ./go.mod | awk '{print $2}')
tokenizer_version=$(echo "${tokenizer_version}" | awk -F'-' '{print $NF}')

echo "tokenizer_version: $tokenizer_version"

curl -LO "https://github.com/daulet/tokenizers/releases/download/${tokenizer_version}/libtokenizers.linux-amd64.tar.gz"
tar -xzf libtokenizers.linux-amd64.tar.gz
rm libtokenizers.linux-amd64.tar.gz
mv "./libtokenizers.a" /usr/lib/libtokenizers.a