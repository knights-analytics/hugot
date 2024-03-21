#!/bin/bash

set -e

# Directory of *this* script
this_dir="$( cd "$( dirname "$0" )" && pwd )"
export src_dir="$(realpath "${this_dir}/..")"

export commit_hash=$(git rev-parse --short HEAD)
export test_folder="$src_dir/testTarget"
mkdir -p $test_folder
export host_uid=$(id -u "$USER")

# build with compose
docker compose -f $src_dir/compose-test.yaml build
echo "Running tests for commit hash: $commit_hash"
docker compose -f $src_dir/compose-test.yaml up && \
docker compose -f $src_dir/compose-test.yaml logs --no-color >& $test_folder/logs.txt
docker compose -f $src_dir/compose-test.yaml rm -fsv

echo "Extracting lib artifacts"
docker build . --output "$src_dir/artifacts" --target artifacts
echo "lib artifacts extracted"