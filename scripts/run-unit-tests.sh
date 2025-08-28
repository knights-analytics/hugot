#!/bin/bash

set -e

# Directory of *this* script
this_dir="$( cd "$( dirname "$0" )" && pwd )"
src_dir="$(realpath "${this_dir}/..")"
export src_dir

commit_hash=$(git rev-parse --short HEAD)
export commit_hash
test_folder="$src_dir/testTarget"
export test_folder
mkdir -p "$test_folder"
host_uid=$(id -u "$USER")
export host_uid

# build with compose
docker compose -f "$src_dir/compose-test.yaml" build hugot && \
docker compose -f "$src_dir/compose-test.yaml" build hugot-test

echo "Running tests for commit hash: $commit_hash"
docker compose -f "$src_dir/compose-test.yaml" up && \
docker compose -f "$src_dir/compose-test.yaml" logs --no-color >& "$test_folder/logs.txt"
docker compose -f "$src_dir/compose-test.yaml" rm -fsv

echo "Extracting lib artifacts"
docker build -f ./test.Dockerfile . --output "$src_dir/artifacts" --target artifacts
echo "lib artifacts extracted"
