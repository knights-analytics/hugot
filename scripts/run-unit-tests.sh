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
docker buildx bake \
--file "$src_dir/docker-bake.hcl" \
"hugot" && \
docker buildx bake \
--file "$src_dir/docker-bake.hcl" \
"hugot-test"

echo "Running tests for commit hash: $commit_hash"
docker compose -f "$src_dir/compose-test.yaml" up --exit-code-from hugot-test && \
docker compose -f "$src_dir/compose-test.yaml" logs --no-color >& "$test_folder/logs.txt"
docker compose -f "$src_dir/compose-test.yaml" rm -fsv

echo "Extracting lib artifacts"
artifact_container=$(docker create hugot-test:latest)
trap 'docker rm -f "$artifact_container" >/dev/null 2>&1 || true' EXIT
mkdir -p "$src_dir/artifacts"
docker cp "$artifact_container:/usr/lib/libonnxruntime.so" \
  "$src_dir/artifacts/libonnxruntime-linux-x64.so"
docker cp "$artifact_container:/usr/lib/libonnxruntime-genai.so" \
  "$src_dir/artifacts/libonnxruntime-genai-linux-x64.so"
docker rm "$artifact_container" >/dev/null
trap - EXIT
echo "lib artifacts extracted"
