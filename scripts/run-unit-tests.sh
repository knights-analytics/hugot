#!/bin/bash

set -e

# Directory of *this* script
this_dir="$( cd "$( dirname "$0" )" && pwd )"
export src_dir="$(realpath "${this_dir}/..")"

export commit_hash=$(git rev-parse --short HEAD)
export test_folder="$src_dir/testTarget"
mkdir -p $test_folder
export host_uid=$(id -u "$USER")

echo "Downloading the models required for testing from the huggingface hub"

if [[ ! -d "$src_dir/models" ]]; then
    mkdir -p $src_dir/models
    (cd $src_dir/models && git clone https://huggingface.co/KnightsAnalytics/all-MiniLM-L6-v2 && \
    git clone https://huggingface.co/KnightsAnalytics/distilbert-base-uncased-finetuned-sst-2-english && \
    git clone https://huggingface.co/KnightsAnalytics/distilbert-NER)
fi

# build with compose
docker compose -f $src_dir/.ci/docker-compose.yaml build
# (cd $src_dir/.ci && docker compose build)

echo "Running tests for commit hash: $commit_hash"

docker compose -f $src_dir/.ci/docker-compose.yaml up && \
docker compose -f $src_dir/.ci/docker-compose.yaml logs --no-color >& $test_folder/logs.txt && \
docker compose -f $src_dir/.ci/docker-compose.yaml rm -fsv
