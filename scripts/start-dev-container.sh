#!/bin/bash

set -e

# Directory of *this* script
this_dir="$( cd "$( dirname "$0" )" && pwd )"
export src_dir="$(realpath "${this_dir}/..")"

echo "Downloading the models required for testing from the huggingface hub"

if [[ ! -d "$src_dir/models" ]]; then
    mkdir -p $src_dir/models
    (cd $src_dir/models && git clone https://huggingface.co/KnightsAnalytics/all-MiniLM-L6-v2 &&\
    git clone https://huggingface.co/KnightsAnalytics/distilbert-base-uncased-finetuned-sst-2-english && \
    git clone https://huggingface.co/KnightsAnalytics/distilbert-NER)
fi

docker compose -f ./docker-compose-dev.yaml build
docker compose -f ./docker-compose-dev.yaml up -d