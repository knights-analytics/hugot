#!/bin/bash

set -e

# Directory of *this* script
this_dir="$( cd "$( dirname "$0" )" && pwd )"
src_dir="$(realpath "${this_dir}/..")"
export src_dir

models_dir="${src_dir}/models"
mkdir -p "$models_dir"

# Download test models
echo "Downloading test models to $models_dir"

# optimum-cli export onnx --model MoritzLaurer/deberta-v3-base-zeroshot-v1 --task=zero-shot-classification "$models_dir/deberta-v3-base-zeroshot-v1"
optimum-cli export onnx --model distilbert/distilbert-base-uncased --task=feature-extraction "$models_dir/distilbert-base-uncased"


echo "All models downloaded to $models_dir"
echo "OK."