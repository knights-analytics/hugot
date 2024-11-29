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

optimum-cli export onnx --model distilbert/distilbert-base-uncased-finetuned-sst-2-english --opset=13 --task=text-classification "$models_dir/KnightsAnalytics_distilbert-base-uncased-finetuned-sst-2-english"
optimum-cli export onnx --model SamLowe/roberta-base-go_emotions --opset=14 --task=text-classification "$models_dir/KnightsAnalytics_roberta-base-go_emotions"
optimum-cli export onnx --model MoritzLaurer/deberta-v3-base-zeroshot-v1 --opset=13 --task=zero-shot-classification "$models_dir/KnightsAnalytics_deberta-v3-base-zeroshot-v1"
optimum-cli export onnx --model dslim/distilbert-NER --opset=13 --task=token-classification "$models_dir/KnightsAnalytics_distilbert-NER"

echo "All models downloaded to $models_dir"
echo "OK."