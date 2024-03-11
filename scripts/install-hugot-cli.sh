#!/bin/bash

hugot_version=v0.0.5

echo "Installing hugot cli version $hugot_version..."

mkdir -p $HOME/lib/hugot
mkdir -p $HOME/.local/bin
onnxruntime_path=$HOME/lib/hugot/onnxruntime.so
hugot_path=$HOME/.local/bin/hugot

curl https://github.com/knights-analytics/hugot/releases/download/$hugot_version/onnxruntime.so -o $onnxruntime_path
curl https://github.com/knights-analytics/hugot/releases/download/$hugot_version/hugot-cli-linux-amd64 -o $hugot_path

echo "onnxruntime.so shared library installed at $onnxruntime_path"
echo "hugot binary installed at $hugot_path"

echo "Installation complete."