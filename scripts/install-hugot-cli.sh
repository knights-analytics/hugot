#!/bin/bash

hugot_version=v0.0.7

echo "Installing hugot cli version $hugot_version..."

mkdir -p $HOME/lib/hugot
mkdir -p $HOME/.local/bin
onnxruntime_path=$HOME/lib/hugot/onnxruntime.so
hugot_path=$HOME/.local/bin/hugot

curl -L https://github.com/knights-analytics/hugot/releases/download/$hugot_version/onnxruntime.so -o $onnxruntime_path
curl -L https://github.com/knights-analytics/hugot/releases/download/$hugot_version/hugot-cli-linux-amd64 -o $hugot_path
chmod +x $hugot_path

echo "onnxruntime.so shared library installed at $onnxruntime_path"
echo "hugot binary installed at $hugot_path"

echo "Installation complete."
