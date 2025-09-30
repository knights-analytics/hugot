curl https://raw.githubusercontent.com/microsoft/onnxruntime-genai/main/examples/python/phi3-qa.py -o phi3-qa.py

hf download microsoft/Phi-3.5-mini-instruct-onnx --include gpu/gpu-int4-awq-block-128/* --local-dir .

python3 phi3-qa.py -m gpu/gpu-int4-awq-block-128 -e cuda