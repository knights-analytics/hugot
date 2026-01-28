# Script to create and export a custom binary decision tree classifier to ONNX
import torch
import torch.nn as nn

class CustomDecisionTreeONNX(nn.Module):
	def forward(self, x):
		# x: (batch_size, 2)
		cond = (x[:, 0] > 0) | (x[:, 1] > 0)
		proba = torch.zeros(x.shape[0], 2)
		proba[cond, 0] = 0.2
		proba[cond, 1] = 0.8
		proba[~cond, 0] = 0.8
		proba[~cond, 1] = 0.2
		return proba

if __name__ == "__main__":
	model = CustomDecisionTreeONNX()
	dummy_input = torch.randn(1, 2)
	torch.onnx.export(
		model,
		dummy_input,
		"custom_decision_tree.onnx",
		input_names=["input"],
		output_names=["proba"],
		dynamic_axes={"input": {0: "batch_size"}, "proba": {0: "batch_size"}},
		opset_version=11
	)
	print("Exported custom_decision_tree.onnx")
 
# try running inference using onnxruntime
import onnxruntime as ort

# Create an inference session
ort_session = ort.InferenceSession("/home/rpinosio/repositories/knights/hugot/scripts/custom_decision_tree.onnx")

# prepare one input
import numpy as np
input_name = ort_session.get_inputs()[0].name
# Run inference on one example
outputs = ort_session.run(None, {input_name: np.array([1, 0]).reshape(1, -1).astype(np.float32)})
print("Inference outputs:", outputs)
