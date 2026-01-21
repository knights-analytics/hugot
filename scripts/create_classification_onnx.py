# Step 1: Train the Decision Tree Model in scikit-learn
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx
from pathlib import Path

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=42)

# Train a shallow decision tree model so that it can output probabilities for each class and is not perfect
model = DecisionTreeClassifier(max_depth=2)
model.fit(X_train, y_train)

model_path = "/home/rpinosio/repositories/knights/hugot/models/KnightsAnalytics_iris-decision-tree/model.onnx"
Path(model_path).parent.mkdir(parents=True, exist_ok=True)

# Step 2: Convert the Model to ONNX Format

# Define the initial types based on the input shape
initial_type = [('input', FloatTensorType([None, X_train.shape[1]]))]

# Convert the model
onnx_model = convert_sklearn(model, initial_types=initial_type, 
                            options={id(model): {'zipmap': False}})
onnx.save_model(onnx_model, model_path)

# try running inference using onnxruntime
import onnxruntime as ort

# Create an inference session
ort_session = ort.InferenceSession(model_path)

# prepare one input
import numpy as np
input_name = ort_session.get_inputs()[0].name
# Run inference on one example
outputs = ort_session.run(None, {input_name: X_test[0].reshape(1, -1).astype(np.float32)})
print("Inference outputs:", outputs)

X_test[0]