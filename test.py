from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM
from optimum.pipelines import pipeline

model = ORTModelForCausalLM.from_pretrained("models/Microsoft_Phi-3-mini-4k-instruct-onnx")
tokenizer = AutoTokenizer.from_pretrained("models/Microsoft_Phi-3-mini-4k-instruct-onnx")

p = pipeline("text-generation", model=model, tokenizer=tokenizer)

message = ["How are you today?"]

# messages = [
#     {"role": "system", "content": "You are a helpful AI assistant."},
#     {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
#     {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
#     {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
# ]

res = p(message, max_new_tokens=100)
print(res)

from transformers.pipelines import pipeline