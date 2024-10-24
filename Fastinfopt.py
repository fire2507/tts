import torch

# Load fine-tuned model
model = torch.load(r'C:\Users\user\Desktop\Practice\IITR\trained_model.py')

# Convert to quantized model
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Save quantized model
torch.save(quantized_model, "quantized_model.pth")

import torch.nn.utils.prune as prune

# Apply pruning to a layer
prune.l1_unstructured(model.layer_name, name='weight', amount=0.4)

# Remove the pruning reparameterization to speed up inference
prune.remove(model.layer_name, 'weight')

import time

start = time.time()
speech = model.synthesize("Technical term example.")
end = time.time()

print("Inference Time (quantized):", end - start)
