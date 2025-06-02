import onnxruntime as ort
from torchvision import transforms
from PIL import Image
import torch
import numpy as np

# Load and preprocess image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Converts to [C, H, W] and scales to [0, 1]
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Typical for ImageNet models
])

img = Image.open("/workspace/E2EMLOps/data/processed/vegfruits/test/apple/Image_1.jpg").convert("RGB")
input_tensor = transform(img).unsqueeze(0).numpy()  # Add batch dimension

# Load ONNX model
session = ort.InferenceSession("/workspace/E2EMLOps/s3_files/vegfruits.onnx")
input_name = session.get_inputs()[0].name

# Run inference
outputs = session.run(None, {input_name: input_tensor})

# Print output
print("Output:", outputs[0])
