import torch
import onnx
import torchvision.models as models

model_path = "mambout_model.pt"
device = "cpu"
model = model.to(device)  # Move the model to the device
model = torch.load(model_path)
model.eval()  # Set to evaluation mode

# Adjust the size according to your model's input dimensions
dummy_input = torch.randn(1, 3, 224, 224).to(device)  # Example for an image model

onnx_path = "cat_dog_classifier.onnx"
torch.onnx.export(
    model,                     # Model to export
    dummy_input,               # Dummy input tensor
    onnx_path,                 # Path to save the ONNX file
    export_params=True,        # Store model parameters in the ONNX file
    opset_version=18,          # ONNX opset version (adjust based on requirements)
    do_constant_folding=True,  # Optimize constant folding for inference
    input_names=['input'],     # Input names
    output_names=['output'],   # Output names
    dynamic_axes={             # Allow dynamic input and output dimensions
        'input': {0: 'batch_size'}, 
        'output': {0: 'batch_size'}
    }
)

print(f"Model has been converted to ONNX and saved at {onnx_path}")
