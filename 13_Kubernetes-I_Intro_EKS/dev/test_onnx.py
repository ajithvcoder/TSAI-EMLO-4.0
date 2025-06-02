import onnxruntime as ort
import numpy as np
from torchvision import transforms
from PIL import Image

# Step 1: Load the ONNX model
onnx_model_path = "cat_dog_classifier.onnx"
session = ort.InferenceSession(onnx_model_path)

# Step 2: Define preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to match model's input size
    transforms.ToTensor(),         # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet stats
])

# Step 3: Preprocess the test image
image_path = "dog.jpg"
image = Image.open(image_path).convert("RGB")  # Open and convert to RGB
input_tensor = transform(image).unsqueeze(0)   # Add batch dimension
input_tensor = input_tensor.numpy()            # Convert to NumPy array (required for ONNX)

# Step 4: Run inference
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
predictions = session.run([output_name], {input_name: input_tensor})

# Step 5: Interpret predictions
predicted_class = np.argmax(predictions[0])  # Get the class with the highest score
class_labels = ["Cat", "Dog"]  # Adjust based on your labels
print(f"Predicted class: {class_labels[predicted_class]}")
