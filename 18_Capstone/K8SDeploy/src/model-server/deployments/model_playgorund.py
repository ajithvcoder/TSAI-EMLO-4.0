from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

processor = AutoImageProcessor.from_pretrained("/home/ubuntu/session-17/models/cat-classifier/processor", use_fast=True)
model = AutoModelForImageClassification.from_pretrained("/home/ubuntu/session-17/models/cat-classifier/model")

print(processor)
print(model)

image = Image.open("/home/ubuntu/session-17/cat.jpg")
inputs = processor(images=image, return_tensors="pt")

outputs = model(**inputs)
logits = outputs.logits

predicted_class_idx = logits.argmax(-1).item()
print(predicted_class_idx)