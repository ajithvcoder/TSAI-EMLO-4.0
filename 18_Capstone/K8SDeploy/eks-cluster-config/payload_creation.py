import base64
import json

# Path to the image you want to encode
image_path = "1.jpg"  # change this to your image path
output_json_path = "input_sports.json"

# Read image as bytes
with open(image_path, "rb") as image_file:
    image_bytes = image_file.read()

# Encode image to base64
image_b64 = base64.b64encode(image_bytes).decode("utf-8")

# Create payload
payload = {
    "instances": [
        {
            "data": image_b64
        }
    ]
}

# Write to JSON file
with open(output_json_path, "w") as json_file:
    json.dump(payload, json_file, indent=4)

print(f"JSON payload written to {output_json_path}")
