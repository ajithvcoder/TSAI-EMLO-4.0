import requests
from urllib.request import urlopen
import base64
import rootutils
from io import BytesIO

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from PIL import Image

def test_single_image():
    # Get test image
    url = 'https://raw.githubusercontent.com/ajithvcoder/emlo4-session-09-ajithvcoder/refs/heads/main/assets/cat_for_test.jpg'
    img_data = urlopen(url).read()
    
    # img_path = "data/cats_and_dogs_filtered/test/cats/cat.2149.jpg"
    # img = Image.open(img_path)
    # # Convert the image to a bytes-like object
    # buffer = BytesIO()
    # img.save(buffer, format="JPEG")  # Save as JPEG or the correct format
    # img_bytes = buffer.getvalue()
    
    # Convert to base64 string
    img_bytes = base64.b64encode(img_data).decode('utf-8')
    
    # Send request
    response = requests.post(
        "http://localhost:8000/predict",
        json={"image": img_bytes}  # Send as JSON instead of files
    )
    
    if response.status_code == 200:
        predictions = response.json()["predictions"]
        print("Top 5 Predictions:")
        for pred in predictions:
            print(f"{pred['label']}: {pred['probability']:.2%}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_single_image()