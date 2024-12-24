import requests
import os
from pathlib import Path

def predict_image(image_path: str, api_url: str = "http://localhost:8000/classify") -> dict:
    """
    Send an image to the prediction API and get results
    
    Args:
        image_path (str): Path to the image file
        api_url (str): URL of the prediction endpoint
        
    Returns:
        dict: Prediction results
    """
    # Check if file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Prepare the file for upload
    with open(image_path, "rb") as f:
        files = {"file": (Path(image_path).name, f, "image/jpeg")}
        
        # Make the request
        response = requests.post(api_url, files=files)
        print(response.text)
        # Check if request was successful
        response.raise_for_status()
        
        return response.json()

def main():
    # Test images
    test_images = ["examples/cat.jpg", "examples/dog.jpg"]
    
    print("Testing Cat/Dog Classification API")
    print("-" * 30)
    
    for image_path in test_images:
        try:
            print(f"\nTesting image: {image_path}")
            result = predict_image(image_path)
            
            # Print predictions
            print("Predictions:")
            for class_name, probability in result["predictions"].items():
                print(f"{class_name}: {probability:.4f}")
                
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")

if __name__ == "__main__":
    main()