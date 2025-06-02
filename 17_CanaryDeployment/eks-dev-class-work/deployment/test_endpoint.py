import requests

def test_classifier():
    # Path to the image file
    image_path = 'cat_class.jpg'
    
    # URL for the prediction endpoint
    url = 'http://localhost:8085/predictions/cat-classifier'
    
    # Read the image file in binary mode
    with open(image_path, 'rb') as image_file:
        # Set the headers to specify content type
        headers = {'Content-Type': 'image/jpeg'}
        
        # Make the POST request
        response = requests.post(url, headers=headers, data=image_file)
        
        # Check if request was successful
        if response.status_code == 200:
            print("Prediction successful!")
            print("Response:", response.json())
        else:
            print(f"Error: Status code {response.status_code}")
            print("Response:", response.text)

if __name__ == "__main__":
    test_classifier()