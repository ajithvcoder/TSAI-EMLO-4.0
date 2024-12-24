from fastapi.testclient import TestClient
from app import app  # Replace with the path to your FastAPI app
from pathlib import Path

client = TestClient(app)

def test_root():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": True}

def test_predict_image():

    test_files = [
        ("examples/cat.jpg", "image/jpeg"),
        ("examples/dog.jpg", "image/jpeg"),
    ]

    for image_path, content_type in test_files:
        with open(image_path, "rb") as f:
            files = {"file": (image_path, f, content_type)}
            response = client.post("/classify", files=files)

            # Assertions
            assert response.status_code == 200
            response_data = response.json()
            print(response_data)
            assert "predictions" in response_data
            assert Path(image_path).stem.capitalize() in response_data["predictions"]
            print(f"Predictions for {image_path}: {response_data['predictions']}")