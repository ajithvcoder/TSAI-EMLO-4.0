import torch
import timm
from PIL import Image
import io
import litserve as ls
import base64


import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.cat_dog_classifier import CatDogClassifier

class ImageClassifierAPI(ls.LitAPI):
    def setup(self, device):
        """Initialize the model and necessary components"""
        self.device = device
        # Create model and move to appropriate device

        # Use custom-trained model from scratch
        # self.model = CatDogClassifier.load_from_checkpoint(checkpoint_path="model_storage/epoch-checkpoint_patch_size-8_embed_dim-128.ckpt", \
        #     base_model='convnext_tiny', pretrained=False,patch_size=8, \
        #     embed_dim=128, num_classes=2, dims="(16, 32, 64, 128)", depths="(2, 2, 4, 2)")

        # Use pre-trained model
        self.model = CatDogClassifier( \
            base_model='convnext_tiny', pretrained=True, num_classes=2)
        self.model = self.model.to(device)
        self.model.eval()

        # Get model specific transforms
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)

        # Load ImageNet labels
        import requests
        url = '<https://storage.googleapis.com/bit_models/ilsvrc2012_wordnet_lemmas.txt>'
        # self.labels = requests.get(url).text.strip().split('\\n')
        self.labels = {0: "Cats", 1: "Dogs"}

    def decode_request(self, request):
        """Convert base64 encoded image to tensor"""
        image_bytes = request.get("image")
        if not image_bytes:
            raise ValueError("No image data provided")
        
        # Decode base64 string to bytes
        img_bytes = base64.b64decode(image_bytes)
        
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(img_bytes))
        # Convert to tensor and move to device
        tensor = self.transforms(image).unsqueeze(0).to(self.device)
        return tensor

    @torch.no_grad()
    def predict(self, x):
        outputs = self.model(x)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        return probabilities

    def encode_response(self, output):
        """Convert model output to API response"""
        # Get top 5 predictions
        probs, indices = torch.topk(output[0], k=1)
        
        return {
            "predictions": [
                {
                    "label": self.labels[idx.item()],
                    "probability": prob.item()
                }
                for prob, idx in zip(probs, indices)
            ]
        }

if __name__ == "__main__":
    api = ImageClassifierAPI()
    # Configure server with batching
    server = ls.LitServer(
        api,
        accelerator="gpu",
    )
    server.run(port=8000)