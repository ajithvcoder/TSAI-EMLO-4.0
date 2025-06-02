import os
import torch
import base64
import logging
from typing import List, Dict, Any
from ts.torch_handler.base_handler import BaseHandler
from ts.context import Context
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
from io import BytesIO

logger = logging.getLogger(__name__)
logger.info(f"Loading {__name__} handler...")

class ImageClassificationHandler(BaseHandler):
    """
    TorchServe handler class for image classification using a Transformers-based model.
    """

    def __init__(self):
        super().__init__()
        self.initialized = False
        self.model = None
        self.processor = None

    def initialize(self, ctx: Context):
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")

        logger.info(f'properties.get("gpu_id")={properties.get("gpu_id")}')
        logger.info(f"torch.cuda.is_available()={torch.cuda.is_available()}")

        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )

        logger.info(f"self.device={self.device}")

        # Load pre-trained model and processor
        self.model = AutoModelForImageClassification.from_pretrained(model_dir + "/model").to(self.device)
        self.processor = AutoImageProcessor.from_pretrained(model_dir + "/processor", use_fast=True)

        self.initialized = True
        logger.info("Model and Processor Loaded Successfully!")

    def preprocess(self, data: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images = []

        for row in data:
            image = row.get("data") or row.get("body")
            if isinstance(image, (bytearray, bytes)):
                image = Image.open(BytesIO(image))
            elif isinstance(image, str):
                # Handle input images passed as base64 encoded strings
                image = Image.open(BytesIO(base64.b64decode(image)))
            else:
                raise ValueError("Unsupported image data type")
            
            # Convert to RGB if image is grayscale or has an alpha channel
            if image.mode != 'RGB':
                image = image.convert('RGB')

            images.append(image)

        # Use processor to convert images to PyTorch tensors
        inputs = self.processor(images=images, return_tensors="pt")

        return inputs.to(self.device)

    def inference(self, input_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.model(**input_batch)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)

        return probabilities

    def postprocess(self, probabilities: torch.Tensor) -> List[Dict[str, Any]]:
        results = []
        top1_indices = torch.argmax(probabilities, dim=-1)
        top1_probs = probabilities.gather(1, top1_indices.unsqueeze(-1)).squeeze(-1)

        for idx, prob in zip(top1_indices, top1_probs):
            class_label = self.model.config.id2label[idx.item()]
            results.append({"class": class_label, "probability": prob.item()})

        return results

    def handle(self, data: List[Dict[str, Any]], context: Context) -> List[Dict[str, Any]]:
        inputs = self.preprocess(data)
        probabilities = self.inference(inputs)
        return self.postprocess(probabilities)