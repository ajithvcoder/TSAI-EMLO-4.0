import base64
import concurrent.futures
import time
import numpy as np
import requests
import torch
import timm
import matplotlib.pyplot as plt
from urllib.request import urlopen
from PIL import Image
import psutil
try:
    import gpustat
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.cat_dog_classifier import CatDogClassifier
from thop import profile

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CatDogClassifier( \
        base_model='convnext_tiny', pretrained=True, num_classes=2)
    # self.model.load_from_checkpoint(checkpoint_path="epoch-checkpoint_patch_size-8_embed_dim-128.ckpt")
    model = model.to(device)
    model.eval()
    batch_size = 1
    # Create random input data
    x = torch.randn(batch_size, 3, 224, 224).to(device)
    # model = SimpleModel()
    # input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3-channel, 32x32 image

    # Measure FLOPs and Params
    flops, params = profile(model, inputs=(x,))
    print(f"FLOPs: {flops}")
    print(f"Parameters: {params}")