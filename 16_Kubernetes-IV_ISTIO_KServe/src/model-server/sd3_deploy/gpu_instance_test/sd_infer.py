#!/usr/bin/env python3

import torch
from diffusers import StableDiffusionPipeline #,StableDiffusion3Pipeline
import os
import time
import numpy as np
from PIL import Image

def main():
    print("Downloading sd-small model...")

    # Create output directory
    os.makedirs("sd-small-model", exist_ok=True)

    # Download and save model
    # pipe = StableDiffusion3Pipeline.from_pretrained(
    #     "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.bfloat16
    # )
    
    # pipe = StableDiffusion3Pipeline.from_pretrained(
    #     "OFA-Sys/small-stable-diffusion-v0 model", torch_dtype=torch.bfloat16
    # )
    inputs = ["a photo of an astronaut riding a horse"]
    # inputs = torch.tensor(inputs).to("cuda")
    # "stabilityai/stable-diffusion-3-medium-diffusers",
    pipe = StableDiffusionPipeline.from_pretrained(
        "OFA-Sys/small-stable-diffusion-v0",
        safety_checker=None,
        torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda") 

    print("Saving model to disk...")
    # pipe.save_pretrained("./sd-small-model")
    print("Model downloaded and saved successfully!")

    # logger.info(f"Starting inference with {len(inputs)} inputs")
    print(f"Starting inference with {len(inputs)} inputs")
    start_time = time.time()


    inferences = pipe(
        inputs,
        num_inference_steps=28,
        guidance_scale=7.0,
        width=256,
        height=256,
    ).images

    if torch.cuda.is_available():
        print(f"GPU Memory after inference: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        
    # Assuming `inferences` is a list of PIL Images
    images = []

    # Collect the images (if further processing is needed)
    for image in inferences:
        images.append(image)  # Directly store PIL Images

    # Save the first image in the output path
    output_path = "output_1.png"
    print(f"Saving image to {output_path}")
    images[0].save(output_path)  # Directly save the PIL Image


if __name__ == "__main__":
    main()