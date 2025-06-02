#!/usr/bin/env python3

import torch
from diffusers import StableDiffusionPipeline #,StableDiffusion3Pipeline
import os

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

    pipe = StableDiffusionPipeline.from_pretrained(
        # "stabilityai/stable-diffusion-3-medium-diffusers",
        "OFA-Sys/small-stable-diffusion-v0",
        torch_dtype=torch.bfloat16
    )
    pipe = pipe.to("cuda")
    print("Saving model to disk...")
    pipe.save_pretrained("./sd-small-model")
    print("Model downloaded and saved successfully!")

if __name__ == "__main__":
    main()