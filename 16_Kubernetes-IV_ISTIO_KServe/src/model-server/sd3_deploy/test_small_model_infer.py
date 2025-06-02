#!/usr/bin/env python3

import torch
from diffusers import StableDiffusion3Pipeline
import os
from PIL import Image

def main():
    print("Downloading SD3 model...")

    # Create output directory
    os.makedirs("sd3-model", exist_ok=True)

    # Download and save model
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.bfloat16
    )
    self.pipe = self.pipe.to("cuda")
    # print("Saving model to disk...")
    # pipe.save_pretrained("./sd3-model")
    # print("Model downloaded and saved successfully!")
    print("Torch presence: ", torch.cuda.is_available())
    print(f"Starting inference with {len(inputs)} inputs")
    start_time = time.time()

    inferences = self.pipe(
        inputs,
        num_inference_steps=28,
        guidance_scale=7.0,
        width=256,
        height=256,
    ).images

    end_time = time.time()
    print("Total time taken - ", end_time-start_time)

    if torch.cuda.is_available():
        print(
            f"GPU Memory after inference: {torch.cuda.memory_allocated()/1e9:.2f}GB"
        )
    
    # Check this because i did something like this to save the image after inference. its just saving as a pil file
    out_image = np.array(inferences[0])
    img = Image(out_image)
    img.save("new_image.png")

if __name__ == "__main__":
    main()
