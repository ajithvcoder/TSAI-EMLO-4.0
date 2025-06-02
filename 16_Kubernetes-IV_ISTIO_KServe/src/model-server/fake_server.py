from fastapi import FastAPI
from io import BytesIO
from PIL import Image
import random
import os
import zlib
import json
import logging
import socket
import traceback
import base64
import json
import httpx
import numpy as np

from fastapi import FastAPI, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from typing import Annotated

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - WebServer - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Web Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add constants
HOSTNAME = socket.gethostname()

@app.post("/v1/models/sd_small:predict")
async def generate_image(payload: dict):
    # Extract text from the payload (for demonstration, we're not using it here)
    text = payload.get('instances', [{}])[0].get('data', '')

    # Simulate generating an image (for demonstration purposes, creating a random image)
    # img = Image.new('RGB', (256, 256), color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    img = Image.open('dog2.jpg')
    img.save("new_image.png")
    # Save the image to a BytesIO object to send as a response
    # img_byte_arr = BytesIO()
    # img.save(img_byte_arr, format='PNG')
    # img_byte_arr.seek(0)


    # Get the bytes from the BytesIO stream
    # image_bytes = img_byte_arr.getvalue()

    # # Return the image bytes as a response with the appropriate media type
    # return Response(content=image_bytes, media_type="image/png")

     # Encode the image bytes to Base64
    # encoded_image = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

    # # Return the encoded image as a plain text response
    # return {"text": encoded_image}
    # Convert the array into a list and return as JSON
    img_array = np.array(img)
    img_list = img_array.tolist()
    return img_list


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)