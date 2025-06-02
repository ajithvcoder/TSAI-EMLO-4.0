import contextlib
import json
import os
import platform
import socket

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

INPUT_SIZE = (224, 224)
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def get_classes(name:str):
    fname = os.path.join('checkpoints','classnames',f'{name}.json')
    if not os.path.isfile(fname):
        return []
    with open(fname) as class2idxfile:
        cls2lbs:dict = json.load(class2idxfile)
    return list(cls2lbs.keys())


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")                            # Convert to RGB if not already
    image = image.resize(INPUT_SIZE)                        # Resize
    img_array = np.array(image).astype(np.float32) / 255.0  # Convert to numpy array and normalize
    img_array = (img_array - MEAN) / STD                    # Apply mean and std normalization
    img_array = img_array.transpose(2, 0, 1)                # Transpose to channel-first format (NCHW)
    img_array = np.expand_dims(img_array, 0)                # Add batch dimension
    return img_array


class Sportprediction:
    def load_model(self)->None:
        self.session = ort.InferenceSession( os.path.join('checkpoints','onnxs','sports.onnx') )
        self.session_input_name = self.session.get_inputs()[0].name
        self.classes = get_classes('sports')

    def predict(self,image:Image.Image):
        img = preprocess_image(image=image)
        outputs = self.session.run(None,{self.session_input_name:img.astype(np.float32)})
        logits = outputs[0][0]
        probabilities = np.exp(logits) / np.sum(np.exp(logits))
        # predictions = {class_labels[i]: float(prob) for i, prob in enumerate(probabilities)}
        predicted_label = self.classes[np.argmax(probabilities)]
        confidence = np.max(probabilities)
        return {
            'confidence': float(confidence),
            'label':str(predicted_label)
        }


sports_prediction = Sportprediction()


@contextlib.asynccontextmanager
async def lifespan(app:FastAPI):
    sports_prediction.load_model()
    yield


app = FastAPI(title="Image Classification API",lifespan=lifespan,description="FastAPI application serving an ONNX model for image classification",    version="1.0.0",)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def home():
    return {'msg':'welcome to fastapi'}


@app.post("/sports",response_model=dict)
async def post_object_detection(image:UploadFile=File(...))->dict:
    image_objext = Image.open(image.file).convert("RGB")
    return sports_prediction.predict(image_objext)


@app.get("/health")
async def health_check():
    return JSONResponse(
        content={
            "socket_name": socket.getfqdn(),
            "hostname": socket.gethostname(),
            "architecture": platform.architecture()[0],
            "os": platform.system(),
            "os_release": platform.release(),
            "status": "healthy",
            "model_loaded": True
        }, status_code=200
    )


if __name__=='__main__':
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=9090)
