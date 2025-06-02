import io
import base64
from typing import Annotated
import numpy as np
import onnxruntime as ort
from PIL import Image
from fastapi import FastAPI, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from fasthtml import FastHTML
from fasthtml.common import (
    Html, Head,Script, Title, Body, Div, Form, Input, Img, P, to_xml
)
from shad4fast import (
    ShadHead,
    Card,
    CardHeader,
    CardTitle,
    CardDescription,
    CardContent,
    CardFooter,
    Alert,
    AlertTitle,
    AlertDescription,
    Button,
    Badge,
    Separator,
    Lucide,
    Progress,
)
import uvicorn
import socket

# Application Configuration
INPUT_SIZE = (224, 224)
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
LABELS = ["Cat", "Dog"]
# Get hostname
hostname = socket.gethostname()

class ModelInference:
    def __init__(self, model_path):
        """
        Initialize ONNX model inference session
        
        Args:
            model_path (str): Path to the ONNX model file
        """
        try:
            print("Loading ONNX model...")
            self.session = ort.InferenceSession(model_path)
            
            # Warm-up inference to ensure model is ready
            self.session.run(
                ["output"], 
                {"input": np.random.randn(1, 3, *INPUT_SIZE).astype(np.float32)}
            )
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess image for model inference
        
        Args:
            image (PIL.Image): Input image
        
        Returns:
            np.ndarray: Preprocessed image array
        """
        # Convert to RGB and resize
        image = image.convert("RGB").resize(INPUT_SIZE)
        
        # Convert to numpy array and normalize
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # Apply mean and std normalization
        img_array = (img_array - MEAN) / STD
        
        # Transpose to channel-first format
        img_array = img_array.transpose(2, 0, 1)
        
        # Add batch dimension
        return np.expand_dims(img_array, 0)

    def predict(self, image: Image.Image) -> dict:
        """
        Perform model inference
        
        Args:
            image (PIL.Image): Input image
        
        Returns:
            dict: Prediction probabilities
        """
        processed_image = self.preprocess_image(image)
        
        outputs = self.session.run(
            ["output"], 
            {"input": processed_image.astype(np.float32)}
        )
        
        # Process logits to probabilities
        logits = outputs[0][0]
        probabilities = np.exp(logits) / np.sum(np.exp(logits))
        
        return {LABELS[i]: float(prob) for i, prob in enumerate(probabilities)}

# Create application instances
app = FastAPI(
    title=f"Cat vs Dog Classifier - Pod: {hostname}",
    description="AI-powered image classification service",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionResponse(BaseModel):
    """Response model for predictions"""

    predictions: dict  # Change to dict for class probabilities
    success: bool
    message: str

# Initialize model
try:
    model = ModelInference("mambaout_model.onnx")
    print("Model laoded")
except Exception as e:
    print(f"Fatal error: {e}")
    model = None

# FastAPI routes
@app.get("/", response_class=HTMLResponse)
async def ui_home():
    content = Html(
        Head(
            Title(f"Cat vs Dog Classifier - Pod: {hostname}"),
            ShadHead(tw_cdn=True, theme_handle=True),
            Script(
                src="https://unpkg.com/htmx.org@2.0.3",
                integrity="sha384-0895/pl2MU10Hqc6jd4RvrthNlDiE9U1tWmX7WRESftEDRosgxNsQG/Ze9YMRzHq",
                crossorigin="anonymous",
            ),
        ),
        Body(
            Div(
                Card(
                    CardHeader(
                        Div(
                            CardTitle("Cat vs Dog Classifier  🐱 🐶", cls="text-3xl font-extrabold text-indigo-600"),
                            Badge("AI Powered", variant="secondary", cls="bg-indigo-100 text-indigo-800 px-3 py-1 rounded-full text-sm"),
                            cls="flex items-center justify-between",
                        ),
                        CardDescription(
                            f"Upload an image to classify whether it's a cat or a dog. - Pod: {hostname} !",
                            cls="text-lg text-gray-600"
                        ),
                    ),
                    CardContent(
                        Form(
                            Div(
                                Div(
                                    Input(
                                        type="file",
                                        name="file",
                                        accept="image/*",
                                        required=True,
                                        cls="mb-4 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-600 file:text-white hover:file:bg-blue-700 file:cursor-pointer",
                                    ),
                                    P(
                                        "Drag and drop an image or click to browse",
                                        cls="text-sm text-gray-500 text-center mt-2 font-medium",
                                    ),
                                    cls="border-2 border-dashed rounded-lg p-4 hover:border-indigo-400 transition-colors",
                                ),
                                Button(
                                    Lucide("sparkles", cls="mr-2 h-4 w-4"),
                                    "Predict Image",
                                    type="submit",
                                    cls="w-full bg-indigo-600 text-white py-2 px-4 rounded-md text-lg font-semibold hover:bg-indigo-700 transition-all",
                                ),
                                cls="space-y-4",
                            ),
                            enctype="multipart/form-data",
                            hx_post="/predict",
                            hx_target="#result",
                        ),
                        Div(id="result", cls="mt-6"),
                    ),
                    cls="w-full max-w-3xl shadow-lg bg-white rounded-lg",
                    standard=True,
                ),
                cls="container flex items-center justify-center min-h-screen p-4 bg-gradient-to-br from-gray-100 to-indigo-100",
            ),
            cls="bg-gray-50 text-gray-900",
        ),
    )
    return to_xml(content)

@app.post("/predict", response_class=HTMLResponse)
async def predict_ui(file: Annotated[bytes, File()]):
    """
    Handle image prediction and render results
    """
    try:
        # Open image and predict
        image = Image.open(io.BytesIO(file))
        predictions = model.predict(image)
        
        # Prepare base64 image for display
        image_b64 = base64.b64encode(file).decode('utf-8')
        
        # Determine top prediction
        top_class = max(predictions, key=predictions.get)
        confidence = predictions[top_class]
        
        # Create results display
        results = Div(
            Div(
                Img(
                    src=f"data:image/jpeg;base64,{image_b64}", 
                    alt="Uploaded Image",
                    cls="w-full rounded-lg mb-4 shadow-md"
                ),
                Badge(
                    f"Prediction: {top_class} (Confidence: {confidence:.1%})",
                    variant="outline",
                    cls="w-full text-center text-lg bg-green-100 text-green-800 rounded-md px-4 py-2",
                ),
                Progress(value=int(confidence * 100), cls="mt-2 h-2 bg-indigo-200"),
                cls="space-y-4"
            ),
            cls="animate-in fade-in duration-500"
        )
        
        return to_xml(results)
    
    except Exception as e:
        return to_xml(Div(f"Error : {str(e)}", cls="text-red-500 text-lg font-bold"))


@app.post("/classify", response_model=PredictionResponse)
async def classify(file: Annotated[bytes, File(description="Image file to classify")]):
    try:
        image = Image.open(io.BytesIO(file))

        predictions = model.predict(image)
        
        # Prepare base64 image for display
        image_b64 = base64.b64encode(file).decode('utf-8')
        
        # Determine top prediction
        top_class = max(predictions, key=predictions.get)
        confidence = predictions[top_class]

        return PredictionResponse(
            predictions={top_class: confidence}, success=True, message="Classification successful"
        )

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Basic health check endpoint
    """
    return JSONResponse(
        content={"status": "healthy", "model_loaded":  model is not None }, status_code=200
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)