import json
import os
import sys
import traceback
from timeit import default_timer as timer

import gradio as gr
import torch
from PIL import Image
from torchvision import transforms
from dotenv import load_dotenv
import boto3

# --- Setup ---
load_dotenv()
print("Starting application with debug info...")
print(f"Python version: {sys.version}")
print(f"Torch version: {torch.__version__}")
print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    torch.set_default_device(device)
    print(f"Default device set to: {device}")
except Exception as e:
    print(f"Error setting default device: {e}")
    # Fall back to older method if needed
    if torch.__version__ < '2.0.0':
        print("Using older torch version method for device handling")

# --- Download from S3 (CPU models only) ---
def download_from_s3():
    print("Attempting to download artifacts from S3...")
    BUCKET_NAME = 'mybucket-emlo-mumbai'
    ARTIFACTS = [
        'kserve-ig/vegfruits-classifier-prod/pths/vegfruits_cpu.pt',
        'kserve-ig/sports-classifier-prod/pths/sports_cpu.pt',

        'kserve-ig/vegfruits-classifier-prod/index_to_name.json',
        'kserve-ig/sports-classifier-prod/index_to_name.json',
    ]
    os.makedirs("vegfruits", exist_ok=True)
    os.makedirs("sports", exist_ok=True)

    try:
        aws_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
        print(f"AWS credentials available: {bool(aws_key and aws_secret)}")
        
        s3 = boto3.client(
            "s3",
            aws_access_key_id=aws_key,
            aws_secret_access_key=aws_secret,
            region_name="ap-south-1"
        )

        for artifact in ARTIFACTS:
            if not os.path.exists(artifact):
                artifact_extract = artifact.split("/")[-1]
                if "vegfruits" in artifact:
                    local_name = "vegfruits"
                if "sports" in artifact:
                    local_name = "sports"
                s3.download_file(BUCKET_NAME, artifact, os.path.join(local_name, artifact_extract))
                print(f"Successfully downloaded {artifact} as {os.path.join(local_name, artifact_extract)}")
            else:
                print(f"{artifact} already exists, skipping download")
    except Exception as e:
        print(f"Error during S3 download: {e}")
        traceback.print_exc()

# --- Image Transform ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Load models ---
def load_model(name):
    print(f"Loading model: {name}")
    # mybucket-emlo-mumbai/kserve-ig/vegfruits-classifier-prod/pths/
    path = f"{name}/{name}_cpu.pt"
    try:
        if not os.path.exists(path):
            print(f"ERROR: Model file not found at {path}")
            return None
        
        model = torch.jit.load(path)
        print(f"Model loaded successfully from {path}")
        model.to(device)
        print(f"Model moved to {device}")
        model.eval()
        print(f"Model set to evaluation mode")
        return model
    except Exception as e:
        print(f"Error loading model {name}: {e}")
        traceback.print_exc()
        return None

# --- Load class mappings ---
def load_classnames(name):
    print(f"Loading class mappings for: {name}")
    file_path = f"{name}/index_to_name.json"
    try:
        if not os.path.exists(file_path):
            print(f"ERROR: Class mapping file not found at {file_path}")
            return {}
            
        with open(file_path) as f:
            mapping = json.load(f)
        print(f"Class mappings loaded successfully from {file_path}")
        return mapping
        # # Debug info
        # print(f"Raw mapping sample (first 3 items): {list(mapping.items())[:3]}")
        
        # # Convert keys to integers and create reverse mapping
        # try:
        #     idx2lbl = {int(v): k for k, v in mapping.items()}
        #     print(f"Converted mapping sample (first 3 items): {list(idx2lbl.items())[:3]}")
        #     return idx2lbl, mapping
        # except Exception as e:
        #     print(f"Error converting class mappings: {e}")
        #     # Fallback to string keys if int conversion fails
        #     return {v: k for k, v in mapping.items()}
    except Exception as e:
        print(f"Error loading class mappings for {name}: {e}")
        traceback.print_exc()
        return {}

# --- Predict functions ---
@torch.no_grad()
def predict(img, model, idx2lbl):
    print(f"Prediction request received. Input type: {type(img)}")
    
    # Handle non-image inputs
    if img is None:
        print("Received None image")
        return {"No image provided": 1.0}, 0.0
    
    if isinstance(img, bool):
        print(f"Received boolean input: {img}")
        return {"Boolean input received, expected image": 1.0}, 0.0
    
    # Verify we have a valid image
    if not isinstance(img, Image.Image):
        print(f"WARNING: Input is not a PIL Image but {type(img)}")
        try:
            if hasattr(img, 'convert'):
                print("Object has convert method, attempting to use as image")
            else:
                print("Object cannot be used as an image")
                return {"Invalid image format": 1.0}, 0.0
        except Exception as e:
            print(f"Error checking image: {e}")
            return {"Error processing input": 1.0}, 0.0
    
    try:
        print("Starting prediction process")
        start = timer()
        
        # Debug image properties
        print(f"Image size: {img.size if hasattr(img, 'size') else 'unknown'}")
        print(f"Image mode: {img.mode if hasattr(img, 'mode') else 'unknown'}")
        
        # Transform image
        print("Transforming image")
        img_tensor = transform(img).to(device)
        print(f"Image transformed to tensor of shape {img_tensor.shape}")
        
        # Run model
        print("Running model inference")
        logits = model(img_tensor.unsqueeze(0))
        print(f"Model output shape: {logits.shape}")
        
        # Process output
        print("Processing model output")
        probs = torch.softmax(logits, dim=-1)
        top5 = torch.topk(probs, min(5, probs.shape[1]))
        
        # Create predictions dictionary
        print("Creating predictions dictionary")
        preds = {}
        for i, (v, idx) in enumerate(zip(top5.values[0], top5.indices[0])):
            idx_item = idx.item()
            print(f"Processing top prediction {i+1}: idx={idx_item}, value={v.item():.4f}")
            
            if str(idx_item) in idx2lbl:
                print(f"inside predict - {idx_item}")
                label = idx2lbl[str(idx_item)]
                preds[label] = round(v.item(), 4)
                print(f"Mapped to label: {label}")
            else:
                print(f"WARNING: Index {idx_item} not found in class mapping")
                preds[f"Unknown-{idx_item}"] = round(v.item(), 4)
        
        elapsed = round(timer() - start, 4)
        print(f"Prediction completed in {elapsed}s")
        return preds, elapsed
    except Exception as e:
        print(f"Prediction error: {e}")
        traceback.print_exc()
        return {"Error": 0.0}, 0.0

# --- App logic ---
def main():
    print("Initializing application...")
    
    try:
        download_from_s3()
    except Exception as e:
        print(f"Error in S3 download: {e}")
        traceback.print_exc()
    
    print("Loading models and class mappings")
    smodel = load_model("sports")
    vfmodel = load_model("vegfruits")
    sports_map = load_classnames("sports")
    vegfruits_map = load_classnames("vegfruits")

    def sports_fn(img):
        print("\n--- Sports Classification Request ---")
        print(f"Input type: {type(img)}")
        if img is None:
            print("No image provided")
            return {"No image provided": 1.0}, 0.0
        if isinstance(img, bool):
            print(f"Received boolean: {img}")
            return {"Boolean received (expected image)": 1.0}, 0.0
        try:
            return predict(img, smodel, sports_map)
        except Exception as e:
            print(f"Error in sports_fn: {e}")
            traceback.print_exc()
            return {"Error in sports classifier": 1.0}, 0.0
    
    def veg_fn(img):
        print("\n--- VegFruits Classification Request ---")
        print(f"Input type: {type(img)}")
        if img is None:
            print("No image provided")
            return {"No image provided": 1.0}, 0.0
        if isinstance(img, bool):
            print(f"Received boolean: {img}")
            return {"Boolean received (expected image)": 1.0}, 0.0
        try:
            return predict(img, vfmodel, vegfruits_map)
        except Exception as e:
            print(f"Error in veg_fn: {e}")
            traceback.print_exc()
            return {"Error in vegfruits classifier": 1.0}, 0.0
    
    print("Creating Gradio interfaces")
    try:
        sports_interface = gr.Interface(
            fn=sports_fn,
            inputs=gr.Image(type="pil"),
            outputs=[
                gr.Label(num_top_classes=5),
                gr.Number(label="Prediction Time (s)")
            ],
            title="Sports Classifier",
            cache_examples=False
        )
        print("Sports interface created successfully")
        
        veg_interface = gr.Interface(
            fn=veg_fn,
            inputs=gr.Image(type="pil"),
            outputs=[
                gr.Label(num_top_classes=5),
                gr.Number(label="Prediction Time (s)")
            ],
            title="VegFruits Classifier",
            cache_examples=False
        )
        print("VegFruits interface created successfully")
        
        demo = gr.TabbedInterface(
            interface_list=[sports_interface, veg_interface],
            tab_names=["Sports", "VegFruits"]
        )
        print("TabbedInterface created successfully")
        
        print("Launching Gradio app...")
        demo.launch(share=True)
    except Exception as e:
        print(f"Error creating Gradio interface: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()