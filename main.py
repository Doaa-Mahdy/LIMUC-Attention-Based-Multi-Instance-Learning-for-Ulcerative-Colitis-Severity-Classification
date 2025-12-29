import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
from contextlib import asynccontextmanager
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import numpy as np
import cv2
import base64
import os
import io 
from PIL import Image

# Import our custom modules
from model_arch import LIMUCSytem, MayoRegressor
from utils import process_image, interpret_score

# --- CONFIGURATION ---
MIL_WEIGHTS_PATH = "best_model_weights.pth"
REGRESSOR_WEIGHTS_PATH = "mayo_regressor_weights.pth"

# Global model dictionary
models = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- LIFESPAN (Load models on startup) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"🚀 Starting API on device: {device}")
    
    # 1. Load Image-Level Regressor (Task 3)
    try:
        if os.path.exists(REGRESSOR_WEIGHTS_PATH):
            reg_model = MayoRegressor().to(device)
            state_dict = torch.load(REGRESSOR_WEIGHTS_PATH, map_location=device)
            reg_model.load_state_dict(state_dict)
            reg_model.eval()
            models["regressor"] = reg_model
            print(f"✅ Regressor (Task 3) loaded successfully.")
        else:
            print(f"❌ Error: {REGRESSOR_WEIGHTS_PATH} not found!")
            models["regressor"] = None
    except Exception as e:
        print(f"❌ Error loading Regressor: {e}")
        models["regressor"] = None

    # 2. Load Patient-Level MIL Model (Task 1 & Task 2)
    try:
        if os.path.exists(MIL_WEIGHTS_PATH):
            # This model is used for BOTH Patient Classification AND Single Frame Classification
            mil_model = LIMUCSytem(num_classes=4).to(device)
            state_dict = torch.load(MIL_WEIGHTS_PATH, map_location=device)
            mil_model.load_state_dict(state_dict)
            mil_model.eval()
            models["mil"] = mil_model
            print(f"✅ MIL Model (Task 1 & 2) loaded successfully.")
        else:
            print(f"❌ Error: {MIL_WEIGHTS_PATH} not found!")
            models["mil"] = None
    except Exception as e:
        print(f"❌ Error loading MIL Model: {e}")
        models["mil"] = None

    yield
    models.clear()
    print("🛑 Models unloaded")

app = FastAPI(title="LIMUC Ulcerative Colitis AI", version="1.1", lifespan=lifespan)

# --- ROUTES ---

@app.get("/")
def health_check():
    return {
        "status": "active", 
        "models_loaded": {k: (v is not None) for k, v in models.items()}
    }

# --- TASK 1: Single Image Classification (New) ---
@app.post("/predict/classification")
async def predict_classification(file: UploadFile = File(...)):
    """
    Task 1: Single Image Classification (Mayo 0-3)
    Uses the MIL model (Task 2 model) as a 'Bag of 1' to classify a single frame.
    """
    if models["mil"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        # 1. Process image
        contents = await file.read()
        # process_image returns [1, 3, 224, 224]
        input_tensor = process_image(contents).to(device)
        
        # 2. Inference
        with torch.no_grad():
            # Pass single image. The Attention layer will handle it as a bag of 1.
            logits, _ = models["mil"].forward_patient(input_tensor)
            
            # Get probabilities
            probs = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            confidence = probs[0][prediction].item()

        return {
            "task": "Task 1 - Single Frame Classification",
            "predicted_class": f"Mayo {prediction}",
            "confidence": f"{confidence:.4f}",
            "probabilities": {
                "Mayo 0": f"{probs[0][0]:.4f}",
                "Mayo 1": f"{probs[0][1]:.4f}",
                "Mayo 2": f"{probs[0][2]:.4f}",
                "Mayo 3": f"{probs[0][3]:.4f}"
            }
        }

    except Exception as e:
        print(f"Error in classification: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- TASK 3: Regression (Gray Zone) ---
@app.post("/predict/regression")
async def predict_regression(file: UploadFile = File(...)):
    """
    Task 3: Single Image Regression (The Gray Zone)
    Renamed endpoint from /predict/frame to /predict/regression for clarity
    """
    if models["regressor"] is None:
        raise HTTPException(status_code=503, detail="Regressor model is not loaded.")

    try:
        contents = await file.read()
        input_tensor = process_image(contents).to(device)
        
        with torch.no_grad():
            score = models["regressor"](input_tensor).item()
            
        return interpret_score(score)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error.")

# --- TASK 2: Patient Level MIL ---
@app.post("/predict/patient")
async def predict_patient(files: List[UploadFile] = File(...)):
    """
    Task 2: Patient Level Classification (MIL)
    """
    if models["mil"] is None:
        raise HTTPException(status_code=503, detail="MIL model is not loaded.")

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    try:
        batch_tensors = []
        for file in files:
            content = await file.read()
            img_t = process_image(content).squeeze(0) # [3, 224, 224]
            batch_tensors.append(img_t)
        
        # Stack into Bag -> [Num_Images, 3, 224, 224]
        bag_tensor = torch.stack(batch_tensors).to(device)
        
        with torch.no_grad():
            logits, _ = models["mil"].forward_patient(bag_tensor)
            probs = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            confidence = probs[0][prediction].item()
        
        return {
            "task": "Task 2 - Patient Diagnosis",
            "patient_diagnosis": f"Mayo {prediction}",
            "confidence": f"{confidence:.4f}",
            "num_frames_analyzed": len(files)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- EXPLAINABILITY (Grad-CAM) ---
@app.post("/explain/classification")
async def explain_classification(file: UploadFile = File(...)):
    """
    Generates a Grad-CAM Heatmap for Task 1.
    """
    if models["mil"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        # 1. Read and Process Image
        contents = await file.read()
        
        # We need the original numpy image for visualization later
        original_img = Image.open(io.BytesIO(contents)).convert('RGB')
        original_img = np.array(original_img)
        original_img = cv2.resize(original_img, (224, 224))
        original_img = np.float32(original_img) / 255.0  # Normalize 0-1 for Grad-CAM lib
        
        # Prepare Tensor for Model
        input_tensor = process_image(contents).to(device)

        # 2. Define Target Layer for ConvNeXt
        # ConvNeXt usually has stages. We target the last block of the last stage.
        # Structure: model -> backbone -> stages -> [3] -> blocks -> [-1]
        target_layers = [models["mil"].backbone.stages[-1].blocks[-1]]

        # 3. Create GradCAM
        # We need a wrapper because our model returns (logits, attention), but GradCAM expects just logits
        class ModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x):
                logits, _ = self.model.forward_patient(x)
                return logits

        wrapped_model = ModelWrapper(models["mil"])
        cam = GradCAM(model=wrapped_model, target_layers=target_layers)

        # 4. Generate Heatmap
        # We target the class with the highest score (None automatically picks highest)
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)
        
        # 5. Overlay Heatmap on Image
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(original_img, grayscale_cam, use_rgb=True)
        
        # 6. Encode Image to Base64 to send to Frontend
        _, buffer = cv2.imencode('.jpg', visualization)
        img_str = base64.b64encode(buffer).decode('utf-8')

        return {"heatmap_base64": img_str}

    except Exception as e:
        print(f"Grad-CAM Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- EXPLAINABILITY: TASK 3 (Regression) ---
@app.post("/explain/regression")
async def explain_regression(file: UploadFile = File(...)):
    """
    Generates Grad-CAM for the Regression Model (Task 3).
    Fixed for latest Grad-CAM library versions.
    """
    if models["regressor"] is None:
        raise HTTPException(status_code=503, detail="Regressor model not loaded.")

    try:
        contents = await file.read()
        
        # Prepare for Visualization
        original_img = Image.open(io.BytesIO(contents)).convert('RGB')
        original_img = np.array(original_img)
        original_img = cv2.resize(original_img, (224, 224))
        original_img = np.float32(original_img) / 255.0

        # Prepare for Model
        input_tensor = process_image(contents).to(device) # [1, 3, 224, 224]

        # Target Layer
        target_layers = [models["regressor"].backbone.stages[-1].blocks[-1]]

        # Wrapper to handle Regression output
        class RegressorWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x):
                # FIX: Check if GradCAM added an extra dimension [1, 1, 3, 224, 224]
                if x.dim() == 5:
                    x = x.squeeze(0)
                return self.model(x)

        wrapped_model = RegressorWrapper(models["regressor"])
        
        # FIX: Removed 'use_cuda' argument (deprecated in new versions)
        cam = GradCAM(model=wrapped_model, target_layers=target_layers)

        # Generate Heatmap
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)
        
        # Overlay
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(original_img, grayscale_cam, use_rgb=True)
        
        # Encode
        _, buffer = cv2.imencode('.jpg', visualization)
        img_str = base64.b64encode(buffer).decode('utf-8')

        return {"heatmap_base64": img_str}

    except Exception as e:
        print(f"Regression Grad-CAM Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- EXPLAINABILITY: TASK 2 (Patient MIL) ---
@app.post("/explain/patient")
async def explain_patient(files: List[UploadFile] = File(...)):
    """
    Identifies most important frames and generates heatmaps.
    Fix: Removes incorrect unsqueeze causing 5D tensor crash.
    """
    if models["mil"] is None:
        raise HTTPException(status_code=503, detail="MIL model not loaded.")

    try:
        batch_tensors = []
        original_images = []
        
        # 1. Process Images
        for file in files:
            content = await file.read()
            # Tensor [3, 224, 224]
            img_t = process_image(content).squeeze(0) 
            batch_tensors.append(img_t)
            
            # Numpy for visualization
            img_pil = Image.open(io.BytesIO(content)).convert('RGB')
            img_np = np.array(img_pil)
            img_np = cv2.resize(img_np, (224, 224))
            img_np = np.float32(img_np) / 255.0
            original_images.append(img_np)

        # Bag Tensor: [N, 3, 224, 224]
        # This represents N images in the bag.
        bag_tensor = torch.stack(batch_tensors).to(device)

        # 2. Get Attention Weights
        # CRITICAL FIX: Do NOT use .unsqueeze(0) here. 
        # The model expects [Num_Images, 3, H, W], not [1, Num_Images, 3, H, W]
        with torch.no_grad():
            _, weights = models["mil"].forward_patient(bag_tensor)
            
            # Weights shape is usually [1, N] or [N, 1] depending on model definition
            # We flatten it to just [N]
            weights = weights.view(-1).cpu().numpy()

        # 3. Get Top 3 Indices
        num_images = len(files)
        top_k = min(3, num_images)
        top_indices = weights.argsort()[-top_k:][::-1]
        
        results = []

        # 4. Setup Grad-CAM
        target_layers = [models["mil"].backbone.stages[-1].blocks[-1]]
        
        # Wrapper to handle single frame classification
        class DirectClassifierWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, x):
                # Ensure input is 4D [Batch, 3, 224, 224]
                if x.dim() == 5: 
                    x = x.view(-1, 3, 224, 224)
                
                # Run Backbone -> Classifier directly
                feats = self.model.backbone(x)
                logits = self.model.mil_classifier(feats)
                return logits

        wrapped_model = DirectClassifierWrapper(models["mil"])
        cam = GradCAM(model=wrapped_model, target_layers=target_layers)

        for idx in top_indices:
            # Input tensor: [1, 3, 224, 224]
            input_tensor = bag_tensor[idx].unsqueeze(0)
            
            # Generate Heatmap
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)
            grayscale_cam = grayscale_cam[0, :]
            
            # Overlay
            vis = show_cam_on_image(original_images[idx], grayscale_cam, use_rgb=True)
            
            # Encode
            _, buffer = cv2.imencode('.jpg', vis)
            img_str = base64.b64encode(buffer).decode('utf-8')
            
            results.append({
                "frame_index": int(idx),
                "importance_score": float(weights[idx]),
                "heatmap_base64": img_str
            })

        return {"top_frames": results}

    except Exception as e:
        print(f"Patient Explain Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)