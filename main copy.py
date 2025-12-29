import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
from contextlib import asynccontextmanager
import os

# Import our custom modules (from the files created in Step 2 and 3)
from model_arch import LIMUCSytem, MayoRegressor
from utils import process_image, interpret_score

# --- CONFIGURATION ---
# Ensure your .pth files are in the same directory as this script
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
            # Load weights
            state_dict = torch.load(REGRESSOR_WEIGHTS_PATH, map_location=device)
            reg_model.load_state_dict(state_dict)
            reg_model.eval()
            models["regressor"] = reg_model
            print(f"✅ Regressor loaded successfully from {REGRESSOR_WEIGHTS_PATH}")
        else:
            print(f"❌ Error: {REGRESSOR_WEIGHTS_PATH} not found!")
            models["regressor"] = None
    except Exception as e:
        print(f"❌ Error loading Regressor: {e}")
        models["regressor"] = None

    # 2. Load Patient-Level MIL Model (Task 2)
    try:
        if os.path.exists(MIL_WEIGHTS_PATH):
            mil_model = LIMUCSytem(num_classes=4).to(device)
            # Load weights
            state_dict = torch.load(MIL_WEIGHTS_PATH, map_location=device)
            mil_model.load_state_dict(state_dict)
            mil_model.eval()
            models["mil"] = mil_model
            print(f"✅ MIL Model loaded successfully from {MIL_WEIGHTS_PATH}")
        else:
            print(f"❌ Error: {MIL_WEIGHTS_PATH} not found!")
            models["mil"] = None
    except Exception as e:
        print(f"❌ Error loading MIL Model: {e}")
        models["mil"] = None

    yield
    models.clear()
    print("🛑 Models unloaded")

app = FastAPI(title="LIMUC Ulcerative Colitis AI", version="1.0", lifespan=lifespan)

# --- ROUTES ---

@app.get("/")
def health_check():
    """Health check to verify API is running."""
    return {
        "status": "active", 
        "models_loaded": {k: (v is not None) for k, v in models.items()}
    }

@app.post("/predict/frame")
async def predict_frame(file: UploadFile = File(...)):
    """
    Task 3: Single Image Regression (The Gray Zone)
    """
    if models["regressor"] is None:
        raise HTTPException(status_code=503, detail="Regressor model is not loaded.")

    try:
        # Read and preprocess
        contents = await file.read()
        input_tensor = process_image(contents).to(device)
        
        # Inference
        with torch.no_grad():
            score = models["regressor"](input_tensor).item()
            
        # Interpret result (Mild side vs Severe side)
        return interpret_score(score)
    
    except Exception as e:
        print(f"Error during frame prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error processing image.")

@app.post("/predict/patient")
async def predict_patient(files: List[UploadFile] = File(...)):
    """
    Task 2: Patient Level Classification (MIL)
    Upload multiple images for one patient.
    """
    if models["mil"] is None:
        raise HTTPException(status_code=503, detail="MIL model is not loaded.")

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    try:
        # 1. Preprocess all images
        batch_tensors = []
        for file in files:
            content = await file.read()
            img_t = process_image(content).squeeze(0) # [3, 224, 224]
            batch_tensors.append(img_t)
        
        # Stack into a "Bag" -> [Num_Images, 3, 224, 224]
        # This tensor effectively acts as a batch for the CNN backbone
        bag_tensor = torch.stack(batch_tensors).to(device)
        
        # 2. Inference
        with torch.no_grad():
            # FIXED LINE: Removed .unsqueeze(0)
            # Input shape must be [Num_Images, 3, 224, 224]
            logits, attention_weights = models["mil"].forward_patient(bag_tensor)
            
            # Get probabilities
            probs = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            confidence = probs[0][prediction].item()
        
        return {
            "patient_diagnosis": f"Mayo {prediction}",
            "confidence": f"{confidence:.4f}",
            "num_frames_analyzed": len(files),
            "clinical_note": "Automated AI prediction. Please verify with endoscopy."
        }

    except Exception as e:
        print(f"Error during patient prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)