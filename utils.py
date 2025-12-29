import io
from torchvision import transforms
from PIL import Image
import torch

# Define constants
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def get_transforms():
    """Returns the validation/inference transform pipeline."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

def process_image(image_bytes):
    """Reads bytes, converts to RGB, applies transforms."""
    transform = get_transforms()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return transform(image).unsqueeze(0) # Add batch dimension [1, 3, 224, 224]

def interpret_score(score):
    """
    Interprets the continuous Mayo score (Task 3).
    Logic derived from notebook.
    """
    score = float(score)
    rounded_class = round(score)
    diff = score - rounded_class
    
    zone = "Solid Classification"
    if diff < -0.10: 
        zone = "Mild / Improving Side"
    elif diff > 0.10: 
        zone = "Severe / Worsening Side"

    return {
        "continuous_score": round(score, 2),
        "nearest_mayo_class": int(rounded_class),
        "clinical_status": zone,
        "explanation": f"Score is {round(score,2)}, which is on the {zone} of Mayo {int(rounded_class)}"
    }