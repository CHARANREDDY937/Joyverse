import os
import logging
from typing import List
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the Transformer Model
class FaceEmotionTransformer(nn.Module):
    def __init__(self, input_dim=3, seq_length=468, num_classes=7, embed_dim=128, num_heads=8, num_layers=4):
        super(FaceEmotionTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dropout=0.1, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.fc(x)

# Pydantic Models
class LandmarkData(BaseModel):
    landmarks: List[List[float]]

class EmotionData(BaseModel):
    emotion: str

# Emotion labels mapping
emotion_labels = {
    0: "Anger",
    1: "Disgust",
    2: "Fear",
    3: "Happiness",
    4: "Sadness",
    5: "Surprise",
    6: "Neutral",
}

# Setup device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
MODEL_PATH = "Emotion_model2000.pth"  # Ensure this file exists in the correct directory

def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found at {MODEL_PATH}")
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    
    try:
        logger.info("Loading model...")
        model = FaceEmotionTransformer(num_classes=7).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Model load failed: {str(e)}", exc_info=True)
        raise e
    return model

# Lifespan context: Load model at startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the model
    load_model()
    yield
    # Shutdown
    logger.info("Server is shutting down...")

# Initialize FastAPI app
app = FastAPI(title="Joyverse FastAPI Backend", lifespan=lifespan)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://joyverse.onrender.com", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Joyverse FastAPI Backend is running"}

# Health check endpoint
@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}

# API endpoint: Predict Emotion
@app.post("/predict")
async def predict_emotion(data: LandmarkData):
    try:
        # Log received data for debugging
        logger.info("Received predict request with landmarks shape: %s", np.array(data.landmarks).shape)
        
        # Validate input data
        if not data.landmarks:
            logger.error("Landmarks data is empty")
            raise HTTPException(status_code=400, detail="Landmarks data is empty")
        
        landmarks = np.array(data.landmarks, dtype=np.float32)
        if landmarks.shape != (468, 3):
            logger.error("Invalid landmarks shape: %s, expected (468, 3)", landmarks.shape)
            raise HTTPException(status_code=400, detail=f"Invalid landmarks shape: {landmarks.shape}, expected (468, 3)")
        
        # Ensure model is loaded
        if model is None:
            logger.error("Model not loaded")
            raise HTTPException(status_code=500, detail="Model not loaded")

        # Prepare input tensor
        input_tensor = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0).to(device)
        logger.info("Input tensor shape: %s", input_tensor.shape)
        
        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            logger.info("Model output shape: %s", output.shape)
            _, predicted = torch.max(output, 1)
            emotion = emotion_labels.get(predicted.item(), "Unknown")

        logger.info("Predicted emotion: %s", emotion)
        return {"predicted_emotion": emotion}
    except ValueError as ve:
        logger.error("Input validation error: %s", str(ve), exc_info=True)
        raise HTTPException(status_code=400, detail=f"Invalid input data: {str(ve)}")
    except Exception as e:
        logger.error("Prediction failed: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# API endpoint: Game Next Level
@app.post("/api/game/next-level")
async def next_level(data: EmotionData):
    try:
        emotion = data.emotion
        level_increment = 1 if emotion in ["Happiness", "Surprise"] else 0
        logger.info("Received emotion for next level: %s, Increment: %d", emotion, level_increment)
        return {
            "status": "success",
            "emotion": emotion,
            "level_increment": level_increment
        }
    except Exception as e:
        logger.error("Next level computation failed: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Next level computation failed: {str(e)}")

# Run the server
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)