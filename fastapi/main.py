import os
import csv
from datetime import datetime
from typing import List
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import logging

# Define the Transformer Model
class FaceEmotionTransformer(nn.Module):
    def __init__(self, input_dim=3, seq_length=468, num_classes=7, embed_dim=128, num_heads=8, num_layers=4):
        super(FaceEmotionTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch, embed_dim)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # Global average pooling
        return self.fc(x)

# Pydantic Model for incoming landmark data
class LandmarkData(BaseModel):
    landmarks: List[List[float]]

csv_file = "emotions_log.csv"
logging_enabled = True  # Flag to control emotion logging

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

# Setup device and load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FaceEmotionTransformer(num_classes=7).to(device)
try:
    print("🔄 Loading model...")
    model.load_state_dict(torch.load("Emotion_model2000.pth", map_location=device))
    model.eval()
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Model load failed:", e)

# Prepare CSV file if it doesn't exist
if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "emotion"])

# Lifespan context
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Shutdown logic
    print("🚪 Server is shutting down...")

# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Restrict to frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("emotion_percentages")

# API endpoint: Predict Emotion
@app.post("/predict")
async def predict_emotion(data: LandmarkData):
    global logging_enabled
    if not logging_enabled:
        logger.debug("Emotion logging is paused")
        return {"status": "paused", "message": "Emotion logging is currently paused"}

    landmarks = np.array(data.landmarks).astype(np.float32)
    if landmarks.shape != (468, 3):
        logger.error("Invalid landmarks shape")
        return {"error": "Invalid landmarks shape"}

    input_tensor = torch.tensor(landmarks).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        emotion = emotion_labels.get(predicted.item(), "Unknown")

    # Log the prediction
    with open(csv_file, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now().isoformat(), emotion])

    logger.debug(f"Emotion predicted: {emotion}")
    return {"predicted_emotion": emotion}

# API endpoint: Get overall emotion percentages (for potential future use)
@app.get("/emotion_percentages")
async def get_emotion_percentages():
    if not os.path.exists(csv_file):
        logger.error("Emotion log file not found")
        return {"error": "Emotion log file not found"}

    try:
        # Load CSV using pandas
        df = pd.read_csv(csv_file, names=["timestamp", "emotion"], skiprows=1)
        logger.debug(f"CSV loaded with {len(df)} rows")
        
        # Filter out invalid emotions
        valid_emotions = list(emotion_labels.values())
        df = df[df['emotion'].isin(valid_emotions)]
        logger.debug(f"Filtered to {len(df)} valid emotion rows")

        # Count emotions
        emotion_counts = df['emotion'].value_counts()
        total_predictions = len(df)

        if total_predictions == 0:
            logger.warning("No valid emotion data available")
            return {"error": "No valid emotion data available"}

        # Calculate percentages
        emotion_percentages = (emotion_counts / total_predictions) * 100
        percentages = {emotion: f"{percent:.2f}%" for emotion, percent in emotion_percentages.to_dict().items()}
        logger.debug(f"Emotion percentages calculated: {percentages}")
        return {"emotion_percentages": percentages}
    except Exception as e:
        logger.error(f"Error reading emotion log file: {e}")
        return {"error": f"Error reading emotion log file: {str(e)}"}

# API endpoint: Append emotion percentages to CSV
@app.post("/append_emotion_percentages")
async def append_emotion_percentages():
    if not os.path.exists(csv_file):
        logger.error("Emotion log file not found")
        return {"error": "Emotion log file not found"}

    try:
        # Load CSV using pandas
        df = pd.read_csv(csv_file, names=["timestamp", "emotion"], skiprows=1)
        logger.debug(f"CSV loaded with {len(df)} rows")
        
        # Filter out invalid emotions
        valid_emotions = list(emotion_labels.values())
        df = df[df['emotion'].isin(valid_emotions)]
        logger.debug(f"Filtered to {len(df)} valid emotion rows")

        # Count emotions
        emotion_counts = df['emotion'].value_counts()
        total_predictions = len(df)

        if total_predictions == 0:
            logger.warning("No valid emotion data to append")
            return {"error": "No valid emotion data to append"}

        # Calculate percentages
        emotion_percentages = (emotion_counts / total_predictions) * 100
        percentages = {emotion: f"{percent:.2f}%" for emotion, percent in emotion_percentages.to_dict().items()}
        logger.debug(f"Appending percentages to CSV: {percentages}")

        # Append to CSV
        with open(csv_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([])  # Blank line
            writer.writerow(["Summary"])  # Header
            for emotion, percentage in percentages.items():
                writer.writerow([emotion, percentage])

        logger.debug("Emotion percentages appended to CSV")
        return {"status": "success", "message": "Emotion percentages appended to CSV"}
    except Exception as e:
        logger.error(f"Error appending emotion percentages: {e}")
        return {"error": f"Error appending emotion percentages: {str(e)}"}

# API endpoint: Clear emotions_log.csv file
@app.post("/clear_emotions_log")
async def clear_emotions_log():
    try:
        with open(csv_file, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "emotion"])  # Write header only
        logger.debug("Emotions log cleared")
        return {"status": "success", "message": "Emotions log cleared"}
    except Exception as e:
        logger.error(f"Failed to clear emotions log: {e}")
        return {"status": "error", "message": str(e)}

# API endpoint: Pause emotion logging
@app.post("/pause_emotion_logging")
async def pause_emotion_logging():
    global logging_enabled
    logging_enabled = False
    logger.debug("Emotion logging paused")
    return {"status": "success", "message": "Emotion logging paused"}

# API endpoint: Resume emotion logging
@app.post("/resume_emotion_logging")
async def resume_emotion_logging():
    global logging_enabled
    logging_enabled = True
    logger.debug("Emotion logging resumed")
    return {"status": "success", "message": "Emotion logging resumed"}

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)