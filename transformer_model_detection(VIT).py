import os
import csv
import collections
from datetime import datetime
from typing import List

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# Paths
csv_path = "emotion_changes.csv"
percentage_file = "emotion_percentages.csv"

# Ensure CSV file exists
if not os.path.exists(csv_path):
    df = pd.DataFrame(columns=["Timestamp", "Emotion"])
    df.to_csv(csv_path, index=False)

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

# Track last emotion to avoid logging duplicates
last_emotion = None

# Function to calculate and save emotion percentages
def calculate_and_save_percentages():
    if not os.path.exists(csv_path):
        print("No emotion data file found.")
        return
    
    df = pd.read_csv(csv_path)

    if df.empty or "Emotion" not in df.columns:
        print("No emotion data to calculate percentages.")
        return

    emotion_counts = df["Emotion"].value_counts()
    total = emotion_counts.sum()

    if total == 0:
        print("No emotion entries to calculate.")
        return

    emotion_percentages = (emotion_counts / total * 100).round(2)

    # Save to CSV
    perc_df = pd.DataFrame(list(emotion_percentages.items()), columns=["Emotion", "Percentage"])
    perc_df.to_csv(percentage_file, index=False)

    print("✅ Emotion percentages saved successfully.")

# API endpoint: Predict Emotion
@app.post("/predict")
async def predict_emotion(data: LandmarkData):
    global last_emotion  # Allow modification of the global last_emotion

    landmarks = np.array(data.landmarks).astype(np.float32)
    if landmarks.shape != (468, 3):
        return {"error": "Invalid landmarks shape"}

    input_tensor = torch.tensor(landmarks).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        emotion = emotion_labels.get(predicted.item(), "Unknown")

    # Only log if emotion has changed
    if emotion != last_emotion:
        last_emotion = emotion
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Append new row
        new_row = pd.DataFrame({"Timestamp": [timestamp], "Emotion": [emotion]})
        new_row.to_csv(csv_path, mode='a', header=False, index=False)

    return {"predicted_emotion": emotion}

# API endpoint: Calculate percentages manually if needed
@app.post("/calculate_percentages")
async def calculate_percentages():
    calculate_and_save_percentages()
    return {"message": "Emotion percentages calculated and saved."}

# Shutdown event to save percentages on server stop
@app.on_event("shutdown")
def shutdown_event():
    print("🚪 Server is shutting down... calculating final emotion percentages.")
    calculate_and_save_percentages()

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
