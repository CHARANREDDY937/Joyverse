import os
import csv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from typing import List

class FaceEmotionTransformer(nn.Module):
    def __init__(self, input_dim=3, seq_length=468, num_classes=7, embed_dim=128, num_heads=8, num_layers=4):
        super(FaceEmotionTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)
        return self.fc(x)

class LandmarkData(BaseModel):
    landmarks: List[List[float]]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FaceEmotionTransformer(num_classes=7).to(device)
try:
    print("🔄 Loading model...")
    model.load_state_dict(torch.load("Emotion_model2000.pth", map_location=device))
    model.eval()
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Model load failed:", e)


csv_file = "emotions_log.csv"
if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "emotion"])

emotion_labels = {
    0: "Anger",
    1: "Disgust",
    2: "Fear",
    3: "Happiness",
    4: "Sadness",
    5: "Surprise",
    6: "Neutral",
}

@app.post("/predict")
async def predict_emotion(data: LandmarkData):
    landmarks = np.array(data.landmarks).astype(np.float32)
    if landmarks.shape != (468, 3):
        return {"error": "Invalid landmarks shape"}

    input_tensor = torch.tensor(landmarks).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        emotion = emotion_labels.get(predicted.item(), "Unknown")

    # Save the new prediction
    with open(csv_file, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now().isoformat(), emotion])

    # Read the entire CSV to calculate percentages
    emotion_counts = {label: 0 for label in emotion_labels.values()}
    total = 0

    with open(csv_file, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row["emotion"] in emotion_counts:
                emotion_counts[row["emotion"]] += 1
                total += 1

    if total > 0:
        emotion_percentages = {emotion: (count / total) * 100 for emotion, count in emotion_counts.items()}
    else:
        emotion_percentages = {emotion: 0.0 for emotion in emotion_counts}

    # Append the percentage summary at the end of the CSV
    with open(csv_file, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([])
        writer.writerow(["Emotion", "Percentage"])
        for emotion, percentage in emotion_percentages.items():
            writer.writerow([emotion, f"{percentage:.2f}%"])
        writer.writerow([])  # Add an empty line for separation

    return {
        "predicted_emotion": emotion,
        "percentages": emotion_percentages
    }
