# # import os
# # import csv
# # import collections
# # from datetime import datetime
# # from typing import List
# # import pandas as pd
# # import numpy as np
# # import torch
# # import torch.nn as nn
# # from fastapi import FastAPI
# # from fastapi.middleware.cors import CORSMiddleware
# # from pydantic import BaseModel
# # from contextlib import asynccontextmanager

# # # Define the Transformer Model
# # class FaceEmotionTransformer(nn.Module):
# #     def __init__(self, input_dim=3, seq_length=468, num_classes=7, embed_dim=128, num_heads=8, num_layers=4):
# #         super(FaceEmotionTransformer, self).__init__()
# #         self.embedding = nn.Linear(input_dim, embed_dim)
# #         encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=0.1)
# #         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
# #         self.fc = nn.Linear(embed_dim, num_classes)

# #     def forward(self, x):
# #         x = self.embedding(x)
# #         x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch, embed_dim)
# #         x = self.transformer_encoder(x)
# #         x = x.mean(dim=0)  # Global average pooling
# #         return self.fc(x)

# # # Pydantic Model for incoming landmark data
# # class LandmarkData(BaseModel):
# #     landmarks: List[List[float]]

# # csv_file = "emotions_log.csv"

# # # Emotion labels mapping
# # emotion_labels = {
# #     0: "Anger",
# #     1: "Disgust",
# #     2: "Fear",
# #     3: "Happiness",
# #     4: "Sadness",
# #     5: "Surprise",
# #     6: "Neutral",
# # }

# # # Setup device and load model
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # model = FaceEmotionTransformer(num_classes=7).to(device)
# # try:
# #     print("🔄 Loading model...")
# #     model.load_state_dict(torch.load("Emotion_model2000.pth", map_location=device))
# #     model.eval()
# #     print("✅ Model loaded successfully.")
# # except Exception as e:
# #     print("❌ Model load failed:", e)

# # # Prepare CSV file if it doesn't exist
# # if not os.path.exists(csv_file):
# #     with open(csv_file, "w", newline="") as file:
# #         writer = csv.writer(file)
# #         writer.writerow(["timestamp", "emotion"])

# # # Lifespan context
# # @asynccontextmanager
# # async def lifespan(app: FastAPI):
# #     yield
# #     # Shutdown logic here
# #     print("🚪 Server is shutting down... Calculating emotion percentages...")
# #     emotion_counts = collections.Counter()
# #     total = 0

# #     with open(csv_file, "r", newline="") as file:
# #         reader = csv.DictReader(file)
# #         for row in reader:
# #             if row.get('emotion') in emotion_labels.values():
# #                 emotion_counts[row["emotion"]] += 1
# #                 total += 1

# #     if total == 0:
# #         print("⚠️ No emotion data to calculate.")
# #         return

# #     # Write percentages into the same emotions_log.csv file
# #     with open(csv_file, "a", newline="") as file:
# #         writer = csv.writer(file)
# #         writer.writerow([])  # blank line
# #         writer.writerow(["Summary"])  # heading
# #         for emotion, count in emotion_counts.items():
# #             percentage = (count / total) * 100
# #             writer.writerow([emotion, f"{percentage:.2f}%"])

# #     print("✅ Emotion percentages appended to emotions_log.csv.")

# # # Initialize FastAPI app
# # app = FastAPI(lifespan=lifespan)

# # # Enable CORS
# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["*"],  # ✔️ Good for dev. Use specific origins in production.
# #     allow_credentials=True,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )


# # # API endpoint: Predict Emotion
# # @app.post("/predict")
# # async def predict_emotion(data: LandmarkData):
# #     landmarks = np.array(data.landmarks).astype(np.float32)
# #     if landmarks.shape != (468, 3):
# #         return {"error": "Invalid landmarks shape"}

# #     input_tensor = torch.tensor(landmarks).unsqueeze(0).to(device)
# #     with torch.no_grad():
# #         output = model(input_tensor)
# #         _, predicted = torch.max(output, 1)
# #         emotion = emotion_labels.get(predicted.item(), "Unknown")

# #     # Log the prediction
# #     with open(csv_file, "a", newline="") as file:
# #         writer = csv.writer(file)
# #         writer.writerow([datetime.now().isoformat(), emotion])

# #     return {"predicted_emotion": emotion}

# # # Run the server
# # if __name__ == "__main__":
# #     import uvicorn
# #     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


# # # 1. Load the CSV file
# # df = pd.read_csv('emotions_log.csv', names=["timestamp", "emotion"], skiprows=1)

# # # 2. Count how many times each emotion appears
# # emotion_counts = df['emotion'].value_counts()

# # # 3. Total number of predictions
# # total_predictions = len(df)

# # # 4. Calculate percentages
# # emotion_percentages = (emotion_counts / total_predictions) * 100

# # # 5. Convert to dictionary
# # emotion_percentages_dict = emotion_percentages.to_dict()

# # # 6. Print percentages
# # print("Emotion percentages:")
# # for emotion, percent in emotion_percentages_dict.items():
# #     print(f"{emotion}: {percent:.2f}%")

# # # 7. Save percentages to a new CSV file (optional)
# # emotion_percentages.to_csv('emotion_percentages.csv', header=True)
# import os
# import collections
# from datetime import datetime
# from typing import List
# import numpy as np
# import torch
# import torch.nn as nn
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from contextlib import asynccontextmanager

# # Define the Transformer Model
# class FaceEmotionTransformer(nn.Module):
#     def __init__(self, input_dim=3, seq_length=468, num_classes=7, embed_dim=128, num_heads=8, num_layers=4):
#         super(FaceEmotionTransformer, self).__init__()
#         self.embedding = nn.Linear(input_dim, embed_dim)
#         encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=0.1)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#         self.fc = nn.Linear(embed_dim, num_classes)

#     def forward(self, x):
#         x = self.embedding(x)
#         x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch, embed_dim)
#         x = self.transformer_encoder(x)
#         x = x.mean(dim=0)  # Global average pooling
#         return self.fc(x)

# # Pydantic Model for incoming landmark data
# class LandmarkData(BaseModel):
#     landmarks: List[List[float]]

# # Emotion labels mapping
# emotion_labels = {
#     0: "Anger",
#     1: "Disgust",
#     2: "Fear",
#     3: "Happiness",
#     4: "Sadness",
#     5: "Surprise",
#     6: "Neutral",
# }

# # Setup device and load model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = FaceEmotionTransformer(num_classes=7).to(device)
# try:
#     print("Loading model...")
#     model.load_state_dict(torch.load("Emotion_model2000.pth", map_location=device))
#     model.eval()
#     print("Model loaded successfully.")
# except Exception as e:
#     print("Model load failed:", e)

# # Lifespan context
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     yield
#     # Shutdown logic here
#     print("🚪 Server is shutting down...")

# # Initialize FastAPI app
# app = FastAPI(lifespan=lifespan)

# # Enable CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # ✔️ Good for dev. Use specific origins in production.
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # API endpoint: Predict Emotion
# @app.post("/predict")
# async def predict_emotion(data: LandmarkData):
#     landmarks = np.array(data.landmarks).astype(np.float32)
#     if landmarks.shape != (468, 3):
#         return {"error": "Invalid landmarks shape"}

#     input_tensor = torch.tensor(landmarks).unsqueeze(0).to(device)
#     with torch.no_grad():
#         output = model(input_tensor)
#         _, predicted = torch.max(output, 1)
#         emotion = emotion_labels.get(predicted.item(), "Unknown")

#     print(emotion)
#     return {"predicted_emotion": emotion}

# # Run the server
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)



# import os
# import collections
# from datetime import datetime
# from typing import List
# import numpy as np
# import torch
# import torch.nn as nn
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from contextlib import asynccontextmanager

# # Define the Transformer Model
# class FaceEmotionTransformer(nn.Module):
#     def __init__(self, input_dim=3, seq_length=468, num_classes=7, embed_dim=128, num_heads=8, num_layers=4):
#         super(FaceEmotionTransformer, self).__init__()
#         self.embedding = nn.Linear(input_dim, embed_dim)
#         encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=0.1)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#         self.fc = nn.Linear(embed_dim, num_classes)

#     def forward(self, x):
#         x = self.embedding(x)
#         x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch, embed_dim)
#         x = self.transformer_encoder(x)
#         x = x.mean(dim=0)  # Global average pooling
#         return self.fc(x)

# # Pydantic Models
# class LandmarkData(BaseModel):
#     landmarks: List[List[float]]

# class EmotionData(BaseModel):
#     emotion: str

# # Emotion labels mapping
# emotion_labels = {
#     0: "Anger",
#     1: "Disgust",
#     2: "Fear",
#     3: "Happiness",
#     4: "Sadness",
#     5: "Surprise",
#     6: "Neutral",
# }

# # Setup device and load model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = FaceEmotionTransformer(num_classes=7).to(device)
# try:
#     print("Loading model...")
#     model.load_state_dict(torch.load("Emotion_model2000.pth", map_location=device))
#     model.eval()
#     print("Model loaded successfully.")
# except Exception as e:
#     print("Model load failed:", e)

# # Lifespan context
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     yield
#     print("🚪 Server is shutting down...")

# # Initialize FastAPI app
# app = FastAPI(lifespan=lifespan)

# # Enable CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["https://<frontend>.onrender.com", "http://localhost:5173"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # API endpoint: Predict Emotion
# @app.post("/predict")
# async def predict_emotion(data: LandmarkData):
#     landmarks = np.array(data.landmarks).astype(np.float32)
#     if landmarks.shape != (468, 3):
#         return {"error": "Invalid landmarks shape"}

#     input_tensor = torch.tensor(landmarks).unsqueeze(0).to(device)
#     with torch.no_grad():
#         output = model(input_tensor)
#         _, predicted = torch.max(output, 1)
#         emotion = emotion_labels.get(predicted.item(), "Unknown")

#     print(emotion)
#     return {"predicted_emotion": emotion}

# # API endpoint: Game Next Level
# @app.post("/api/game/next-level")
# async def next_level(data: EmotionData):
#     emotion = data.emotion
#     # Placeholder logic for game progression
#     print(f"Received emotion for next level: {emotion}")
#     return {"status": "success", "emotion": emotion}

# # Run the server
# if __name__ == "__main__":
#     import uvicorn
#     port = int(os.getenv("PORT", 10000))
#     uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

import os
import collections
from datetime import datetime
from typing import List
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

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

# Setup device and model (lazy-loaded)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
MODEL_PATH = "fastapi/Emotion_model2000.pth"

def load_model():
    global model
    if model is None:
        try:
            print("Loading model...")
            model = FaceEmotionTransformer(num_classes=7).to(device)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model.eval()
            model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Model load failed: {e}")
            raise e
    return model

# Lifespan context
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    print("🚪 Server is shutting down...")

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
        landmarks = np.array(data.landmarks).astype(np.float32)
        if landmarks.shape != (468, 3):
            raise HTTPException(status_code=400, detail=f"Invalid landmarks shape: {landmarks.shape}, expected (468, 3)")

        model = load_model()
        input_tensor = torch.tensor(landmarks).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            emotion = emotion_labels.get(predicted.item(), "Unknown")

        print(f"Predicted emotion: {emotion}")
        return {"predicted_emotion": emotion}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# API endpoint: Game Next Level
# @app.post("/api/game/next-level")
# async def next_level(data: EmotionData):
#     try:
#         emotion = data.emotion
#         level_increment = 1 if emotion in ["Happiness", "Surprise"] else 0
#         print(f"Received emotion for next level: {emotion}, Increment: {level_increment}")
#         return {
#             "status": "success",
#             "emotion": emotion,
#             "level_increment": level_increment
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Next level computation failed: {str(e)}")

# Run the server
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)