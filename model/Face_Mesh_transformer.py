import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# Load Dataset
file_path = r"C:\Users\bannu\OneDrive\Desktop\Joyverse\face_mesh_data.csv"
df = pd.read_csv(file_path)

# Encode Emotion Labels
label_encoder = LabelEncoder()
df['Expression'] = label_encoder.fit_transform(df['Expression'])

# Extract Features and Labels
df_numeric = df.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')  # Convert to numeric
df_numeric.fillna(0, inplace=True)  # Fill NaN values
X = df_numeric.values.astype(np.float32)  
y = df['Expression'].values

# Reshape input data
X_reshaped = X.reshape(len(X), 468, 3)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

# Convert to PyTorch Tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# DataLoader
#Bundles inputs and labels together into one dataset object.
# Each item in this dataset returns a tuple:
# (features, label)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define Transformer Model
class FaceEmotionTransformer(nn.Module):
    def __init__(self, input_dim=3, seq_length=468, num_classes=6, embed_dim=512, num_heads=12, num_layers=6):
        super(FaceEmotionTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # (seq_length, batch_size, embed_dim)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)
        return self.fc(x)

# Initialize Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(label_encoder.classes_)
model = FaceEmotionTransformer(num_classes=num_classes).to(device)

# Define Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001)

# Check if trained model exists
model_path = "Emotion_model2000(1).pth"
if os.path.exists(model_path):
    print("Loading pre-trained model...")
    model.load_state_dict(torch.load(model_path, map_location=device))
else:
    print("Training model from scratch...")
    num_epochs = 2000
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            X_batch, y_batch = batch
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

    # Save the trained model
    torch.save(model.state_dict(), model_path)
    print("Model saved successfully!")


# Evaluate Model
model.eval()
y_pred, y_true = [], []
with torch.no_grad():
    for batch in test_loader:
        X_batch, y_batch = batch
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(y_batch.cpu().numpy())

accuracy = accuracy_score(y_true, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Real-time Emotion Detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

def extract_landmarks(image, draw=False):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    if results.multi_face_landmarks:
        landmarks = [[lm.x, lm.y, lm.z] for lm in results.multi_face_landmarks[0].landmark]
        landmarks = np.array(landmarks)

        if draw:
            for lm in results.multi_face_landmarks[0].landmark:
                h, w, _ = image.shape
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(image, (x, y), 1, (0, 255, 0), -1) 

        return landmarks
    return None

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    landmarks = extract_landmarks(frame, draw=True)  # Draw face mesh on frame
    
    if landmarks is not None:
        landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(landmarks_tensor)
            _, predicted = torch.max(output, 1)
            emotion = label_encoder.inverse_transform([predicted.cpu().item()])[0]
        
        cv2.putText(frame, f'Emotion: {emotion}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Emotion Detection with Face Mesh", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()