import os
import glob
import torch
import pandas as pd
import time
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor, Trainer, TrainingArguments
from PIL import Image
import evaluate

# Paths
train_dir = r'C:\Users\bannu\OneDrive\Desktop\clone\Facial-Emotion-Recognition\archive\train'
test_dir = r'C:\Users\bannu\OneDrive\Desktop\clone\Facial-Emotion-Recognition\archive\test'
model_path = r'C:\Users\bannu\OneDrive\Desktop\clone\fine_tuned_vit_model'

# Emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
label_map = {label: idx for idx, label in enumerate(emotion_labels)}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Image processor
image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Dataset class
class EmotionDataset(Dataset):
    def __init__(self, folder, label_map):
        self.images = []
        self.labels = []
        self.label_map = label_map
        self._load_images_and_labels(folder)
    
    def _load_images_and_labels(self, folder):
        for label in self.label_map:
            label_dir = os.path.join(folder, label)
            if os.path.isdir(label_dir):
                img_paths = glob.glob(f"{label_dir}/*.jpg") + glob.glob(f"{label_dir}/*.png") + glob.glob(f"{label_dir}/*.jpeg")
                self.images.extend(img_paths)
                self.labels.extend([self.label_map[label]] * len(img_paths))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        image = transform(image)
        label = self.labels[idx]
        return {'pixel_values': image, 'label': label}

# Load datasets
train_dataset = EmotionDataset(train_dir, label_map)
test_dataset = EmotionDataset(test_dir, label_map)

# Load or train model
if os.path.exists(model_path):
    print("Loading pre-trained model...")
    model = ViTForImageClassification.from_pretrained(model_path).to(device)
    image_processor = ViTImageProcessor.from_pretrained(model_path)
else:
    print("Training model from scratch...")
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224', 
        num_labels=len(emotion_labels),
        ignore_mismatched_sizes=True
    ).to(device)
    
    # Training setup
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        save_strategy="epoch",
        logging_dir='./logs',
        logging_steps=10,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),  
    )
    
    metric = evaluate.load('accuracy')
    
    def compute_metrics(p):
        return metric.compute(predictions=p.predictions.argmax(-1), references=p.label_ids)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    
    
    model.save_pretrained(model_path)
    image_processor.save_pretrained(model_path)

# -------------------------------------------------------------------
# Real-time Emotion Detection with OpenCV and Logging to CSV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model.eval()
cap = cv2.VideoCapture(0)

# CSV file setup
csv_path = "emotion_changes.csv"
df = pd.DataFrame(columns=["Timestamp", "Emotion"])
last_emotion = None  

def preprocess_face(face):
    """ Preprocess detected face for model inference """
    face_rgb = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
    face_resized = cv2.resize(face_rgb, (224, 224))
    face_pil = Image.fromarray(face_resized)
    inputs = image_processor(images=face_pil, return_tensors='pt')
    return {k: v.to(device) for k, v in inputs.items()}  # Move to GPU

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(40, 40))
    
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]  
        inputs = preprocess_face(face)  
        
        with torch.no_grad():
            outputs = model(**inputs) 
        logits = outputs.logits
        predicted_label = logits.argmax(-1).item()
        emotion = emotion_labels[predicted_label]
        
        
        if emotion != last_emotion:
            last_emotion = emotion
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

            
            df = df._append({"Timestamp": timestamp, "Emotion": emotion}, ignore_index=True)
            df.to_csv(csv_path, index=False)

    
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    cv2.imshow('Emotion Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
