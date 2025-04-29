import pandas as pd
from pymongo import MongoClient
import os

# Read the CSV file
csv_path = os.path.join(os.path.dirname(__file__), 'emotion_percentages.csv')
df = pd.read_csv(csv_path)

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['emotion_database']
collection = db['emotion_percentages']

# Convert DataFrame to list of dictionaries
records = df.to_dict('records')

# Insert the records into MongoDB
collection.insert_many(records)

print("Data successfully stored in MongoDB!")