import csv
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import os
import sys

# Configuration
CSV_FILE = "emotion_percentages.csv"
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "emotion_percentage"
COLLECTION_NAME = "emotions"

def store_csv_to_mongodb():
    try:
        # Connect to MongoDB
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        print("✅ Connected to MongoDB successfully")
        
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        
        # Verify CSV file
        if not os.path.exists(CSV_FILE):
            print(f"❌ CSV file {CSV_FILE} not found")
            return False
        
        # Read and process CSV
        entries = []
        existing_timestamps = set(str(doc['timestamp']) for doc in collection.find({}, {'timestamp': 1}))
        
        with open(CSV_FILE, 'r') as file:
            reader = csv.reader(file)
            
            for row in reader:
                # Skip empty rows and summary rows
                if len(row) == 2 and not any(x in row[1] for x in ['%', 'Summary']):
                    try:
                        timestamp = datetime.fromisoformat(row[0])
                        # Only add if timestamp doesn't exist
                        if str(timestamp) not in existing_timestamps:
                            entry = {
                                'timestamp': timestamp,
                                'emotion': row[1].strip()
                            }
                            entries.append(entry)
                    except ValueError as e:
                        continue
        
        # Store in MongoDB
        if entries:
            result = collection.insert_many(entries)
            print(f"✅ Successfully added {len(result.inserted_ids)} new entries to MongoDB")
            
            # Show total statistics
            stats = collection.aggregate([
                {"$group": {"_id": "$emotion", "count": {"$sum": 1}}}
            ])
            print("\n📊 Total Emotion Statistics:")
            for stat in stats:
                print(f"  {stat['_id']}: {stat['count']} entries")
            
            print(f"\n💾 Total entries in database: {collection.count_documents({})}")
            return True
        else:
            print("ℹ️ No new entries to add to MongoDB")
            print(f"💾 Current entries in database: {collection.count_documents({})}")
            return True
            
    except ConnectionFailure as e:
        print(f"❌ MongoDB Connection Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        print(traceback.format_exc())
        return False
    finally:
        if 'client' in locals():
            client.close()
            print("👋 MongoDB connection closed")

if __name__ == "__main__":
    print("🚀 Starting CSV to MongoDB transfer...")
    store_csv_to_mongodb()