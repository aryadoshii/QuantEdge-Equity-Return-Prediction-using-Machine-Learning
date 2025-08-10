# config.py
import os

# --- PASTE THE ABSOLUTE PATH FROM YOUR TERMINAL HERE ---
# It should look something like '/Users/your_username/Desktop/stock-return-prediction'
PROJECT_ROOT = "/Users/aryadoshii/Desktop/stock-return-prediction"

# --- The rest of the paths are now guaranteed to be correct ---
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
PROCESSED_FILE = os.path.join(PROCESSED_DATA_DIR, "all_features.parquet")

print("--- CONFIG FILE LOADED ---")
print(f"Project Root is: {PROJECT_ROOT}")
print(f"Processed file will be at: {PROCESSED_FILE}")
print("--------------------------")