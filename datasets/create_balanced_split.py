import os
import json
import random
from glob import glob

# --- CONFIGURATION ---
# The directory where you just generated the captioned .pt files
DATA_DIR = "/mount/studenten/projects/rasoulta/dataset/captioned" 
OUTPUT_JSON = "balanced_splits.json"
VAL_SPLIT_RATIO = 0.2  # 20% for validation

def main():
    print(f"Scanning {DATA_DIR}...")
    
    # Find all .pt files recursively
    files = glob(os.path.join(DATA_DIR, "**/*.pt"), recursive=True)
    
    if not files:
        print("Error: No .pt files found!")
        return

    # Sort to ensure deterministic shuffling (same seed = same split every time)
    files.sort()
    
    # Shuffle
    random.seed(42)
    random.shuffle(files)
    
    # Calculate split index
    split_idx = int(len(files) * (1 - VAL_SPLIT_RATIO))
    
    train_files = files[:split_idx]
    val_files = files[split_idx:]
    
    # Structure for the Dataset Class
    data = {
        "train": train_files,
        "val": val_files
    }
    
    # Save to JSON
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(data, f, indent=4)
        
    print(f"\n[SUCCESS] Split file created: {OUTPUT_JSON}")
    print(f"Total Files: {len(files)}")
    print(f"Train Set:   {len(train_files)} ({(len(train_files)/len(files))*100:.1f}%)")
    print(f"Val Set:     {len(val_files)} ({(len(val_files)/len(files))*100:.1f}%)")

if __name__ == "__main__":
    main()