import os
import sys
import torch
import json
import random
from tqdm import tqdm

# --- 1. SETUP IMPORTS (CRITICAL FOR TORCH.LOAD) ---
# Add the current directory (project root) to sys.path to find utils
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) # Go up one level from 'datasets/'
sys.path.append(project_root)

# We MUST import TemporalData so torch.load knows how to reconstruct the object
try:
    from utils import TemporalData
except ImportError:
    print("WARNING: Could not import TemporalData. Torch.load might fail if the objects use this class.")

# CONFIG
DATA_DIR = "/mount/studenten/projects/rasoulta/dataset/captioned"
OUTPUT_FILE = "balanced_splits.json"
VAL_SPLIT_RATIO = 0.1 

def get_behavior_type(caption):
    text = caption.lower()
    
    # Rare behaviors (Keep ALL)
    if any(x in text for x in ["turn", "change lane", "intersection", "roundabout", "merge"]):
        return "rare"
    
    # Common behaviors (Downsample)
    return "common"

def main():
    print(f"Scanning files in {DATA_DIR}...")
    
    all_files = []
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            if f.endswith(".pt"):
                all_files.append(os.path.join(root, f))
                
    print(f"Found {len(all_files)} files. Analyzing content...")
    
    rare_files = []
    common_files = []
    errors = 0
    
    # DEBUG: Check the first file to ensure structure is correct
    if len(all_files) > 0:
        try:
            test_data = torch.load(all_files[0])
            print("\n[DEBUG] Inspecting first file structure:")
            print(f"Type: {type(test_data)}")
            if hasattr(test_data, 'caption_dict'):
                print(f"Caption Dict: {test_data.caption_dict}")
            else:
                print("(!) 'caption_dict' attribute MISSING on first file.")
                print(f"Available keys/attributes: {test_data.__dict__.keys()}")
        except Exception as e:
            print(f"\n[CRITICAL] Failed to load first file: {e}")
            return

    for fpath in tqdm(all_files):
        try:
            data = torch.load(fpath)
            
            # Robust extraction
            caption = ""
            if hasattr(data, 'caption_dict'):
                caption = data.caption_dict.get('driving_behavior', "")
            
            if not caption:
                # Fallback: maybe it didn't save correctly?
                continue

            b_type = get_behavior_type(caption)
            
            if b_type == "rare":
                rare_files.append(fpath)
            else:
                common_files.append(fpath)
        except Exception as e:
            errors += 1
            continue

    print("\n--- Analysis Results ---")
    print(f"Rare Cases (Turns/LaneChange): {len(rare_files)}")
    print(f"Common Cases (Straight/Stop): {len(common_files)}")
    print(f"Errors/Skipped: {errors}")
    
    if len(rare_files) == 0 and len(common_files) == 0:
        print("ERROR: No valid data found. Check the DEBUG output above.")
        return

    # Balancing Strategy
    # We want at least a 30/70 split if possible, or 50/50
    target_common_count = len(rare_files) * 2 
    
    # If we have very few rare cases, don't delete all common files, keep a minimum floor
    target_common_count = max(target_common_count, 1000) 
    
    if len(common_files) > target_common_count:
        print(f"Downsampling Common files from {len(common_files)} to {target_common_count}...")
        random.shuffle(common_files)
        common_files = common_files[:target_common_count]
        
    final_dataset = rare_files + common_files
    random.shuffle(final_dataset)
    
    print(f"Final Balanced Dataset Size: {len(final_dataset)}")
    
    # Split
    val_size = int(len(final_dataset) * VAL_SPLIT_RATIO)
    val_set = final_dataset[:val_size]
    train_set = final_dataset[val_size:]
    
    splits = {
        "train": train_set,
        "val": val_set
    }
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(splits, f, indent=2)
        
    print(f"Saved splits to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()