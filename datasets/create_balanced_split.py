import os
import torch
import json
import random
from tqdm import tqdm

# CONFIG
DATA_DIR = "/mount/studenten/projects/rasoulta/dataset/captioned"
OUTPUT_FILE = "balanced_splits.json"
VAL_SPLIT_RATIO = 0.1 # 10% for validation

def get_behavior_type(caption):
    """Classifies caption into 'Common' (Drop) or 'Rare' (Keep)"""
    text = caption.lower()
    
    # Rare behaviors (Keep ALL of these)
    if "turn" in text: return "rare"
    if "change lane" in text: return "rare"
    if "intersection" in text: return "rare"
    if "roundabout" in text: return "rare"
    
    # Common behaviors (Downsample these)
    if "continue straight" in text: return "common"
    if "stop" in text: return "common"
    if "stationary" in text: return "common"
    
    return "common"

def main():
    print(f"Scanning files in {DATA_DIR}...")
    
    # 1. Collect all PT files
    all_files = []
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            if f.endswith(".pt"):
                all_files.append(os.path.join(root, f))
                
    print(f"Found {len(all_files)} files. Analyzing content...")
    
    # 2. Categorize
    rare_files = []
    common_files = []
    
    for fpath in tqdm(all_files):
        try:
            # We assume the file has 'caption_dict'
            # To be fast, we rely on the fact that torch.load loads the whole object. 
            # If this is too slow, we just have to wait once.
            data = torch.load(fpath)
            
            if not hasattr(data, 'caption_dict'): continue
            
            caption = data.caption_dict.get('driving_behavior', "")
            b_type = get_behavior_type(caption)
            
            if b_type == "rare":
                rare_files.append(fpath)
            else:
                common_files.append(fpath)
        except:
            continue

    print(f"Rare Cases (Turns/LaneChange): {len(rare_files)}")
    print(f"Common Cases (Straight/Stop): {len(common_files)}")
    
    # 3. Balance
    # Strategy: Keep ALL rare files. Keep equal amount of common files.
    target_common_count = len(rare_files) * 2 # Allow 2:1 ratio (Straight is still important)
    
    if len(common_files) > target_common_count:
        print(f"Downsampling Common files from {len(common_files)} to {target_common_count}...")
        random.shuffle(common_files)
        common_files = common_files[:target_common_count]
        
    final_dataset = rare_files + common_files
    random.shuffle(final_dataset)
    
    print(f"Final Balanced Dataset Size: {len(final_dataset)}")
    
    # 4. Split Train/Val
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