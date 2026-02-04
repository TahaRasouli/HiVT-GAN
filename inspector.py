import os
import torch
import glob
import random
from textwrap import fill

# --- CONFIGURATION ---
DATA_DIR = "/mount/studenten/projects/rasoulta/dataset/captioned"
NUM_SAMPLES_TO_INSPECT = 20

def main():
    if not os.path.exists(DATA_DIR):
        print(f"Error: Directory {DATA_DIR} does not exist.")
        return

    print(f"Scanning {DATA_DIR}...")
    files = glob.glob(os.path.join(DATA_DIR, "**/*.pt"), recursive=True)
    
    if not files:
        print("No .pt files found.")
        return

    print(f"Found {len(files)} files. Sampling {NUM_SAMPLES_TO_INSPECT} for inspection...\n")
    
    # Pick random files
    samples = random.sample(files, min(len(files), NUM_SAMPLES_TO_INSPECT))

    for i, file_path in enumerate(samples):
        try:
            # Load data
            data = torch.load(file_path, weights_only=False)
            
            # Extract Fields
            if isinstance(data, dict):
                cap_dict = data.get('caption_dict', {})
            else:
                cap_dict = getattr(data, 'caption_dict', {})
            
            category = cap_dict.get('maneuver_category', "N/A")
            lane_type = cap_dict.get('lane_type', "N/A")
            description = cap_dict.get('scene_description', "N/A")
            
            # --- DISPLAY ---
            print(f"SAMPLE #{i+1}: {os.path.basename(file_path)}")
            print("-" * 60)
            print(f"LABEL:      {category}")
            print(f"LANE TYPE:  {lane_type}")
            print("CAPTION:")
            print(fill(description, width=60, initial_indent="  ", subsequent_indent="  "))
            print("-" * 60 + "\n")

        except Exception as e:
            print(f"Error reading {os.path.basename(file_path)}: {e}")

if __name__ == "__main__":
    main()
