import torch
import os
from tqdm import tqdm

# Update this path
ROOT = "/mount/studenten/projects/rasoulta/dataset/tmpl-captioned"

files = [f for f in os.listdir(ROOT) if f.endswith('.pt')]
unique_labels = set()

print(f"Scanning {len(files)} files for unique labels...")

for f in tqdm(files):
    try:
        data = torch.load(os.path.join(ROOT, f))
        
        # Check where the label lives based on your debug output
        label = None
        
        # Priority 1: Top-level 'maneuver_category' (e.g., 'Straight Drive')
        if hasattr(data, 'maneuver_category'):
            label = data.maneuver_category
            
        # Priority 2: Inside caption_dict['category']
        elif hasattr(data, 'caption_dict') and 'category' in data.caption_dict:
            label = data.caption_dict['category']
            
        if label:
            unique_labels.add(label)
            
    except Exception as e:
        continue

print("\n--- FOUND CLASSES ---")
print("Copy these exact strings into your mapping:")
for label in sorted(list(unique_labels)):
    print(f"'{label}'")