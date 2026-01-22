import torch
import os

# Point this to your data path
ROOT = "/mount/studenten/projects/rasoulta/dataset/tmpl-captioned"

# Get a file
files = [f for f in os.listdir(ROOT) if f.endswith('.pt')]
file_path = os.path.join(ROOT, files[0])

print(f"Loading: {file_path}")
data = torch.load(file_path)

print("\n--- Keys in Data Object ---")
print(data.keys)

if hasattr(data, 'caption_dict'):
    print("\n--- Content of caption_dict ---")
    print(data.caption_dict)

if hasattr(data, 'maneuver_id'):
    print(f"\n--- Top Level maneuver_id ---")
    print(data.maneuver_id)
else:
    print("\nXXX maneuver_id is NOT a top level attribute XXX")