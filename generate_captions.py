import torch
import numpy as np
import os
from glob import glob
from tqdm import tqdm
from nuscenes.map_expansion.map_api import NuScenesMap

# --- CONFIGURATION ---
# Input: Where your current .pt files are
INPUT_DATA_ROOT = "/mount/studenten/projects/rasoulta/dataset"

# Output: Where the NEW, cleaned .pt files will be saved
OUTPUT_DATA_ROOT = "/mount/studenten/projects/rasoulta/geo-caption" 

# Map Data Path
NUSCENES_MAP_ROOT = "/mount/arbeitsdaten/analysis/rasoulta/nuscenes/nuscenes_meta"

# Balancing: Maximum number of samples allowed per category
# (This prevents having 50,000 straight samples and only 500 turns)
MAX_SAMPLES_PER_CLASS = 1500 

MAP_CACHE = {}

def get_nusc_map(city):
    """Lazy loads map to memory."""
    if city not in MAP_CACHE:
        try:
            MAP_CACHE[city] = NuScenesMap(dataroot=NUSCENES_MAP_ROOT, map_name=city)
        except Exception as e:
            # print(f"Warning: Could not load map for {city}: {e}")
            return None
    return MAP_CACHE[city]

def get_geometric_label(nusc_map, origin, theta, trajectory):
    """
    The 'Geometric Oracle'. 
    Analytically determines the ground truth caption based on physics + map.
    """
    # 1. Calculate Kinematics (in Agent-Centric Frame)
    # x is forward, y is left/right deviation
    x_final, y_final = trajectory[-1]
    displacement = np.linalg.norm(trajectory[-1])
    
    # Calculate Heading Change (Start Vector vs End Vector)
    # Use indices [0->5] vs [-6->-1] to smooth out noise
    v_start = trajectory[5] - trajectory[0]
    v_end = trajectory[-1] - trajectory[-6]
    
    angle_start = np.arctan2(v_start[1], v_start[0])
    angle_end = np.arctan2(v_end[1], v_end[0])
    
    diff = angle_end - angle_start
    # Normalize angle diff to [-pi, pi]
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    diff_deg = np.degrees(diff)

    # 2. Check Map Context (Is it an Intersection?)
    is_intersection = False
    if nusc_map:
        x_global, y_global = origin[0], origin[1]
        # Check a small box around the ego vehicle
        patch_box = (x_global - 2, y_global - 2, x_global + 2, y_global + 2)
        try:
            layers = nusc_map.get_records_in_patch(patch_box, ['road_segment'], mode='intersect')
            if 'road_segment' in layers:
                for token in layers['road_segment']:
                    rec = nusc_map.get('road_segment', token)
                    if rec['is_intersection']:
                        is_intersection = True
                        break
        except: pass
    
    location_str = "at an intersection" if is_intersection else "on the road"

    # 3. Classify Maneuver (Priority Logic)
    category = "unknown"
    caption = ""

    # A. Stationary (Moved less than 2 meters in 3 seconds)
    if displacement < 2.0:
        category = "stationary"
        caption = "The ego vehicle is stationary."
        
    # B. U-Turn (Heading changed > 100 degrees)
    elif abs(diff_deg) > 100:
        category = "u_turn"
        caption = f"The ego vehicle performs a U-turn {location_str}."

    # C. Turns (Heading changed > 25 degrees)
    # Note: We prioritize Turns over Lane Changes. A big turn IS a turn.
    elif diff_deg > 25:
        category = "turn_left"
        caption = f"The ego vehicle turns left {location_str}."
    elif diff_deg < -25:
        category = "turn_right"
        caption = f"The ego vehicle turns right {location_str}."
        
    # D. Lane Changes (Lateral Deviation > 2.0m BUT Heading change is small)
    # If heading change is huge, it's a turn, not a lane change.
    elif y_final > 2.0:
        category = "lane_change_left"
        caption = f"The ego vehicle changes lane to the left {location_str}."
    elif y_final < -2.0:
        category = "lane_change_right"
        caption = f"The ego vehicle changes lane to the right {location_str}."
        
    # E. Straight (Everything else)
    else:
        category = "straight"
        caption = f"The ego vehicle moves straight {location_str}."

    return caption, category

def process_dataset():
    # 1. Setup Output
    if not os.path.exists(OUTPUT_DATA_ROOT):
        os.makedirs(OUTPUT_DATA_ROOT)
    
    print(f"Scanning {INPUT_DATA_ROOT}...")
    files = glob(os.path.join(INPUT_DATA_ROOT, "**/*.pt"), recursive=True)
    print(f"Found {len(files)} files.")

    class_counts = {
        "stationary": 0, "straight": 0,
        "turn_left": 0, "turn_right": 0,
        "lane_change_left": 0, "lane_change_right": 0,
        "u_turn": 0
    }
    
    processed_count = 0

    print("Generating Labels & Balancing...")
    # Use TQDM for progress bar
    for file_path in tqdm(files):
        try:
            # Load Data
            data = torch.load(file_path)
            
            # Extract Features
            city = data['city']
            origin = data['origin'] if isinstance(data['origin'], np.ndarray) else data['origin'].numpy()
            theta = data['theta'] if isinstance(data['theta'], float) else data['theta'].item()
            # Handle Trajectory (tensor or numpy)
            trajectory = data['y'] 
            if isinstance(trajectory, torch.Tensor):
                trajectory = trajectory.cpu().numpy()

            # --- GEOMETRIC ORACLE ---
            nusc_map = get_nusc_map(city)
            new_caption, category = get_geometric_label(nusc_map, origin, theta, trajectory)
            
            # --- BALANCING FILTER ---
            # If we already have 1500 of this type, SKIP it.
            if class_counts.get(category, 0) >= MAX_SAMPLES_PER_CLASS:
                continue

            # Update Counts
            class_counts[category] += 1
            processed_count += 1

            # --- SAVE NEW FILE ---
            # Inject the new caption directly into the object
            # Note: We save it as a raw string. 
            # (Your training dataloader needs to tokenize this string)
            data['caption_string'] = new_caption 
            data['maneuver_category'] = category # Useful for debugging/metrics later
            
            # Construct Output Path (preserving subfolders like 'train'/'val')
            rel_path = os.path.relpath(file_path, INPUT_DATA_ROOT)
            save_path = os.path.join(OUTPUT_DATA_ROOT, rel_path)
            
            # Ensure folder exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save
            torch.save(data, save_path)

        except Exception as e:
            # If map fails or data corrupt, just skip
            continue

    print("\n" + "="*40)
    print("DATASET GENERATION COMPLETE")
    print("="*40)
    print(f"Total Files Saved: {processed_count}")
    print("\nClass Distribution:")
    for k, v in class_counts.items():
        print(f"{k:<20}: {v}")
    print(f"\nNew dataset location: {OUTPUT_DATA_ROOT}")

if __name__ == "__main__":
    process_dataset()
