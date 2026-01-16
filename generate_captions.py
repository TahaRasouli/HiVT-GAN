import torch
import numpy as np
import os
from glob import glob
from tqdm import tqdm
from nuscenes.map_expansion.map_api import NuScenesMap

# --- CONFIGURATION ---
INPUT_DATA_ROOT = "/mount/studenten/projects/rasoulta/dataset"
OUTPUT_DATA_ROOT = "/mount/studenten/projects/rasoulta/geo-caption" # Updated to your path
NUSCENES_MAP_ROOT = "/mount/arbeitsdaten/analysis/rasoulta/nuscenes/nuscenes_meta"

MAX_SAMPLES_PER_CLASS = 1500 
MAP_CACHE = {}

def get_nusc_map(city):
    if city not in MAP_CACHE:
        try:
            MAP_CACHE[city] = NuScenesMap(dataroot=NUSCENES_MAP_ROOT, map_name=city)
        except Exception as e:
            # Only print this once per city to avoid spam
            if city not in MAP_CACHE:
                print(f"[Map Load Error] Could not load map for {city}: {e}")
                MAP_CACHE[city] = None # cache failure so we don't retry
            return None
    return MAP_CACHE[city]

def get_attr(data, key):
    """Helper to safely get attributes from either Dict or PyG Data object"""
    if isinstance(data, dict):
        return data.get(key)
    else:
        # PyTorch Geometric Data object
        if hasattr(data, key):
            return getattr(data, key)
        return None

def get_geometric_label(nusc_map, origin, theta, trajectory):
    # 1. Kinematics
    x_final, y_final = trajectory[-1]
    displacement = np.linalg.norm(trajectory[-1])
    
    # Heading
    v_start = trajectory[5] - trajectory[0]
    v_end = trajectory[-1] - trajectory[-6]
    angle_start = np.arctan2(v_start[1], v_start[0])
    angle_end = np.arctan2(v_end[1], v_end[0])
    diff = (angle_end - angle_start + np.pi) % (2 * np.pi) - np.pi
    diff_deg = np.degrees(diff)

    # 2. Map Context
    is_intersection = False
    if nusc_map is not None:
        try:
            x_global, y_global = origin[0], origin[1]
            patch_box = (x_global - 2, y_global - 2, x_global + 2, y_global + 2)
            layers = nusc_map.get_records_in_patch(patch_box, ['road_segment'], mode='intersect')
            if 'road_segment' in layers:
                for token in layers['road_segment']:
                    if nusc_map.get('road_segment', token)['is_intersection']:
                        is_intersection = True
                        break
        except Exception as e:
            pass # Map query failed, assume no intersection
    
    location_str = "at an intersection" if is_intersection else "on the road"

    # 3. Categorization
    category = "straight"
    caption = f"The ego vehicle moves straight {location_str}."

    if displacement < 2.0:
        category = "stationary"
        caption = "The ego vehicle is stationary."
    elif abs(diff_deg) > 100:
        category = "u_turn"
        caption = f"The ego vehicle performs a U-turn {location_str}."
    elif diff_deg > 25:
        category = "turn_left"
        caption = f"The ego vehicle turns left {location_str}."
    elif diff_deg < -25:
        category = "turn_right"
        caption = f"The ego vehicle turns right {location_str}."
    elif y_final > 2.0:
        category = "lane_change_left"
        caption = f"The ego vehicle changes lane to the left {location_str}."
    elif y_final < -2.0:
        category = "lane_change_right"
        caption = f"The ego vehicle changes lane to the right {location_str}."

    return caption, category

def process_dataset():
    if not os.path.exists(OUTPUT_DATA_ROOT):
        os.makedirs(OUTPUT_DATA_ROOT)
    
    files = glob(os.path.join(INPUT_DATA_ROOT, "**/*.pt"), recursive=True)
    print(f"Found {len(files)} files.")

    class_counts = {k: 0 for k in ["stationary", "straight", "turn_left", "turn_right", "lane_change_left", "lane_change_right", "u_turn"]}
    
    errors_printed = 0
    processed_count = 0

    print("Starting processing...")
    for file_path in tqdm(files):
        try:
            data = torch.load(file_path)
            
            # --- ROBUST ACCESS ---
            city = get_attr(data, 'city')
            origin = get_attr(data, 'origin')
            theta = get_attr(data, 'theta')
            trajectory = get_attr(data, 'y')

            # Validation
            if city is None or origin is None or theta is None or trajectory is None:
                raise ValueError(f"Missing keys in file. Found: city={city}, origin={origin}, theta={theta}, y={trajectory is not None}")

            # Normalize types
            if hasattr(origin, 'numpy'): origin = origin.numpy()
            if hasattr(theta, 'item'): theta = theta.item()
            if hasattr(trajectory, 'cpu'): trajectory = trajectory.cpu().numpy()

            # --- PROCESS ---
            nusc_map = get_nusc_map(city)
            new_caption, category = get_geometric_label(nusc_map, origin, theta, trajectory)
            
            if class_counts[category] >= MAX_SAMPLES_PER_CLASS:
                continue

            class_counts[category] += 1
            processed_count += 1

            # --- SAVE ---
            # We must be careful not to break the object structure
            # If it's an object, we set attributes. If dict, set keys.
            if isinstance(data, dict):
                data['caption_string'] = new_caption
                data['maneuver_category'] = category
            else:
                data.caption_string = new_caption
                data.maneuver_category = category

            rel_path = os.path.relpath(file_path, INPUT_DATA_ROOT)
            save_path = os.path.join(OUTPUT_DATA_ROOT, rel_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(data, save_path)

        except Exception as e:
            if errors_printed < 5:
                print(f"\n[ERROR] Failed processing {os.path.basename(file_path)}: {e}")
                errors_printed += 1
            continue

    print("\n--- Processing Complete ---")
    print(f"Total Files Saved: {processed_count}")
    print("Class Distribution:")
    for k, v in class_counts.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    process_dataset()