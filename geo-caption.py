import torch
import glob
import os
import random
import numpy as np
from collections import defaultdict
from nuscenes.map_expansion.map_api import NuScenesMap
from tqdm import tqdm
import argparse

# =========================
# 1. CONFIGURATION
# =========================
DEFAULT_OUTPUT_DIR = "/mount/studenten/projects/rasoulta/dataset/tmpl-captioned"

# =========================
# 2. LANGUAGE TEMPLATES
# =========================
TEMPLATES = {
    "Left Turn": [
        "The ego vehicle executes a left turn.",
        "The vehicle initiates a turn to the left.",
        "The car turns left at the intersection.",
        "A left turn maneuver is performed."
    ],
    "Right Turn": [
        "The ego vehicle executes a right turn.",
        "The vehicle initiates a turn to the right.",
        "The car turns right at the intersection.",
        "A right turn maneuver is performed."
    ],
    "U-Turn": [
        "The ego vehicle performs a U-turn.",
        "The vehicle executes a complete U-turn.",
        "The car reverses its direction via a U-turn.",
        "A U-turn maneuver is executed."
    ],
    "Lane Change Left": [
        "The ego vehicle changes lanes to the left.",
        "The vehicle merges into the left lane.",
        "The car shifts to the lane on its left.",
        "A lane change to the left is performed."
    ],
    "Lane Change Right": [
        "The ego vehicle changes lanes to the right.",
        "The vehicle merges into the right lane.",
        "The car shifts to the lane on its right.",
        "A lane change to the right is performed."
    ],
    "Straight Drive": [
        "The ego vehicle drives straight.",
        "The vehicle proceeds forward without turning.",
        "The car maintains its course.",
        "It drives straight ahead."
    ],
    "Stationary Stop": [
        "The ego vehicle remains stationary.",
        "The vehicle is stopped.",
        "The car holds its position.",
        "It is waiting in place."
    ]
}

LANE_TEMPLATES = {
    "maintain": [
        "It maintains its current lane.",
        "The vehicle stays within the lane.",
        "It follows the lane center.",
        "No lane deviation occurs."
    ],
    "change_left": [
        "It is changing lanes to the left.",
        "A merge to the left is occurring.",
        "It crosses into the left lane.",
        "The lateral move targets the left lane."
    ],
    "change_right": [
        "It is changing lanes to the right.",
        "A merge to the right is occurring.",
        "It crosses into the right lane.",
        "The lateral move targets the right lane."
    ],
    "turn_action": [
        "It turns onto a new path.",
        "The vehicle enters the intersection.",
        "It follows the turning lane.",
        "The trajectory curves significantly."
    ],
    "stop_action": [
        "It holds its position.",
        "No movement is detected.",
        "It waits at the current location.",
        "The velocity is near zero."
    ]
}

# =========================
# 3. ROBUST LOGIC (Topology + Geometry)
# =========================
def classify_maneuver(nusc_map, global_traj):
    """
    Robust classification using Map Topology (Neighbors) + S-Curve Geometry.
    """
    p_start = global_traj[0]
    p_mid   = global_traj[15] # Approx mid-point (1.5s)
    p_end   = global_traj[-1]

    # --- 1. Basic Geometry ---
    v_start = global_traj[5] - global_traj[0]
    v_end   = global_traj[-1] - global_traj[-6]
    
    # Heading Change (Delta Yaw)
    angle_start = np.arctan2(v_start[1], v_start[0])
    angle_end = np.arctan2(v_end[1], v_end[0])
    delta_deg = np.degrees(angle_end - angle_start)
    delta_deg = (delta_deg + 180) % 360 - 180
    
    # Displacement
    dist_total = np.linalg.norm(p_end - p_start)

    # --- 2. Map Queries (The Source of Truth) ---
    def get_lane(p):
        try: 
            # Search with a small radius (1m)
            layers = nusc_map.layers_on_point(p[0], p[1])
            if layers and 'lane' in layers:
                return layers['lane']
            return ''
        except: return ''

    l_start = get_lane(p_start)
    l_end   = get_lane(p_end)

    # Lookahead for End Lane if missing (project forward 5m)
    if not l_end:
        v_last = global_traj[-1] - global_traj[-2]
        p_proj = p_end + (v_last * 5.0)
        l_end = get_lane(p_proj)

    # --- 3. STATIONARY CHECK ---
    if dist_total < 2.0:
        return "Stationary Stop", "stop_action"

    # --- 4. TOPOLOGICAL CHECK ---
    if l_start and l_end and l_start != l_end:
        
        # A. Check Connectivity (Successors = Straight/Turn)
        outgoing = nusc_map.get_outgoing_lane_ids(l_start)
        incoming = nusc_map.get_incoming_lane_ids(l_end)
        
        if l_end in outgoing or l_start in incoming:
             # Longitudinal transition
             if abs(delta_deg) > 45: 
                 return ("Left Turn" if delta_deg > 0 else "Right Turn"), "turn_action"
             elif abs(delta_deg) > 135:
                 return "U-Turn", "turn_action"
             else:
                 return "Straight Drive", "maintain"

        # B. Check Adjacency (Neighbors = Lane Change)
        left_neighbors = nusc_map.get_left_lane_ids(l_start)
        right_neighbors = nusc_map.get_right_lane_ids(l_start)
        
        if l_end in left_neighbors:
            return "Lane Change Left", "change_left"
        if l_end in right_neighbors:
            return "Lane Change Right", "change_right"
            
        # C. Recursive Neighbor Check
        for ln in left_neighbors:
            if l_end in nusc_map.get_outgoing_lane_ids(ln):
                return "Lane Change Left", "change_left"
        for rn in right_neighbors:
            if l_end in nusc_map.get_outgoing_lane_ids(rn):
                return "Lane Change Right", "change_right"

    # --- 5. GEOMETRIC FALLBACK ---
    if abs(delta_deg) > 135: return "U-Turn", "turn_action"
    if delta_deg > 30:       return "Left Turn", "turn_action"
    if delta_deg < -30:      return "Right Turn", "turn_action"
    
    # S-Curve Detection (Implicit Lane Change)
    vec_chord = p_end - p_start
    len_chord = np.linalg.norm(vec_chord) + 1e-6
    unit_chord = vec_chord / len_chord
    unit_normal = np.array([-unit_chord[1], unit_chord[0]])
    vec_mid = p_mid - p_start
    lat_deviation = np.dot(vec_mid, unit_normal)
    
    if not l_start and not l_end:
        # If intersection and huge deviation with same heading -> Lane Change
        if abs(lat_deviation) > 2.5 and abs(delta_deg) < 20:
            if lat_deviation > 0: return "Lane Change Left", "change_left"
            else:                 return "Lane Change Right", "change_right"

    return "Straight Drive", "maintain"

# =========================
# 4. UTILS
# =========================
def ego_to_global(traj, origin, theta):
    traj = np.asarray(traj)
    if traj.shape[0] != 2: traj = traj.T 
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return (R @ traj).T + np.array(origin)

# =========================
# 5. MAIN PROCESSING LOOP
# =========================
def process_dataset(input_dir, output_dir, dataroot):
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Scanning files...")
    
    files = glob.glob(os.path.join(input_dir, "**", "*.pt"), recursive=True)
    print(f"Found {len(files)} files.")

    files.sort()
    maps = {}
    stats = defaultdict(int)

    # Ensure output root exists
    os.makedirs(output_dir, exist_ok=True)

    for f_path in tqdm(files, desc="Processing"):
        try:
            # 1. Load Data
            try: data = torch.load(f_path, weights_only=False)
            except: data = torch.load(f_path)
            
            # 2. Extract Attributes
            if isinstance(data, dict):
                city = data.get('city')
                traj = data.get('y')
                origin = data.get('origin')
                theta = data.get('theta')
                old_caps = data.get('caption_dict', {})
            else:
                city = getattr(data, 'city', None)
                traj = getattr(data, 'y', None)
                origin = getattr(data, 'origin', None)
                theta = getattr(data, 'theta', None)
                old_caps = getattr(data, 'caption_dict', {})
                if not isinstance(old_caps, dict): old_caps = {}

            if city is None: continue

            # 3. Load Map (Lazy Loading)
            if city not in maps:
                maps[city] = NuScenesMap(dataroot=dataroot, map_name=city)
            nusc_map = maps[city]

            # 4. Fix Shapes
            if hasattr(traj, 'cpu'): traj = traj.cpu().numpy()
            if hasattr(origin, 'cpu'): origin = origin.cpu().numpy()
            if hasattr(theta, 'item'): theta = theta.item()
            elif hasattr(theta, 'numpy'): theta = theta.item()

            traj = np.squeeze(traj)
            if traj.ndim == 3: traj = traj[0] # Ego
            origin = np.squeeze(origin)
            if origin.ndim > 1: origin = origin[0]

            # 5. CLASSIFY
            global_traj = ego_to_global(traj, origin, theta)
            cat, lane_key = classify_maneuver(nusc_map, global_traj)

            # 6. GENERATE CAPTION
            man_text = random.choice(TEMPLATES[cat])
            lane_text = random.choice(LANE_TEMPLATES[lane_key])
            scene_desc = old_caps.get('scene_description', "Driving in an urban environment.")
            if len(scene_desc) < 5: scene_desc = "Driving in an urban environment."

            full_caption = f"{man_text} {lane_text} {scene_desc}"
            
            new_caption_dict = {
                "maneuver_type": man_text,
                "lane_status": lane_text,
                "scene_description": scene_desc,
                "category": cat
            }

            # 7. UPDATE DATA OBJECT
            if isinstance(data, dict):
                data['caption_dict'] = new_caption_dict
                data['maneuver_category'] = cat
                data['scene_description'] = full_caption
            else:
                data.caption_dict = new_caption_dict
                data.maneuver_category = cat
                data.scene_description = full_caption

            # 8. CALCULATE NEW PATH (PRESERVE STRUCTURE)
            rel_path = os.path.relpath(f_path, input_dir) # e.g., "boston/file.pt"
            new_f_path = os.path.join(output_dir, rel_path) # e.g., "output/boston/file.pt"
            
            # Ensure subfolder exists
            os.makedirs(os.path.dirname(new_f_path), exist_ok=True)

            # 9. SAVE TO NEW LOCATION
            torch.save(data, new_f_path)
            stats[cat] += 1
            
        except Exception as e:
            # print(f"Error processing {f_path}: {e}")
            pass

    print("\n=== PROCESSING COMPLETE ===")
    print("New Distribution:")
    total = sum(stats.values())
    for k, v in stats.items():
        print(f"{k:<20}: {v} ({v/total*100:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Original dataset folder")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Destination for modified files")
    parser.add_argument("--dataroot", default="./", help="NuScenes root with /maps folder")
    args = parser.parse_args()
    
    process_dataset(args.input_dir, args.output_dir, args.dataroot)