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
# 1. LANGUAGE TEMPLATES (Variety)
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
# 2. GEOMETRIC & TOPOLOGICAL LOGIC (v6)
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
            # Search with a small radius (1m) to catch lanes if point is slightly off
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

    # --- 4. TOPOLOGICAL CHECK (The "Curvature Proof" Method) ---
    # If we have valid IDs for both start and end, TRUST THE MAP.
    if l_start and l_end and l_start != l_end:
        
        # A. Check Connectivity (Successors = Straight)
        outgoing = nusc_map.get_outgoing_lane_ids(l_start)
        incoming = nusc_map.get_incoming_lane_ids(l_end)
        
        # If it's a direct successor sequence, it's NOT a lane change (even if ID changed)
        if l_end in outgoing or l_start in incoming:
             # It's a longitudinal transition. Now check curvature for Turn vs Straight.
             if abs(delta_deg) > 45: 
                 return ("Left Turn" if delta_deg > 0 else "Right Turn"), "turn_action"
             elif abs(delta_deg) > 135:
                 return "U-Turn", "turn_action"
             else:
                 return "Straight Drive", "maintain"

        # B. Check Adjacency (Neighbors = Lane Change)
        # This handles curved roads perfectly. If the map says "Lane B is left of Lane A", it's a Left LC.
        left_neighbors = nusc_map.get_left_lane_ids(l_start)
        right_neighbors = nusc_map.get_right_lane_ids(l_start)
        
        if l_end in left_neighbors:
            return "Lane Change Left", "change_left"
        if l_end in right_neighbors:
            return "Lane Change Right", "change_right"
            
        # C. Recursive Neighbor Check (Handling Multi-segment Lane Changes)
        # Sometimes you change lane AND move forward to the next segment simultaneously.
        # Check if l_end is a successor of a neighbor.
        for ln in left_neighbors:
            if l_end in nusc_map.get_outgoing_lane_ids(ln):
                return "Lane Change Left", "change_left"
        for rn in right_neighbors:
            if l_end in nusc_map.get_outgoing_lane_ids(rn):
                return "Lane Change Right", "change_right"

    # --- 5. GEOMETRIC FALLBACK (For Intersections / Missing IDs) ---
    # If we are here, either l_start==l_end, or IDs are missing/disconnected.
    # We rely on trajectory shape.
    
    # A. Check U-Turn / Turns first
    if abs(delta_deg) > 135: return "U-Turn", "turn_action"
    if delta_deg > 30:       return "Left Turn", "turn_action"
    if delta_deg < -30:      return "Right Turn", "turn_action"
    
    # B. S-Curve Detection for "Implicit" Lane Changes
    # If heading is roughly straight (<30 deg change), but we moved laterally.
    
    # Construct a "Chord" line from start to end
    vec_chord = p_end - p_start
    len_chord = np.linalg.norm(vec_chord) + 1e-6
    unit_chord = vec_chord / len_chord
    
    # Normal to chord (Left)
    unit_normal = np.array([-unit_chord[1], unit_chord[0]])
    
    # Calculate deviation of the MID point from the Chord line
    vec_mid = p_mid - p_start
    # Dot product with normal gives lateral distance from the straight line connecting start/end
    lat_deviation = np.dot(vec_mid, unit_normal)
    
    # Thresholds:
    # A generic lane change involves an "S" shape. 
    # However, if start/end headings are aligned, lateral deviation implies a shift.
    
    # If we have NO lane IDs (Intersection), be conservative.
    if not l_start and not l_end:
        # Only call LC if deviation is significant (e.g., > 2.5m) and headings align
        if abs(lat_deviation) > 2.5 and abs(delta_deg) < 20:
            if lat_deviation > 0: return "Lane Change Left", "change_left"
            else:                 return "Lane Change Right", "change_right"
            
    # If we HAVE Lane IDs but they are the same (l_start == l_end)
    # Check if the car is drifting out of the lane purely geometrically
    elif l_start == l_end and l_start:
        # Use map API to get lane width or orientation? Too slow.
        # Use simple heuristic: If we deviated significantly but stayed in "Same ID",
        # the map ID might be a long polygon.
        pass # Stick to "Maintain" if ID didn't change, to avoid false positives.

    return "Straight Drive", "maintain"

# =========================
# 3. UTILS
# =========================
def ego_to_global(traj, origin, theta):
    traj = np.asarray(traj)
    if traj.shape[0] != 2: traj = traj.T 
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return (R @ traj).T + np.array(origin)

# =========================
# 4. MAIN LOOP
# =========================
def process_dataset(input_dir, dataroot):
    print(f"Scanning {input_dir}...")
    files = glob.glob(os.path.join(input_dir, "**", "*.pt"), recursive=True)
    print(f"Found {len(files)} files.")

    # Sort files to potentially group by city implicitly if naming conventions allow
    # (Optional, but good practice)
    files.sort()

    maps = {}
    stats = defaultdict(int)

    for f_path in tqdm(files, desc="Patching"):
        try:
            # 1. Load
            try: data = torch.load(f_path, weights_only=False)
            except: data = torch.load(f_path)
            
            # 2. Get Attributes
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

            # 3. Load Map
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

            # 5. RUN GEOMETRIC LOGIC
            global_traj = ego_to_global(traj, origin, theta)
            cat, lane_key = classify_maneuver(nusc_map, global_traj)

            # 6. PICK RANDOM TEMPLATES (This is the added variety)
            man_text = random.choice(TEMPLATES[cat])
            lane_text = random.choice(LANE_TEMPLATES[lane_key])

            # 7. Preserve VLM Scene Description
            scene_desc = old_caps.get('scene_description', "Driving in an urban environment.")
            if len(scene_desc) < 5: scene_desc = "Driving in an urban environment."

            # 8. Construct Final Data
            full_caption = f"{man_text} {lane_text} {scene_desc}"
            
            new_caption_dict = {
                "maneuver_type": man_text,
                "lane_status": lane_text,
                "scene_description": scene_desc,
                "category": cat # Storing the raw class is useful for metrics
            }

            # 9. Save
            if isinstance(data, dict):
                data['caption_dict'] = new_caption_dict
                data['maneuver_category'] = cat
                data['scene_description'] = full_caption
            else:
                data.caption_dict = new_caption_dict
                data.maneuver_category = cat
                data.scene_description = full_caption

            torch.save(data, f_path)
            stats[cat] += 1
            
        except Exception as e:
            pass

    print("\n=== PATCHING COMPLETE ===")
    print("New Distribution:")
    total = sum(stats.values())
    for k, v in stats.items():
        print(f"{k:<20}: {v} ({v/total*100:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Folder containing .pt files")
    parser.add_argument("--dataroot", default="./", help="NuScenes root with /maps folder")
    args = parser.parse_args()
    
    process_dataset(args.input_dir, args.dataroot)
