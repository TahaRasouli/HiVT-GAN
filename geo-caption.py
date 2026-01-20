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
# 2. GEOMETRIC LOGIC (v5)
# =========================
def classify_maneuver(nusc_map, global_traj):
    """
    Returns (category, lane_key)
    """
    p_start = global_traj[0]
    p_mid   = global_traj[15]
    p_end   = global_traj[-1]

    # Vectors
    v_start = global_traj[5] - global_traj[0]
    v_end   = global_traj[-1] - global_traj[-6]
    
    # Heading
    angle_start = np.arctan2(v_start[1], v_start[0])
    angle_end = np.arctan2(v_end[1], v_end[0])
    delta_deg = np.degrees(angle_end - angle_start)
    delta_deg = (delta_deg + 180) % 360 - 180
    
    # Lateral Reference
    ref_vec = v_start / (np.linalg.norm(v_start) + 1e-6)
    left_vec = np.array([-ref_vec[1], ref_vec[0]])

    # Map Helper
    def get_lane(p):
        try: return nusc_map.layers_on_point(p[0], p[1]).get('lane', '')
        except: return ''

    l_start = get_lane(p_start)
    l_mid   = get_lane(p_mid)
    l_end   = get_lane(p_end)

    # Projection Lookahead
    if not l_end:
        v_last = global_traj[-1] - global_traj[-2]
        p_proj = p_end + (v_last * 5.0)
        l_proj = get_lane(p_proj)
        if l_proj: l_end = l_proj 

    # Connectivity Check
    def are_connected(id_a, id_b):
        if not id_a or not id_b: return False
        if id_a == id_b: return True
        outgoing = nusc_map.get_outgoing_lane_ids(id_a)
        if id_b in outgoing: return True
        incoming = nusc_map.get_incoming_lane_ids(id_b)
        if id_a in incoming: return True
        return False

    is_graph_connected = are_connected(l_start, l_end)

    # --- DECISION TREE ---

    # 1. Stationary
    if np.linalg.norm(p_end - p_start) < 2.0:
        return "Stationary Stop", "stop_action"

    # 2. Pure Geometry (Fallback)
    if not l_start and not l_end:
        if abs(delta_deg) > 135: return "U-Turn", "turn_action"
        if delta_deg > 25:       return "Left Turn", "turn_action"
        if delta_deg < -25:      return "Right Turn", "turn_action"
        return "Straight Drive", "maintain"

    # 3. Turns
    if abs(delta_deg) > 25:
        in_intersection = (not l_mid) or (not l_end)
        lane_changed_ids = (l_start != l_end)
        
        if in_intersection or (lane_changed_ids and not is_graph_connected):
            if abs(delta_deg) > 135: return "U-Turn", "turn_action"
            if delta_deg > 0:        return "Left Turn", "turn_action"
            else:                    return "Right Turn", "turn_action"
        else:
            return "Straight Drive", "maintain"

    # 4. Lane Changes
    if l_start and l_end and l_start != l_end:
        if is_graph_connected:
            return "Straight Drive", "maintain"
        
        lat_disp = np.dot(p_end - p_start, left_vec)
        
        # Check Mid
        if l_start != l_mid and not are_connected(l_start, l_mid):
             lat_disp_mid = np.dot(p_mid - p_start, left_vec)
             if lat_disp_mid > 0.5: return "Lane Change Left", "change_left"
             if lat_disp_mid < -0.5: return "Lane Change Right", "change_right"

        # Check End
        if lat_disp > 1.0:  return "Lane Change Left", "change_left"
        if lat_disp < -1.0: return "Lane Change Right", "change_right"

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