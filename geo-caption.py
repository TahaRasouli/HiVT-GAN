import os
import torch
import numpy as np
import random
from tqdm import tqdm
from nuscenes.map_expansion.map_api import NuScenesMap
from collections import Counter

# -----------------------
# CONFIGURATION
# -----------------------
IN_DIR = '/mount/studenten/projects/rasoulta/dataset/train_processed'
OUT_DIR = '/mount/studenten/projects/rasoulta/dataset/train_with_captions_v2'
NUSCENES_ROOT = '/mount/arbeitsdaten/analysis/rasoulta/nuscenes/nuscenes_meta'

os.makedirs(OUT_DIR, exist_ok=True)
map_cache = {}
stats = Counter()

# -----------------------
# TEMPLATES
# -----------------------
TEMPLATES = {
    'follow': [
        "The vehicle is maintaining its path within the current lane.",
        "The ego vehicle continues driving straight along the lane.",
        "Following the established lane path consistently."
    ],
    'lane_change_left': [
        "The vehicle is performing a smooth lane change to the left.",
        "The ego is merging into the adjacent lane on the left side."
    ],
    'lane_change_right': [
        "The vehicle is shifting into the right-hand lane.",
        "The ego vehicle is merging right into the neighboring lane."
    ],
    'turn_left': [
        "The vehicle is executing a left turn at the intersection.",
        "Making a left-hand turn to transition to the crossing street.",
        "The car is veering left to follow the intersection's path."
    ],
    'turn_right': [
        "The vehicle is performing a right turn at the junction.",
        "Executing a right-hand turn into the intersecting lane.",
        "The car is turning right to exit the current road segment."
    ],
    'u_turn': [
        "The vehicle is performing a full U-turn to reverse direction.",
        "Executing a 180-degree turn to head back the opposite way."
    ],
    'stationary': [
        "The vehicle remains stationary at its current position.",
        "The ego is stopped and not currently in motion."
    ],
    'off_map': [
        "The vehicle is moving through an unmapped area.",
        "The ego is navigating a region without clearly defined lane data."
    ]
}

def get_map(city_name):
    if city_name not in map_cache:
        map_cache[city_name] = NuScenesMap(NUSCENES_ROOT, city_name)
    return map_cache[city_name]

def generate_ego_caption(data):
    try:
        ego_idx = 0 
        future_traj = data.y[ego_idx].numpy() 
        final_rel_pos = future_traj[-1]
        total_disp = np.linalg.norm(final_rel_pos)
        
        # 1. Stationary Check
        if total_disp < 0.8:
            return random.choice(TEMPLATES['stationary']), 'stationary', total_disp, 0.0

        # 2. Transformation
        origin = data.origin.numpy().flatten()
        theta = data.theta.item() if torch.is_tensor(data.theta) else float(data.theta)
        c, s = np.cos(theta), np.sin(theta)
        rot_mat = np.array([[c, -s], [s, c]])
        
        global_curr = origin 
        global_end = (final_rel_pos @ rot_mat.T) + origin
        
        # 3. Heading Delta (Positive = Left, Negative = Right)
        # Based on local frame where forward is X-axis
        angle = np.degrees(np.arctan2(final_rel_pos[1], final_rel_pos[0]))
        
        # 4. Map Query
        city_name = str(data.city[0]) if isinstance(data.city, list) else str(data.city)
        nmap = get_map(city_name)
        
        start_lane = nmap.get_closest_lane(global_curr[0], global_curr[1], radius=5.0)
        end_lane = nmap.get_closest_lane(global_end[0], global_end[1], radius=5.0)
        
        if not start_lane or not end_lane:
            return random.choice(TEMPLATES['off_map']), 'off_map', total_disp, angle

        # 5. PRIORITIZED LOGIC
        successors = nmap.get_outgoing_lane_ids(start_lane)
        
        # A. Catch U-Turns first
        if abs(angle) > 140:
            m_type = 'u_turn'
        
        # B. Catch Significant Turns (regardless of successor status)
        elif angle > 30:
            m_type = 'turn_left'
        elif angle < -30:
            m_type = 'turn_right'
            
        # C. Handle Moderate Angles
        elif 10 < angle <= 30:
            # If it's a successor, it's likely a curved road or soft turn
            # If NOT a successor, it's a lane change
            m_type = 'turn_left' if end_lane in successors or start_lane == end_lane else 'lane_change_left'
            
        elif -30 <= angle < -10:
            m_type = 'turn_right' if end_lane in successors or start_lane == end_lane else 'lane_change_right'
            
        # D. Everything else is a Straight Follow
        else:
            m_type = 'follow'
                
        return random.choice(TEMPLATES[m_type]), m_type, total_disp, angle
            
    except Exception as e:
        return "The vehicle is in motion.", 'error', 0.0, 0.0

# -----------------------
# EXECUTION
# -----------------------
pt_files = sorted([f for f in os.listdir(IN_DIR) if f.endswith('.pt')])

print(f"Starting prioritized processing of {len(pt_files)} files...")

for i, filename in enumerate(tqdm(pt_files)):
    data = torch.load(os.path.join(IN_DIR, filename))
    caption_str, m_type, disp, angle = generate_ego_caption(data)
    
    data.caption = caption_str
    data.maneuver_type = m_type
    
    torch.save(data, os.path.join(OUT_DIR, filename))
    stats[m_type] += 1

    if (i + 1) % 50 == 0:
        print(f"\n[SANITY] {filename} | Type: {m_type:<15} | Ang: {angle:>6.2f}° | Disp: {disp:>5.2f}m")

print(f"\nUpdated Final Stats: {stats}")