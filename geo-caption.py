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
OUT_DIR = '/mount/studenten/projects/rasoulta/dataset/train_with_captions'
NUSCENES_ROOT = '/mount/arbeitsdaten/analysis/rasoulta/nuscenes/nuscenes_meta'

os.makedirs(OUT_DIR, exist_ok=True)
map_cache = {}
stats = Counter()

# -----------------------
# TEMPLATE DEFINITIONS
# -----------------------
TEMPLATES = {
    'follow': [
        "The vehicle is maintaining its path within the current lane.",
        "The ego vehicle continues driving straight along the lane.",
        "The car is staying in its lane and advancing forward.",
        "Following the established lane path consistently.",
        "The vehicle remains centered and follows the road forward."
    ],
    'lane_change_left': [
        "The vehicle is performing a smooth lane change to the left.",
        "The ego is merging into the adjacent lane on the left side.",
        "Initiating a leftward maneuver to switch lanes.",
        "The car shifts toward the left-hand lane.",
        "Executing a lane change maneuver to the left."
    ],
    'lane_change_right': [
        "The vehicle is shifting into the right-hand lane.",
        "Performing a lane change to the right to exit the current path.",
        "The ego vehicle is merging right into the neighboring lane.",
        "The car maneuvers toward the right side to switch lanes.",
        "Initiating a rightward lane change."
    ],
    'turn_left': [
        "The vehicle is executing a left turn at the intersection.",
        "Making a left-hand turn to transition to the crossing street.",
        "The car is veering left to follow the intersection's path.",
        "Completing a leftward turn to change direction."
    ],
    'turn_right': [
        "The vehicle is performing a right turn at the junction.",
        "Executing a right-hand turn into the intersecting lane.",
        "The car is turning right to exit the current road segment.",
        "Making a right turn to transition onto the cross-street."
    ],
    'u_turn': [
        "The vehicle is performing a full U-turn to reverse direction.",
        "Executing a 180-degree turn to head back the opposite way.",
        "The ego is completing a U-turn maneuver."
    ],
    'stationary': [
        "The vehicle remains stationary at its current position.",
        "The ego is stopped and not currently in motion.",
        "Maintaining a full stop within the lane.",
        "The vehicle is idling at its current location."
    ],
    'off_map': [
        "The vehicle is moving through an unmapped or off-road area.",
        "The ego is navigating a region without clearly defined lane data."
    ]
}

# -----------------------
# HELPER FUNCTIONS
# -----------------------
def get_map(city_name):
    if city_name not in map_cache:
        map_cache[city_name] = NuScenesMap(NUSCENES_ROOT, city_name)
    return map_cache[city_name]

def generate_ego_caption(data):
    try:
        # 1. HiVT uses 0 as ego index
        ego_idx = 0 
        
        # 2. Extract relative future (local frame)
        # y is [N, 30, 2]. t=19 is the origin (0,0)
        future_traj = data.y[ego_idx].numpy() 
        final_rel_pos = future_traj[-1]
        total_disp = np.linalg.norm(final_rel_pos)
        
        if total_disp < 0.7:
            return random.choice(TEMPLATES['stationary']), 'stationary', total_disp, 0.0

        # 3. Global Transformation
        origin = data.origin.numpy().flatten()
        theta = data.theta.item() if torch.is_tensor(data.theta) else float(data.theta)
        
        c, s = np.cos(theta), np.sin(theta)
        rot_mat = np.array([[c, -s], [s, c]])
        
        global_current = origin 
        global_end = (final_rel_pos @ rot_mat.T) + origin
        
        # 4. Local Heading Change for classification
        heading_change = np.degrees(np.arctan2(final_rel_pos[1], final_rel_pos[0]))
        
        # 5. Map Query with robust city extraction
        city_name = str(data.city[0]) if isinstance(data.city, list) else str(data.city)
        nmap = get_map(city_name)
        
        start_lane = nmap.get_closest_lane(global_current[0], global_current[1], radius=5.0)
        end_lane = nmap.get_closest_lane(global_end[0], global_end[1], radius=5.0)
        
        if not start_lane or not end_lane:
            return random.choice(TEMPLATES['off_map']), 'off_map', total_disp, heading_change

        # 6. Connectivity Logic
        successors = nmap.get_outgoing_lane_ids(start_lane)
        
        # Manually check for left/right neighbors instead of get_adjacency_list
        left_lanes = nmap.get_left_lanes(start_lane, 1) # Gets lanes to the left
        right_lanes = nmap.get_right_lanes(start_lane, 1) # Gets lanes to the right

        if start_lane == end_lane or end_lane in successors:
            return random.choice(TEMPLATES['follow']), 'follow', total_disp, heading_change
        
        elif end_lane in left_lanes:
            return random.choice(TEMPLATES['lane_change_left']), 'lane_change_left', total_disp, heading_change
            
        elif end_lane in right_lanes:
            return random.choice(TEMPLATES['lane_change_right']), 'lane_change_right', total_disp, heading_change
        
        else:
            # Intersection Turn Logic remains the same
            if abs(heading_change) > 140:
                m_type = 'u_turn'
            elif heading_change > 15: 
                m_type = 'turn_left'
            elif heading_change < -15: 
                m_type = 'turn_right'
            else:
                m_type = 'follow'
            return random.choice(TEMPLATES[m_type]), m_type, total_disp, heading_change
            
    except Exception as e:
        # Diagnostic print for the first few errors
        if stats['error'] < 5:
            print(f"\n[DEBUG] Error processing file: {e}")
        return "The vehicle is in motion.", 'error', 0.0, 0.0

# -----------------------
# EXECUTION & INTEGRATED SANITY CHECK
# -----------------------
pt_files = sorted([f for f in os.listdir(IN_DIR) if f.endswith('.pt')])

print(f"Starting processing of {len(pt_files)} files...")

for i, filename in enumerate(tqdm(pt_files)):
    data = torch.load(os.path.join(IN_DIR, filename))
    
    caption_str, m_type, disp, angle = generate_ego_caption(data)
    
    # Save attributes
    data.caption = caption_str
    data.maneuver_type = m_type
    
    torch.save(data, os.path.join(OUT_DIR, filename))
    stats[m_type] += 1

    # SANITY CHECK: Every 50 files, print a report of the current file
    if (i + 1) % 50 == 0:
        print(f"\n--- Sanity Check (File {i+1}) ---")
        print(f"File: {filename}")
        print(f"Maneuver: {m_type} | Caption: {caption_str}")
        print(f"Metrics: Disp: {disp:.2f}m | Local Angle: {angle:.2f}°")
        
        # Geometric Logic Verification
        if m_type == 'turn_left' and angle < 5:
            print("   [!] WARNING: Classified Left Turn but local angle is low.")
        elif m_type == 'stationary' and disp > 1.0:
            print("   [!] WARNING: Classified Stationary but vehicle moved.")
        print("-" * 40)

# Final Summary
print("\n--- Final Processing Stats ---")
for k, v in stats.items():
    print(f"{k:<18}: {v} ({100*v/len(pt_files):.2f}%)")