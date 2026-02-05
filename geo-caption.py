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
        "Completing a leftward turn to change direction.",
        "Performing a left turn maneuver at the junction."
    ],
    'turn_right': [
        "The vehicle is performing a right turn at the junction.",
        "Executing a right-hand turn into the intersecting lane.",
        "The car is turning right to exit the current road segment.",
        "Making a right turn to transition onto the cross-street.",
        "Completing a rightward turn at the intersection."
    ],
    'u_turn': [
        "The vehicle is performing a full U-turn to reverse direction.",
        "Executing a 180-degree turn to head back the opposite way.",
        "The ego is completing a U-turn maneuver.",
        "Reversing direction via a controlled U-turn."
    ],
    'stationary': [
        "The vehicle remains stationary at its current position.",
        "The ego is stopped and not currently in motion.",
        "Maintaining a full stop within the lane.",
        "The vehicle is idling at its current location.",
        "Currently stationary and awaiting further movement."
    ],
    'off_map': [
        "The vehicle is moving through an unmapped or off-road area.",
        "The ego is navigating a region without clearly defined lane data.",
        "Driving through a segment where map markings are unavailable."
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
        # 1. Hardcoded index (based on your preprocessor saving ego at 0)
        ego_idx = 0 
        
        # 2. Extract relative future from 'y'
        # y is [N, 30, 2]. HiVT future is relative to origin (t=19)
        future_traj = data.y[ego_idx].numpy() 
        final_rel_pos = future_traj[-1]
        total_dist = np.linalg.norm(final_rel_pos)
        
        if total_dist < 0.7: # Slightly higher threshold for noise
            return random.choice(TEMPLATES['stationary']), 'stationary'

        # 3. Global Transformation Logic
        # We need the ego's position at t=19 (which is the origin (0,0) in local frame)
        # and t=49 (which is the last point in 'y')
        origin = data.origin.numpy().flatten()
        theta = float(data.theta)
        
        c, s = np.cos(theta), np.sin(theta)
        rot_mat = np.array([[c, -s], [s, c]])
        
        global_current = origin # At t=19, local pos is (0,0), so global is just origin
        global_end = (final_rel_pos @ rot_mat.T) + origin
        
        # 4. Heading Change (Angle of the displacement vector)
        heading_change = np.degrees(np.arctan2(final_rel_pos[1], final_rel_pos[0]))
        
        # 5. Map Query
        nmap = get_map(str(data.city))
        start_lane = nmap.get_closest_lane(global_current[0], global_current[1], radius=3.0)
        end_lane = nmap.get_closest_lane(global_end[0], global_end[1], radius=3.0)
        
        if not start_lane or not end_lane:
            return random.choice(TEMPLATES['off_map']), 'off_map'

        # 6. Successor/Neighbor Logic
        successors = nmap.get_outgoing_lane_ids(start_lane)
        adj = nmap.get_adjacency_list(start_lane, 'lane')

        if start_lane == end_lane or end_lane in successors:
            return random.choice(TEMPLATES['follow']), 'follow'
        elif end_lane in adj['left']:
            return random.choice(TEMPLATES['lane_change_left']), 'lane_change_left'
        elif end_lane in adj['right']:
            return random.choice(TEMPLATES['lane_change_right']), 'lane_change_right'
        else:
            if abs(heading_change) > 140:
                return random.choice(TEMPLATES['u_turn']), 'u_turn'
            elif heading_change > 20: 
                return random.choice(TEMPLATES['turn_left']), 'turn_left'
            elif heading_change < -20: 
                return random.choice(TEMPLATES['turn_right']), 'turn_right'
            else:
                return random.choice(TEMPLATES['follow']), 'follow'
            
    except Exception as e:
        # For debugging, you can print(e) here once
        return "The vehicle is in motion.", 'error'

# -----------------------
# EXECUTION
# -----------------------
pt_files = [f for f in os.listdir(IN_DIR) if f.endswith('.pt')]

for filename in tqdm(pt_files):
    data = torch.load(os.path.join(IN_DIR, filename))
    caption_str, m_type = generate_ego_caption(data)
    
    # Use .__setattr__ to be safe with TemporalData objects
    data.caption = caption_str
    data.maneuver_type = m_type
    
    torch.save(data, os.path.join(OUT_DIR, filename))
    stats[m_type] += 1

print(f"\nFinished! Final Stats: {stats}")