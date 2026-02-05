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
        nmap = get_map(data.city)
        ego_idx = data.agent_index
        
        # 1. Stationary Check (Displacement < 0.5m over 3 seconds)
        # Using relative displacement stored in 'y'
        future_traj = data.y[ego_idx].numpy() # [30, 2]
        total_dist = np.linalg.norm(future_traj[-1])
        if total_dist < 0.5:
            return random.choice(TEMPLATES['stationary']), 'stationary'

        # 2. Coordinate Prep
        origin = data.origin.numpy().flatten()
        theta = data.theta.item()
        c, s = np.cos(theta), np.sin(theta)
        rot_mat = np.array([[c, -s], [s, c]])
        
        pos_current = data.positions[ego_idx, 19].numpy()
        pos_end = data.positions[ego_idx, 49].numpy()
        
        global_current = (pos_current @ rot_mat.T) + origin
        global_end = (pos_end @ rot_mat.T) + origin
        
        # 3. Heading Delta (Determine Left/Right)
        # Displacement vector from t=19 to t=49
        v_future = pos_end - pos_current
        heading_change = np.degrees(np.arctan2(v_future[1], v_future[0]))
        
        # 4. Map Matching
        start_lane = nmap.get_closest_lane(global_current[0], global_current[1], radius=3.0)
        end_lane = nmap.get_closest_lane(global_end[0], global_end[1], radius=3.0)
        
        if not start_lane or not end_lane:
            return random.choice(TEMPLATES['off_map']), 'off_map'

        # 5. Semantic Logic Branching
        successors = nmap.get_outgoing_lane_ids(start_lane)
        adj = nmap.get_adjacency_list(start_lane, 'lane')

        # Scenario A: Staying in Lane
        if start_lane == end_lane or end_lane in successors:
            return random.choice(TEMPLATES['follow']), 'follow'
        
        # Scenario B: Lane Changes (Neighbors)
        elif end_lane in adj['left']:
            return random.choice(TEMPLATES['lane_change_left']), 'lane_change_left'
        elif end_lane in adj['right']:
            return random.choice(TEMPLATES['lane_change_right']), 'lane_change_right'
            
        # Scenario C: Turns (Significant heading change)
        else:
            if abs(heading_change) > 140:
                return random.choice(TEMPLATES['u_turn']), 'u_turn'
            elif heading_change > 15: # Positive is Left
                return random.choice(TEMPLATES['turn_left']), 'turn_left'
            elif heading_change < -15: # Negative is Right
                return random.choice(TEMPLATES['turn_right']), 'turn_right'
            else:
                return random.choice(TEMPLATES['follow']), 'follow'
            
    except Exception as e:
        return "The vehicle is in motion.", 'error'

# -----------------------
# EXECUTION
# -----------------------
pt_files = [f for f in os.listdir(IN_DIR) if f.endswith('.pt')]

print(f"Total files to process: {len(pt_files)}")

for i, filename in enumerate(tqdm(pt_files)):
    # Load original TemporalData
    file_path = os.path.join(IN_DIR, filename)
    data = torch.load(file_path)
    
    # Generate Label
    caption_str, maneuver_type = generate_ego_caption(data)
    
    # Inject into object (dynamic assignment)
    data.caption = caption_str
    data.maneuver_type = maneuver_type
    
    # Save to new location
    torch.save(data, os.path.join(OUT_DIR, filename))
    
    # Update Stats
    stats[maneuver_type] += 1
    
    # Periodic Log
    if (i + 1) % 5000 == 0:
        print(f"\nProgress Update {i+1}: {stats}")

# Final Summary
print("\n--- Processing Complete ---")
print(f"Destination: {OUT_DIR}")
for k, v in stats.items():
    print(f"{k}: {v} ({100*v/len(pt_files):.2f}%)")