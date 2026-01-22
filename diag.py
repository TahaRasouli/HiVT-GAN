import os
import json
import torch
import sys
from tqdm import tqdm

def check_tensor(data, key, min_val=None, max_val=None, check_dim_0_size=None):
    if not hasattr(data, key):
        return
    
    tensor = getattr(data, key)
    if tensor is None: return
    if not torch.is_tensor(tensor): return

    # Check 1: Dimensions
    if check_dim_0_size is not None:
        if tensor.shape[0] != check_dim_0_size:
            raise ValueError(f"'{key}' size {tensor.shape[0]} does not match expected size {check_dim_0_size}")

    # Check 2: Values (for indices)
    if tensor.numel() > 0:
        if min_val is not None:
            if tensor.min() < min_val:
                raise ValueError(f"'{key}' contains value {tensor.min()} which is < {min_val}")
        if max_val is not None:
            if tensor.max() >= max_val:
                raise ValueError(f"'{key}' contains value {tensor.max()} which is >= limit {max_val}")

def diagnose(root):
    split_file = os.path.join(root, "/mount/studenten/projects/rasoulta/dataset/tmpl-captioned/balanced_splits.json")
    if not os.path.exists(split_file):
        print(f"Error: Could not find {split_file}")
        return

    with open(split_file, 'r') as f:
        splits = json.load(f)
    
    files = splits.get('train', [])
    print(f"Scanning {len(files)} training files...")

    bad_files = 0
    
    for i, path in enumerate(tqdm(files)):
        try:
            # Load raw data without any sanitization
            data = torch.load(path)
            
            # --- 1. Establish Ground Truth Sizes ---
            # We trust 'x' (Agent Features) as the source of truth for num_nodes
            if not hasattr(data, 'x'):
                raise ValueError("Missing 'x' tensor (Agent features)")
            
            num_nodes = data.x.shape[0]
            
            # Trust 'lane_vectors' for num_lanes
            num_lanes = 0
            if hasattr(data, 'lane_vectors') and data.lane_vectors is not None:
                num_lanes = data.lane_vectors.shape[0]

            # --- 2. Check Node Attribute Consistency ---
            # All these must match num_nodes
            check_tensor(data, 'positions', check_dim_0_size=num_nodes)
            check_tensor(data, 'padding_mask', check_dim_0_size=num_nodes)
            check_tensor(data, 'bos_mask', check_dim_0_size=num_nodes)
            check_tensor(data, 'rotate_angles', check_dim_0_size=num_nodes)
            
            # --- 3. Check AV Index ---
            check_tensor(data, 'av_index', min_val=0, max_val=num_nodes)

            # --- 4. Check Edge Connectivity (The likely crasher) ---
            
            # Lane-Actor Index: [2, E] -> Row 0 refers to Lanes, Row 1 refers to Actors
            if hasattr(data, 'lane_actor_index') and data.lane_actor_index.numel() > 0:
                lai = data.lane_actor_index
                if lai.dim() == 1: lai = lai.reshape(2, 1)
                
                # Check Row 0 (Lane Indices)
                if num_lanes == 0:
                    raise ValueError(f"lane_actor_index exists but num_lanes is 0")
                
                if lai[0].max() >= num_lanes:
                    raise ValueError(f"lane_actor_index refers to lane {lai[0].max()}, but only {num_lanes} lanes exist")
                
                # Check Row 1 (Actor Indices)
                if lai[1].max() >= num_nodes:
                    raise ValueError(f"lane_actor_index refers to node {lai[1].max()}, but only {num_nodes} nodes exist")

            # Edge Index (Agent-Agent): [2, E] -> Both rows refer to Actors
            if hasattr(data, 'edge_index') and data.edge_index.numel() > 0:
                ei = data.edge_index
                if ei.dim() == 1: ei = ei.reshape(2, 1)
                
                if ei.max() >= num_nodes:
                    raise ValueError(f"edge_index refers to node {ei.max()}, but only {num_nodes} nodes exist")

            # --- 5. Check Categorical Bounds (Embedding lookups) ---
            check_tensor(data, 'turn_directions', min_val=0, max_val=3) # 0,1,2 allowed
            check_tensor(data, 'is_intersections', min_val=0, max_val=2) # 0,1 allowed
            check_tensor(data, 'traffic_controls', min_val=0, max_val=2) # 0,1 allowed

        except Exception as e:
            print(f"\n[FATAL] Corrupt File Found: {path}")
            print(f"Reason: {e}")
            bad_files += 1
            # Remove the 'break' below if you want to see ALL bad files
            break 

    if bad_files == 0:
        print("\nAll files passed CPU validation. The issue might be in model logic (temporal edges).")
    else:
        print(f"\nFound {bad_files} corrupt files.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    args = parser.parse_args()
    
    diagnose(args.root)