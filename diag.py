import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.nuscenes_dataset import NuScenesHiVTDataset

def validate_val_set(root):
    split_file = os.path.join(root, "balanced_splits.json")
    
    print("Initializing Validation Dataset...")
    # We explicitly load the 'val' split
    dataset = NuScenesHiVTDataset(
        split_file=split_file,
        split="val",
        root=root,
        tokenizer=None 
    )
    
    print(f"Validation set size: {len(dataset)}")
    
    # Use the exact batch size causing the crash
    dataloader = DataLoader(
        dataset, 
        batch_size=64, 
        shuffle=False, 
        num_workers=4,
        collate_fn=NuScenesHiVTDataset.collate_fn
    )
    
    print("Scanning Validation Batches...")
    for i, batch in enumerate(tqdm(dataloader)):
        try:
            # --- 1. Check Node Counts ---
            # x and num_nodes must align
            if hasattr(batch, 'x') and batch.x is not None:
                if batch.x.shape[0] != batch.num_nodes:
                    print(f"\n[FAIL] Batch {i}: x.shape[0] ({batch.x.shape[0]}) != num_nodes ({batch.num_nodes})")
                    return

            # --- 2. Check Edge Index (Agent-Agent) ---
            if hasattr(batch, 'edge_index') and batch.edge_index.numel() > 0:
                max_idx = batch.edge_index.max().item()
                if max_idx >= batch.num_nodes:
                    print(f"\n[FAIL] Batch {i}: edge_index max ({max_idx}) >= num_nodes ({batch.num_nodes})")
                    print("This means ghost edges exist pointing to non-existent nodes.")
                    return

            # --- 3. Check Lane Counts ---
            # lane_vectors must align with lane attributes
            if hasattr(batch, 'lane_vectors') and batch.lane_vectors is not None:
                num_lanes = batch.lane_vectors.shape[0]
                
                # Check attributes match lane count
                attrs = ['is_intersections', 'turn_directions', 'traffic_controls']
                for key in attrs:
                    if hasattr(batch, key) and getattr(batch, key) is not None:
                        attr_len = getattr(batch, key).shape[0]
                        if attr_len != num_lanes:
                            print(f"\n[FAIL] Batch {i}: {key} length ({attr_len}) != num_lanes ({num_lanes})")
                            return

                # --- 4. Check Lane-Actor Index ---
                if hasattr(batch, 'lane_actor_index') and batch.lane_actor_index.numel() > 0:
                    lai = batch.lane_actor_index
                    # Row 0: Lane Indices
                    max_lane = lai[0].max().item()
                    if max_lane >= num_lanes:
                        print(f"\n[FAIL] Batch {i}: lane_actor_index refers to lane {max_lane}, but only {num_lanes} lanes exist.")
                        return
                    
                    # Row 1: Actor Indices
                    max_actor = lai[1].max().item()
                    if max_actor >= batch.num_nodes:
                        print(f"\n[FAIL] Batch {i}: lane_actor_index refers to actor {max_actor}, but only {batch.num_nodes} nodes exist.")
                        return

        except Exception as e:
            print(f"\n[CRASH] Batch {i} triggered python error: {e}")
            return

    print("\nAll Validation batches passed CPU checks.")
    print("If this passed, the issue is likely in the Model's Temporal Logic (historical_steps mismatch).")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help="Path to processed dataset root")
    args = parser.parse_args()
    
    validate_val_set(args.root)