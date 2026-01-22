import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os

# Import your actual dataset class
from datasets.nuscenes_dataset import NuScenesHiVTDataset

def check_batch(batch, batch_idx):
    try:
        # --- 1. Define Batch Boundaries ---
        # In a batch, x is stacked: [Total_Nodes, T, D]
        # lane_vectors is stacked: [Total_Lanes, 2]
        
        total_nodes = 0
        if hasattr(batch, 'x') and batch.x is not None:
            total_nodes = batch.x.shape[0]
        else:
            total_nodes = batch.num_nodes

        total_lanes = 0
        if hasattr(batch, 'lane_vectors') and batch.lane_vectors is not None:
            total_lanes = batch.lane_vectors.shape[0]

        # --- 2. Check Lane-Actor Index (Bipartite) ---
        if hasattr(batch, 'lane_actor_index') and batch.lane_actor_index.numel() > 0:
            lai = batch.lane_actor_index
            
            # Row 0: Lane Indices. Must be < Total Lanes
            max_lane_idx = lai[0].max().item()
            if max_lane_idx >= total_lanes:
                print(f"\n[FAIL] Batch {batch_idx}: lane_actor_index[0] max {max_lane_idx} >= total_lanes {total_lanes}")
                print(f"This implies __inc__ returned a value too large for lanes.")
                return False
                
            # Row 1: Actor Indices. Must be < Total Nodes
            max_actor_idx = lai[1].max().item()
            if max_actor_idx >= total_nodes:
                print(f"\n[FAIL] Batch {batch_idx}: lane_actor_index[1] max {max_actor_idx} >= total_nodes {total_nodes}")
                return False

        # --- 3. Check Edge Index (Agent-Agent) ---
        if hasattr(batch, 'edge_index') and batch.edge_index.numel() > 0:
            ei = batch.edge_index
            max_edge_idx = ei.max().item()
            
            if max_edge_idx >= total_nodes:
                print(f"\n[FAIL] Batch {batch_idx}: edge_index max {max_edge_idx} >= total_nodes {total_nodes}")
                return False

        # --- 4. Check Position Lookups ---
        # The model does: positions[edge_index]
        # If positions tensor is smaller than x (due to bad concat?), this fails.
        if hasattr(batch, 'positions') and batch.positions is not None:
            if batch.positions.shape[0] != total_nodes:
                print(f"\n[FAIL] Batch {batch_idx}: positions size {batch.positions.shape[0]} != total_nodes {total_nodes}")
                return False

        # --- 5. Check Rotate Mat ---
        if hasattr(batch, 'rotate_mat') and batch.rotate_mat is not None:
            if batch.rotate_mat.shape[0] != total_nodes:
                print(f"\n[FAIL] Batch {batch_idx}: rotate_mat size {batch.rotate_mat.shape[0]} != total_nodes {total_nodes}")
                return False

        return True

    except Exception as e:
        print(f"\n[CRASH] Batch {batch_idx} caused python error: {e}")
        return False

def main():
    # Update this path to your actual processed data root
    root = "/mount/studenten/projects/rasoulta/dataset/processed"
    split_file = os.path.join(root, "split_datas.json")
    
    print("Initializing Dataset...")
    # We set max_samples=None to check everything
    dataset = NuScenesHiVTDataset(
        split_file=split_file,
        split="train",
        root=root,
        tokenizer=None 
    )
    
    print(f"Dataset length: {len(dataset)}")
    print("Creating DataLoader...")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=64, 
        shuffle=False, 
        num_workers=4,
        collate_fn=NuScenesHiVTDataset.collate_fn
    )
    
    print("Scanning Batches...")
    for i, batch in enumerate(tqdm(dataloader)):
        if not check_batch(batch, i):
            print("\nDiagnosis finished with FAIL.")
            return
            
    print("\nAll batches passed! The issue is likely INSIDE the model forward pass (e.g. subgraph generation).")

if __name__ == "__main__":
    main()