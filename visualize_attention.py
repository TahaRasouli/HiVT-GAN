import torch
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from models.hivt_x import HiVTX
from datamodules.nuscenes_datamodule import NuScenesHiVTDataModule

# --- CONFIGURATION ---
CKPT_PATH = "/mount/studenten/projects/rasoulta/checkpoints/x_baseline/checkpoints/epoch=29-step=8040.ckpt" # UPDATE THIS!
BACKBONE_PATH = "/mount/studenten/projects/rasoulta/checkpoints/vae-gan-baseline/checkpoints/epoch=45-step=60812.ckpt"
DATA_ROOT = "/mount/studenten/projects/rasoulta/dataset"


def diagnose():
    # Load Data
    datamodule = NuScenesHiVTDataModule(
        root=DATA_ROOT, 
        split_file="balanced_splits.json",
        val_batch_size=1, 
        shuffle=True
    )
    datamodule.setup()
    loader = datamodule.val_dataloader()
    
    # Get one batch
    batch = next(iter(loader))
    print("\n=== DATA KEYS ===")
    print(batch.keys)
    
    print("\n=== LANE VECTOR ANALYSIS ===")
    if hasattr(batch, 'lane_vectors'):
        lv = batch.lane_vectors
        print(f"Shape: {lv.shape}")
        print(f"First 5 rows:\n{lv[:5]}")
        print(f"Min: {lv.min(dim=0)[0]}")
        print(f"Max: {lv.max(dim=0)[0]}")
    
    print("\n=== CHECKING FOR POSITIONS ===")
    potential_keys = ['lane_positions', 'lane_centers', 'lane_points', 'positions']
    for k in potential_keys:
        if hasattr(batch, k):
            print(f"FOUND {k}!")
            val = getattr(batch, k)
            print(f"Shape: {val.shape}")
            print(f"First 5 rows:\n{val[:5]}")
        else:
            print(f"Did NOT find {k}")

if __name__ == "__main__":
    diagnose()