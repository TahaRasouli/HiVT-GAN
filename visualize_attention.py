import torch
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import sys
from models.hivt_x import HiVTX
from datamodules.nuscenes_datamodule import NuScenesHiVTDataModule

# --- CONFIGURATION ---
CKPT_PATH = "/mount/arbeitsdaten/studenten4/rasoulta/HiVT-GAN/lightning_logs/version_54/checkpoints/epoch=29-step=8040.ckpt"
BACKBONE_PATH = "/mount/studenten/projects/rasoulta/checkpoints/vae-gan-baseline/checkpoints/epoch=45-step=60812.ckpt"
DATA_ROOT = "/mount/studenten/projects/rasoulta/dataset"
NUSCENES_MAP_ROOT = "/mount/arbeitsdaten/analysis/rasoulta/nuscenes/nuscenes_meta"

try:
    from nuscenes.map_expansion.map_api import NuScenesMap
except ImportError:
    print("Error: nuscenes-devkit not installed. Run 'pip install nuscenes-devkit'")
    sys.exit(1)

MAP_CACHE = {}

def transform_agent_to_global(trajectory_local, origin, theta):
    """
    Transforms agent-centric coordinates to global map coordinates.
    """
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    trajectory_global = trajectory_local @ R.T + origin
    return trajectory_global

def visualize_sample(model, batch, sample_idx):
    # 1. Unpack Metadata
    city = batch.city[0]
    origin = batch.origin[0].numpy()
    theta = batch.theta[0].item()
    gt_ids = batch.caption_ids[0]
    gt_text = model.tokenizer.decode(gt_ids)

    # 2. Filter: Only plot turns
    if "right" not in gt_text and "left" not in gt_text:
        return False

    # 3. Load Map
    if city not in MAP_CACHE:
        try:
            MAP_CACHE[city] = NuScenesMap(dataroot=NUSCENES_MAP_ROOT, map_name=city)
        except Exception as e:
            print(f"Failed to load map for {city}: {e}")
            return False
    nusc_map = MAP_CACHE[city]

    # 4. Define Patch (Global Coordinates)
    radius = 75
    my_patch = (origin[0] - radius, origin[1] - radius, origin[0] + radius, origin[1] + radius)
    
    # 5. Render Map Patch (Using Official API correctly)
    # The API returns the Figure and Axes. We do NOT pass 'ax' as an argument.
    layers = ['drivable_area', 'lane', 'lane_divider', 'road_divider']
    
    try:
        # Note: figsize determines the resolution/size of the output image
        fig, ax = nusc_map.render_map_patch(my_patch, layers, figsize=(12, 12))
    except TypeError:
        # Fallback if specific version doesn't support figsize in args (rare)
        fig, ax = nusc_map.render_map_patch(my_patch, layers)

    # 6. Run Model Inference
    data = batch.to(model.device)
    with torch.no_grad():
        global_embed, _ = model._get_ego_features(data)
        traj_input = data.y[0].unsqueeze(0) 
        
        logits = model.captioner(global_embed, traj_input, captions=None, return_attn=False)
        pred_ids = logits.argmax(dim=-1)[0]
        pred_text = model.tokenizer.decode(pred_ids)

        # Get GT Trajectory (Local)
        gt_traj_local = data.y[0].cpu().numpy()
    
    # 7. Transform to Global
    gt_traj_global = transform_agent_to_global(gt_traj_local, origin, theta)
    
    # 8. Plot Trajectory on the API-generated axes
    # We use 'ax' returned by render_map_patch
    ax.plot(gt_traj_global[:, 0], gt_traj_global[:, 1], color='#1f77b4', linewidth=5, label='Trajectory', zorder=100)
    ax.scatter(gt_traj_global[0, 0], gt_traj_global[0, 1], color='green', s=200, edgecolors='black', label='Start', zorder=101)
    ax.scatter(gt_traj_global[-1, 0], gt_traj_global[-1, 1], color='red', s=200, edgecolors='black', label='End', zorder=101)

    # 9. Polish and Save
    ax.legend(loc='upper right', fontsize=12)
    ax.set_title(f"GT: {gt_text}\nPred: {pred_text}", fontsize=14, pad=20)
    
    # Tight zoom on the car (optional override of the broad patch)
    margin = 40
    ax.set_xlim(origin[0] - margin, origin[0] + margin)
    ax.set_ylim(origin[1] - margin, origin[1] + margin)

    save_path = f"official_map_viz_{sample_idx}.png"
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()
    return True

def main():
    # Load Model
    with open("vocab.json") as f: vocab = json.load(f)
    print("Loading HiVTX Model...")
    model = HiVTX.load_from_checkpoint(CKPT_PATH, cvae_gan_ckpt=BACKBONE_PATH, vocab_size=len(vocab), strict=False)
    model.eval().cuda()

    # Load Data
    print("Loading NuScenes Data...")
    datamodule = NuScenesHiVTDataModule(root=DATA_ROOT, split_file="balanced_splits.json", val_batch_size=1, shuffle=True)
    datamodule.setup()
    loader = datamodule.val_dataloader()

    # Loop
    count = 0
    for i, batch in enumerate(loader):
        if count >= 5: break
        if visualize_sample(model, batch, i):
            count += 1

if __name__ == "__main__":
    main()