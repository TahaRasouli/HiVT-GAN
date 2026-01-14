import torch
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import sys
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
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
    print("Error: nuscenes-devkit not installed.")
    sys.exit(1)

MAP_CACHE = {}

def get_map_features(city, origin, theta, radius=75):
    # Load Map
    if city not in MAP_CACHE:
        try:
            MAP_CACHE[city] = NuScenesMap(dataroot=NUSCENES_MAP_ROOT, map_name=city)
        except Exception as e:
            print(f"Map Load Error: {e}")
            return {}
    nusc_map = MAP_CACHE[city]

    # Define Patch
    x, y = origin[0], origin[1]
    patch_box = (x - radius, y - radius, x + radius, y + radius)
    
    features = {'drivable_area': [], 'dividers': [], 'centerlines': []}
    
    # 1. FETCH POLYGONS (Drivable Area)
    try:
        records = nusc_map.get_records_in_patch(patch_box, ['drivable_area'], mode='intersect')
        for token in records.get('drivable_area', []):
            poly = nusc_map.get('drivable_area', token)
            nodes = [nusc_map.get('node', t) for t in poly['exterior_node_tokens']]
            points = np.array([[n['x'], n['y']] for n in nodes])
            features['drivable_area'].append(transform_to_local(points, x, y, theta))
    except: pass

    # 2. FETCH DIVIDERS (Paint Lines)
    try:
        layers = ['lane_divider', 'road_divider']
        records = nusc_map.get_records_in_patch(patch_box, layers, mode='intersect')
        for layer in layers:
            for token in records.get(layer, []):
                line = nusc_map.get(layer, token)
                nodes = [nusc_map.get('node', t) for t in line['line_token']]
                points = np.array([[n['x'], n['y']] for n in nodes])
                features['dividers'].append(transform_to_local(points, x, y, theta))
    except: pass
    
    # 3. FETCH CENTERLINES (Actual Lanes)
    try:
        records = nusc_map.get_records_in_patch(patch_box, ['lane'], mode='intersect')
        for token in records.get('lane', []):
            pose_record = nusc_map.get_arcline_path(token)
            points = np.array(pose_record)
            features['centerlines'].append(transform_to_local(points, x, y, theta))
    except: pass

    return features

def transform_to_local(global_points, origin_x, origin_y, theta):
    # Translate and Rotate Global -> Agent Centric
    centered = global_points - np.array([origin_x, origin_y])
    c, s = np.cos(-theta), np.sin(-theta)
    R = np.array([[c, -s], [s, c]])
    return centered @ R.T

def plot_final_thesis_viz(trajectory, save_name, gt_text, pred_text, map_feats):
    # Large Square Figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # A. Plot Asphalt (Background)
    for poly in map_feats['drivable_area']:
        p = Polygon(poly, facecolor='#E8E8E8', edgecolor='none', alpha=0.5, zorder=0)
        ax.add_patch(p)

    # B. Plot Dividers (Black Lines)
    for line in map_feats['dividers']:
        ax.plot(line[:, 0], line[:, 1], color='black', linewidth=1.5, alpha=0.6, zorder=1)

    # C. Plot Lane Centerlines (Purple Dashed - The "Lanes")
    for line in map_feats['centerlines']:
        ax.plot(line[:, 0], line[:, 1], color='purple', linewidth=2.0, linestyle='--', alpha=0.5, zorder=2)

    # D. Plot Predicted Trajectory (Blue)
    traj = trajectory.cpu().numpy()
    ax.plot(traj[:, 0], traj[:, 1], color='#1f77b4', linewidth=6, zorder=10)
    
    # Start/End Markers
    ax.scatter(traj[0, 0], traj[0, 1], color='#2ca02c', s=300, edgecolors='black', linewidth=2, zorder=11)
    ax.scatter(traj[-1, 0], traj[-1, 1], color='#d62728', s=300, edgecolors='black', linewidth=2, zorder=11)

    # E. Manual Legend (Always visible)
    legend_elements = [
        Line2D([0], [0], color='#1f77b4', lw=5, label='Predicted Path'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c', markersize=12, markeredgecolor='k', label='Start'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', markersize=12, markeredgecolor='k', label='End'),
        Line2D([0], [0], color='purple', lw=2, linestyle='--', label='Lane Centerline'),
        Line2D([0], [0], color='black', lw=1.5, label='Lane Divider'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12, framealpha=1.0).set_zorder(102)

    # F. Smart Zoom (Context + 15m)
    x_min, x_max = traj[:, 0].min(), traj[:, 0].max()
    y_min, y_max = traj[:, 1].min(), traj[:, 1].max()
    margin = 15 
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_aspect('equal')
    
    # Title
    ax.set_title(f"GT: {gt_text}\nPred: {pred_text}", fontsize=13, pad=15, weight='bold')
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(save_name, dpi=150)
    print(f"Saved: {save_name}")
    plt.close()

def visualize():
    # Load Model
    with open("vocab.json") as f: vocab = json.load(f)
    print("Loading HiVTX Model...")
    model = HiVTX.load_from_checkpoint(CKPT_PATH, cvae_gan_ckpt=BACKBONE_PATH, vocab_size=len(vocab), strict=False)
    model.eval().cuda()

    # Load Data
    print("Loading Data...")
    datamodule = NuScenesHiVTDataModule(root=DATA_ROOT, split_file="balanced_splits.json", val_batch_size=1, shuffle=True)
    datamodule.setup()
    loader = datamodule.val_dataloader()
    
    print("Searching for Turn samples...")
    count = 0
    for batch in loader:
        if count >= 3: break
        
        gt_ids = batch.caption_ids[0]
        gt_text = model.tokenizer.decode(gt_ids)
        # Filter for Turns
        if "right" not in gt_text and "left" not in gt_text: continue
            
        city = batch.city[0]
        origin = batch.origin[0].numpy()
        theta = batch.theta[0].item()
        
        map_feats = get_map_features(city, origin, theta)
        
        data = batch.to(model.device)
        with torch.no_grad():
            global_embed, _ = model._get_ego_features(data)
            traj_input = data.y[0].unsqueeze(0)
            
            # --- FIX: Only expecting 1 return value (logits) ---
            logits = model.captioner(global_embed, traj_input, captions=None, return_attn=False) 
            
            pred_ids = logits.argmax(dim=-1)[0]
            pred_text = model.tokenizer.decode(pred_ids)
            
            print(f"Plotting: {gt_text}")
            plot_final_thesis_viz(traj_input[0], f"thesis_final_{count}.png", gt_text, pred_text, map_feats)
            count += 1

if __name__ == "__main__":
    visualize()