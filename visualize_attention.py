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
    if city not in MAP_CACHE:
        try:
            MAP_CACHE[city] = NuScenesMap(dataroot=NUSCENES_MAP_ROOT, map_name=city)
        except Exception as e:
            print(f"Map Error: {e}")
            return {}

    nusc_map = MAP_CACHE[city]
    x, y = origin[0], origin[1]
    patch_box = (x - radius, y - radius, x + radius, y + radius)
    
    features = {'drivable_area': [], 'dividers': [], 'centerlines': []}
    
    # 1. DRIVABLE AREA
    try:
        records = nusc_map.get_records_in_patch(patch_box, ['drivable_area'], mode='intersect')
        for token in records.get('drivable_area', []):
            poly = nusc_map.get('drivable_area', token)
            nodes = [nusc_map.get('node', t) for t in poly['exterior_node_tokens']]
            points = np.array([[n['x'], n['y']] for n in nodes])
            features['drivable_area'].append(transform_to_local(points, x, y, theta))
    except: pass

    # 2. DIVIDERS (Paint)
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
    
    # 3. CENTERLINES (Lanes)
    try:
        records = nusc_map.get_records_in_patch(patch_box, ['lane'], mode='intersect')
        for token in records.get('lane', []):
            pose_record = nusc_map.get_arcline_path(token)
            points = np.array(pose_record)
            features['centerlines'].append(transform_to_local(points, x, y, theta))
    except: pass

    return features

def transform_to_local(global_points, origin_x, origin_y, theta):
    centered = global_points - np.array([origin_x, origin_y])
    c, s = np.cos(-theta), np.sin(-theta)
    R = np.array([[c, -s], [s, c]])
    return centered @ R.T

def plot_map_only(trajectory, save_name, gt_text, pred_text, map_feats):
    # Single Plot, High DPI
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # 1. Plot Drivable Area (Very Light Grey)
    for poly in map_feats['drivable_area']:
        p = Polygon(poly, facecolor='#F0F0F0', edgecolor='none', alpha=0.5, zorder=0)
        ax.add_patch(p)

    # 2. Plot Dividers (Black Solid)
    for line in map_feats['dividers']:
        ax.plot(line[:, 0], line[:, 1], color='black', linewidth=1.5, alpha=0.7, zorder=1)

    # 3. Plot Centerlines (Purple Dashed)
    for line in map_feats['centerlines']:
        ax.plot(line[:, 0], line[:, 1], color='purple', linewidth=2.0, linestyle='--', alpha=0.6, zorder=2)

    # 4. Plot Trajectory (Blue Solid)
    traj = trajectory.cpu().numpy()
    ax.plot(traj[:, 0], traj[:, 1], color='#1f77b4', linewidth=6, zorder=10)
    ax.scatter(traj[0, 0], traj[0, 1], color='#2ca02c', s=250, edgecolors='black', zorder=11)
    ax.scatter(traj[-1, 0], traj[-1, 1], color='#d62728', s=250, edgecolors='black', zorder=11)

    # --- MANUAL LEGEND ---
    # This guarantees the legend appears even if the loop data is weird
    legend_elements = [
        Line2D([0], [0], color='#1f77b4', lw=4, label='Predicted Path'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c', markersize=15, markeredgecolor='k', label='Start'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', markersize=15, markeredgecolor='k', label='End'),
        Line2D([0], [0], color='purple', lw=2, linestyle='--', label='Lane Centerline'),
        Line2D([0], [0], color='black', lw=1.5, label='Lane Divider'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12, framealpha=0.95)

    # Zoom
    x_min, x_max = traj[:, 0].min(), traj[:, 0].max()
    y_min, y_max = traj[:, 1].min(), traj[:, 1].max()
    margin = 15 
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_aspect('equal')
    
    # Title
    ax.set_title(f"GT: {gt_text}\nPred: {pred_text}", fontsize=14, pad=15)
    ax.grid(True, linestyle=':', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    print(f"Saved: {save_name}")
    plt.close()

def visualize():
    with open("vocab.json") as f: vocab = json.load(f)
    print(f"Loading Model...")
    model = HiVTX.load_from_checkpoint(CKPT_PATH, cvae_gan_ckpt=BACKBONE_PATH, vocab_size=len(vocab), strict=False)
    model.eval().cuda()

    datamodule = NuScenesHiVTDataModule(root=DATA_ROOT, split_file="balanced_splits.json", val_batch_size=1, shuffle=True)
    datamodule.setup()
    loader = datamodule.val_dataloader()
    
    print("Searching for Turn samples...")
    count = 0
    for batch in loader:
        if count >= 3: break
        
        gt_ids = batch.caption_ids[0]
        gt_text = model.tokenizer.decode(gt_ids)
        if "right" not in gt_text and "left" not in gt_text: continue
            
        city = batch.city[0]
        origin = batch.origin[0].numpy()
        theta = batch.theta[0].item()
        
        map_feats = get_map_features(city, origin, theta)
        
        data = batch.to(model.device)
        with torch.no_grad():
            global_embed, _ = model._get_ego_features(data)
            traj_input = data.y[0].unsqueeze(0)
            
            # --- FIX IS HERE: No unpacking ---
            logits = model.captioner(global_embed, traj_input, captions=None, return_attn=False) 
            
            pred_ids = logits.argmax(dim=-1)[0]
            pred_text = model.tokenizer.decode(pred_ids)
            
            print(f"Plotting: {gt_text}")
            plot_map_only(traj_input[0], f"map_focus_{count}.png", gt_text, pred_text, map_feats)
            count += 1

if __name__ == "__main__":
    visualize()