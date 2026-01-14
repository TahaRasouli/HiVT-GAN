import torch
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import sys
from matplotlib.patches import Polygon
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
            return {}

    nusc_map = MAP_CACHE[city]
    x, y = origin[0], origin[1]
    patch_box = (x - radius, y - radius, x + radius, y + radius)
    
    features = {
        'drivable_area': [],
        'dividers': [],
        'centerlines': []
    }
    
    # 1. DRIVABLE AREA (Asphalt)
    try:
        records = nusc_map.get_records_in_patch(patch_box, ['drivable_area'], mode='intersect')
        for token in records.get('drivable_area', []):
            poly = nusc_map.get('drivable_area', token)
            nodes = [nusc_map.get('node', t) for t in poly['exterior_node_tokens']]
            points = np.array([[n['x'], n['y']] for n in nodes])
            features['drivable_area'].append(transform_to_local(points, x, y, theta))
    except: pass

    # 2. DIVIDERS (Paint Lines)
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
    
    # 3. CENTERLINES (Graph Connections)
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

def plot_static_figure(trajectory, attn_weights, save_name, gt_text, pred_text, map_feats, words):
    fig, ax = plt.subplots(1, 2, figsize=(24, 10), gridspec_kw={'width_ratios': [1.3, 1]})
    
    # --- 1. MAP PLOT ---
    
    # A. Drivable Area (Light Grey Background)
    for poly in map_feats['drivable_area']:
        # Matplotlib Polygon
        p = Polygon(poly, facecolor='#E8E8E8', edgecolor='none', alpha=0.5, zorder=0)
        ax[0].add_patch(p)

    # B. Dividers (Solid Black Lines)
    for line in map_feats['dividers']:
        ax[0].plot(line[:, 0], line[:, 1], color='black', linewidth=1.5, alpha=0.8, zorder=1)

    # C. Centerlines (Dashed Purple - The "Lanes")
    for i, line in enumerate(map_feats['centerlines']):
        label = "Lane Centerline" if i == 0 else None
        ax[0].plot(line[:, 0], line[:, 1], color='purple', linewidth=2.0, linestyle='--', alpha=0.6, label=label, zorder=2)

    # D. Trajectory
    traj = trajectory.cpu().numpy()
    ax[0].plot(traj[:, 0], traj[:, 1], color='#1f77b4', linewidth=5, label="Predicted Path", zorder=10)
    ax[0].scatter(traj[0, 0], traj[0, 1], color='green', s=150, edgecolors='black', label="Start", zorder=11)
    ax[0].scatter(traj[-1, 0], traj[-1, 1], color='red', s=150, edgecolors='black', label="End", zorder=11)

    # E. Smart Zoom (Tight Bounding Box)
    x_min, x_max = traj[:, 0].min(), traj[:, 0].max()
    y_min, y_max = traj[:, 1].min(), traj[:, 1].max()
    margin = 15 # Meters of context around the car
    ax[0].set_xlim(x_min - margin, x_max + margin)
    ax[0].set_ylim(y_min - margin, y_max + margin)
    
    ax[0].set_aspect('equal')
    ax[0].legend(loc='upper right')
    ax[0].set_title(f"Agent-Centric Map\nGT: {gt_text}\nPred: {pred_text}", fontsize=14)

    # --- 2. HEATMAP PLOT ---
    if len(words) > 0:
        cax = ax[1].imshow(attn_weights.cpu().numpy(), aspect='auto', cmap='plasma', interpolation='nearest')
        ax[1].set_xticks(np.arange(0, 30, 5))
        ax[1].set_xlabel("Time Step ($t_0$ to $t_{30}$)", fontsize=12)
        ax[1].set_yticks(np.arange(len(words)))
        ax[1].set_yticklabels(words, fontsize=12, weight='bold')
        ax[1].set_title("Attention Weights", fontsize=14)
        fig.colorbar(cax, ax=ax[1])

    plt.tight_layout()
    plt.savefig(save_name, dpi=300) # High Res
    print(f"Saved High-Res Image: {save_name}")
    plt.close()

def visualize():
    with open("vocab.json") as f: vocab = json.load(f)
    print(f"Loading Model...")
    model = HiVTX.load_from_checkpoint(CKPT_PATH, cvae_gan_ckpt=BACKBONE_PATH, vocab_size=len(vocab), strict=False)
    model.eval().cuda()

    datamodule = NuScenesHiVTDataModule(root=DATA_ROOT, split_file="balanced_splits.json", val_batch_size=1, shuffle=True)
    datamodule.setup()
    loader = datamodule.val_dataloader()
    
    print("Searching for samples...")
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
            logits, attn_weights = model.captioner(global_embed, traj_input, captions=None, return_attn=True)
            
            pred_ids = logits.argmax(dim=-1)[0]
            pred_text = model.tokenizer.decode(pred_ids)
            words = [model.tokenizer.idx2word[i.item()] for i in pred_ids if i.item() > 1]
            
            if len(words) > 0:
                print(f"Plotting: {gt_text}")
                plot_static_figure(traj_input[0], attn_weights[0, :len(words)], f"thesis_static_{count}.png", gt_text, pred_text, map_feats, words)
                count += 1

if __name__ == "__main__":
    visualize()