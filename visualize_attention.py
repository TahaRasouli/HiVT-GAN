import torch
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from models.hivt_x import HiVTX
from datamodules.nuscenes_datamodule import NuScenesHiVTDataModule

# --- NUSCENES IMPORTS ---
try:
    from nuscenes.map_expansion.map_api import NuScenesMap
except ImportError:
    print("Error: nuscenes-devkit not installed. Run 'pip install nuscenes-devkit'")
    sys.exit(1)

# --- CONFIGURATION ---
CKPT_PATH = " /mount/arbeitsdaten/studenten4/rasoulta/HiVT-GAN/lightning_logs/checkpoints/version_54/checkpoints/epoch=29-step=8040.ckpt" # UPDATE THIS
BACKBONE_PATH = "/mount/studenten/projects/rasoulta/checkpoints/vae-gan-baseline/checkpoints/epoch=45-step=60812.ckpt"
DATA_ROOT = "/mount/studenten/projects/rasoulta/dataset"
NUSCENES_MAP_ROOT = "/mount/arbeitsdaten/analysis/rasoulta/nuscenes/nuscenes_meta/maps" # Path to folder containing 'boston-seaport.json', etc.

MAP_CACHE = {}

def get_local_map_features(city, origin, theta, radius=75):
    """
    Fetches global lanes AND drivable area, transforms to Agent-Centric frame.
    """
    # 1. Load Map
    if city not in MAP_CACHE:
        # Verify file exists first
        json_path = os.path.join(NUSCENES_MAP_ROOT, f"{city}.json")
        if not os.path.exists(json_path):
            print(f"ERROR: Map file not found at {json_path}")
            print(f"Check NUSCENES_MAP_ROOT config!")
            return [], []
            
        print(f"Loading map for {city}...")
        MAP_CACHE[city] = NuScenesMap(dataroot=NUSCENES_MAP_ROOT, map_name=city)
    
    nusc_map = MAP_CACHE[city]
    
    # 2. Define Box
    x, y = origin[0], origin[1]
    patch_box = (x - radius, y - radius, x + radius, y + radius)
    
    # DEBUG: Print coordinates to verify they are not (0,0)
    print(f"   > Querying {city} at Origin: ({x:.1f}, {y:.1f})")
    print(f"   > Patch Box: {patch_box}")

    # 3. Get Records
    # We query 'lane' (centerlines) and 'drivable_area' (road polygons)
    try:
        records = nusc_map.get_records_in_patch(patch_box, ['lane', 'road_segment', 'drivable_area'], mode='intersect')
    except Exception as e:
        print(f"   > API Error: {e}")
        return [], []
    
    local_lanes = []
    local_polygons = []
    
    # 4. Process Lanes (Centerlines)
    for lane_token in records['lane']:
        try:
            pose_record = nusc_map.get_arcline_path(lane_token)
            points = np.array(pose_record)
            local_lanes.append(transform_to_local(points, x, y, theta))
        except: continue
            
    # 5. Process Drivable Area (Road Edges/Polygons) - Fallback if lanes are missing
    # This helps visualizing intersections even if centerlines are missing
    for token in records['drivable_area']:
        try:
            # Get polygon exterior
            poly_record = nusc_map.get('drivable_area', token)
            if 'exterior_node_tokens' in poly_record:
                nodes = [nusc_map.get('node', t) for t in poly_record['exterior_node_tokens']]
                points = np.array([[n['x'], n['y']] for n in nodes])
                local_polygons.append(transform_to_local(points, x, y, theta))
        except: continue

    print(f"   > Found {len(local_lanes)} lanes and {len(local_polygons)} road polygons.")
    return local_lanes, local_polygons

def transform_to_local(global_points, origin_x, origin_y, theta):
    # Translate
    centered = global_points - np.array([origin_x, origin_y])
    # Rotate (Negative theta for Global -> Local)
    c, s = np.cos(-theta), np.sin(-theta)
    R = np.array([[c, -s], [s, c]])
    return centered @ R.T

def plot_thesis_figure(trajectory, caption_words, attn_weights, save_name, gt_text, pred_text, lanes, polygons):
    fig, ax = plt.subplots(1, 2, figsize=(24, 9), gridspec_kw={'width_ratios': [1, 1.2]})
    
    # --- PLOT 1: BEV SCENE ---
    # Plot Road Polygons (Light Grey Background)
    for poly in polygons:
        ax[0].fill(poly[:, 0], poly[:, 1], color='#E0E0E0', alpha=0.5, zorder=0)

    # Plot Lanes (Darker Grey Lines)
    for lane in lanes:
        ax[0].plot(lane[:, 0], lane[:, 1], color='#808080', linewidth=1.5, alpha=0.7, linestyle='--', zorder=1)

    # Plot Trajectory
    traj = trajectory.cpu().numpy()
    
    # Color logic: Correct Direction = Blue, Wrong = Orange
    is_right_turn_geo = traj[-1, 1] < -1.5
    pred_right_text = "right" in pred_text.lower()
    path_color = '#1f77b4' if (is_right_turn_geo == pred_right_text) or "continue" in pred_text else '#ff7f0e'
    
    ax[0].plot(traj[:, 0], traj[:, 1], color=path_color, linewidth=5, label="Predicted Path", zorder=5)
    ax[0].scatter(traj[0, 0], traj[0, 1], color='#2ca02c', s=150, edgecolors='black', label="Start", zorder=6)
    ax[0].scatter(traj[-1, 0], traj[-1, 1], color='#d62728', s=150, edgecolors='black', label="End", zorder=6)
    
    box_text = f"Ground Truth:\n{gt_text}\n\nPrediction:\n{pred_text}"
    ax[0].text(0.05, 0.95, box_text, transform=ax[0].transAxes, fontsize=12,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax[0].set_title("Agent-Centric View (Map + Trajectory)", fontsize=16)
    ax[0].axis('equal')
    ax[0].grid(False)
    
    # Zoom
    margin = 20
    ax[0].set_xlim(traj[:,0].min()-margin, traj[:,0].max()+margin)
    ax[0].set_ylim(traj[:,1].min()-margin, traj[:,1].max()+margin)

    # --- PLOT 2: HEATMAP ---
    if attn_weights.shape[0] > 0:
        cax = ax[1].imshow(attn_weights.cpu().numpy(), aspect='auto', cmap='plasma', interpolation='nearest')
        ax[1].set_xticks(np.arange(0, 30, 5))
        ax[1].set_xticklabels(np.arange(0, 30, 5))
        ax[1].set_xlabel("Time Step")
        ax[1].set_yticks(np.arange(len(caption_words)))
        ax[1].set_yticklabels(caption_words, fontsize=13, weight='bold')
        ax[1].set_title("Spatiotemporal Attention Alignment", fontsize=16)
        fig.colorbar(cax, ax=ax[1])
    
    plt.tight_layout()
    plt.savefig(save_name, dpi=150)
    print(f"Saved {save_name}")
    plt.close()

def visualize():
    # Load Model
    with open("vocab.json") as f: vocab = json.load(f)
    print(f"Loading Model...")
    model = HiVTX.load_from_checkpoint(CKPT_PATH, cvae_gan_ckpt=BACKBONE_PATH, vocab_size=len(vocab), strict=False)
    model.eval().cuda()

    # Load Data
    datamodule = NuScenesHiVTDataModule(root=DATA_ROOT, split_file="balanced_splits.json", val_batch_size=1, shuffle=True)
    datamodule.setup()
    loader = datamodule.val_dataloader()
    
    print("Searching for Right Turns...")
    count = 0
    
    for batch in loader:
        if count >= 3: break 
        
        gt_ids = batch.caption_ids[0]
        gt_text = model.tokenizer.decode(gt_ids)
        
        if "right" not in gt_text and "left" not in gt_text: continue
            
        # Extract Map Info
        city = batch.city[0]
        origin = batch.origin[0].numpy()
        theta = batch.theta[0].item()
        
        # Get Map Features
        lanes, polys = get_local_map_features(city, origin, theta)
        
        # Inference
        data = batch.to(model.device)
        with torch.no_grad():
            global_embed, _ = model._get_ego_features(data)
            traj_input = data.y[0].unsqueeze(0)
            logits, attn_weights = model.captioner(global_embed, traj_input, captions=None, return_attn=True)
            
            pred_ids = logits.argmax(dim=-1)[0]
            pred_text = model.tokenizer.decode(pred_ids)
            words = [model.tokenizer.idx2word[i.item()] for i in pred_ids if i.item() > 1] # Skip PAD/SOS
            
            if len(words) > 0:
                print(f"Plotting Sample {count}: {gt_text}")
                plot_thesis_figure(traj_input[0], words, attn_weights[0, :len(words)], 
                                  f"thesis_final_{count}.jpg", gt_text, pred_text, lanes, polys)
                count += 1

if __name__ == "__main__":
    visualize()