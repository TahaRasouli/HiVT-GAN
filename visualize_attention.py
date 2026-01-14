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
    exit()

# --- CONFIGURATION ---
CKPT_PATH = "/mount/studenten/projects/rasoulta/checkpoints/x_baseline/checkpoints/epoch=29-step=8040.ckpt" # UPDATE THIS
BACKBONE_PATH = "/mount/studenten/projects/rasoulta/checkpoints/vae-gan-baseline/checkpoints/epoch=45-step=60812.ckpt"
DATA_ROOT = "/mount/studenten/projects/rasoulta/dataset"
NUSCENES_MAP_ROOT = "/mount/arbeitsdaten/analysis/rasoulta/nuscenes/nuscenes_meta" # Path to folder containing 'boston-seaport.json', etc.

# Cache maps to avoid reloading
MAP_CACHE = {}

def get_local_lanes(city, origin, theta, radius=50):
    """
    Fetches global lanes and transforms them to the Agent-Centric frame.
    """
    if city not in MAP_CACHE:
        print(f"Loading map for {city}...")
        MAP_CACHE[city] = NuScenesMap(dataroot=NUSCENES_MAP_ROOT, map_name=city)
    
    nusc_map = MAP_CACHE[city]
    
    # 1. Define query box in Global Coordinates
    x, y = origin[0], origin[1]
    patch_box = (x, y, radius, radius)
    patch_angle = 0  # We query axis-aligned first
    
    # 2. Get Lane Records
    layer_names = ['lane', 'road_segment', 'drivable_area']
    records = nusc_map.get_records_in_patch(patch_box, layer_names, mode='intersect')
    
    local_lanes = []
    
    # 3. Process Lane Centerlines
    for lane_token in records['lane']:
        # Get Global Line
        try:
            pose_record = nusc_map.get_arcline_path(lane_token)
            global_points = np.array(pose_record) # [N, 2]
            
            # 4. Transform Global -> Local
            # A. Translate (Global - Origin)
            centered = global_points - np.array([x, y])
            
            # B. Rotate (align with agent heading)
            # Rotation matrix for NEGATIVE theta (Global to Local)
            c, s = np.cos(-theta), np.sin(-theta)
            R = np.array([[c, -s], [s, c]])
            
            local_points = centered @ R.T
            local_lanes.append(local_points)
        except Exception as e:
            continue
            
    return local_lanes

def plot_thesis_figure(trajectory, caption_words, attn_weights, save_name, gt_text, pred_text, local_lanes):
    """
    Generates the Thesis-Quality Figure: BEV Map + Attention Heatmap
    """
    fig, ax = plt.subplots(1, 2, figsize=(24, 9), gridspec_kw={'width_ratios': [1, 1.2]})
    
    # --- PLOT 1: BEV SCENE ---
    # Plot Lanes
    for lane in local_lanes:
        ax[0].plot(lane[:, 0], lane[:, 1], color='#B0B0B0', linewidth=1.5, alpha=0.6, zorder=1)

    # Plot Trajectory
    traj = trajectory.cpu().numpy()
    
    # Color Coding based on Accuracy
    is_right_turn_geo = traj[-1, 1] < -1.5
    pred_right_text = "right" in pred_text.lower()
    
    # If text matches geometry OR if it's just a general straight path
    match = (is_right_turn_geo == pred_right_text)
    path_color = '#1f77b4' # Standard Blue
    
    ax[0].plot(traj[:, 0], traj[:, 1], color=path_color, linewidth=5, label="Predicted Path", zorder=5)
    ax[0].scatter(traj[0, 0], traj[0, 1], color='#2ca02c', s=150, edgecolors='black', label="Start", zorder=6)
    ax[0].scatter(traj[-1, 0], traj[-1, 1], color='#d62728', s=150, edgecolors='black', label="End", zorder=6)
    
    # Text Annotation
    box_text = f"Ground Truth:\n{gt_text}\n\nPrediction:\n{pred_text}"
    ax[0].text(0.05, 0.95, box_text, transform=ax[0].transAxes, fontsize=12,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax[0].set_title("Agent-Centric View (Map + Trajectory)", fontsize=16, pad=15)
    ax[0].set_xlabel("Lateral Distance (m)", fontsize=12)
    ax[0].set_ylabel("Longitudinal Distance (m)", fontsize=12)
    ax[0].axis('equal')
    ax[0].grid(True, linestyle=':', alpha=0.5)
    
    # Zoom out slightly to show context
    ax[0].set_xlim(-20, 20)
    ax[0].set_ylim(-10, 40)

    # --- PLOT 2: ATTENTION HEATMAP ---
    if attn_weights.shape[0] > 0:
        cax = ax[1].imshow(attn_weights.cpu().numpy(), aspect='auto', cmap='plasma', interpolation='nearest')
        
        # Axis settings
        ax[1].set_xticks(np.arange(0, 30, 5))
        ax[1].set_xticklabels(np.arange(0, 30, 5), fontsize=10)
        ax[1].set_xlabel("Trajectory Time Step ($t_0$ to $t_{30}$)", fontsize=14, labelpad=10)
        
        ax[1].set_yticks(np.arange(len(caption_words)))
        ax[1].set_yticklabels(caption_words, fontsize=13, weight='bold')
        ax[1].set_title("Spatiotemporal Attention Alignment", fontsize=16, pad=15)
        
        # Colorbar
        cbar = fig.colorbar(cax, ax=ax[1], pad=0.02)
        cbar.set_label("Attention Weight", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"Saved Thesis Figure: {save_name}")
    plt.close()

def visualize():
    # 1. Load Model
    with open("vocab.json") as f: vocab = json.load(f)
    print(f"Loading Model...")
    model = HiVTX.load_from_checkpoint(
        CKPT_PATH, 
        cvae_gan_ckpt=BACKBONE_PATH, 
        vocab_size=len(vocab),
        strict=False
    )
    model.eval().cuda()

    # 2. Load Data
    datamodule = NuScenesHiVTDataModule(
        root=DATA_ROOT, 
        split_file="balanced_splits.json",
        val_batch_size=1, 
        shuffle=True
    )
    datamodule.setup()
    loader = datamodule.val_dataloader()
    
    print("Searching for Lane Change / Turn samples...")
    count = 0
    
    for batch in loader:
        if count >= 5: break 
        
        # 3. Filter for interesting samples
        gt_ids = batch.caption_ids[0]
        gt_text = model.tokenizer.decode(gt_ids)
        
        if "right" not in gt_text and "left" not in gt_text:
            continue
            
        # 4. Extract Map Metadata BEFORE moving to GPU (if simpler) or move everything
        # 'city' is usually list of strings in the batch
        # 'origin' is [B, 2]
        # 'theta' is [B]
        
        # Access the first element of the batch
        city_name = batch.city[0] # assuming list
        origin = batch.origin[0].numpy()
        theta = batch.theta[0].item()
        
        # 5. Fetch Map Lanes
        local_lanes = get_local_lanes(city_name, origin, theta)
        
        # 6. Run Inference
        data = batch.to(model.device)
        with torch.no_grad():
            global_embed, _ = model._get_ego_features(data)
            traj_input = data.y[0].unsqueeze(0)
            
            logits, attn_weights = model.captioner(
                global_embed, traj_input, captions=None, return_attn=True
            )
            
            pred_ids = logits.argmax(dim=-1)[0]
            pred_text = model.tokenizer.decode(pred_ids)
            
            words = []
            for idx in pred_ids:
                word = model.tokenizer.idx2word[idx.item()]
                if word == "<EOS>": break
                if word not in ["<PAD>", "<SOS>"]:
                    words.append(word)
            
            if len(words) > 0:
                relevant_attn = attn_weights[0, :len(words), :]
                
                print(f"Sample {count} | {gt_text}")
                plot_thesis_figure(
                    traj_input[0], 
                    words, 
                    relevant_attn, 
                    f"thesis_viz_{count}.png",
                    gt_text,
                    pred_text,
                    local_lanes
                )
                count += 1

if __name__ == "__main__":
    visualize()