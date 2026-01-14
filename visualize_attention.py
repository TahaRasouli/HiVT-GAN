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


def plot_attention(trajectory, caption_words, attn_weights, save_name, gt_text, pred_text, lane_vectors=None):
    """
    Plots Trajectory, LANES, and Attention Heatmap
    """
    fig, ax = plt.subplots(1, 2, figsize=(22, 8))
    
    # --- PLOT 1: BEV SCENE (Trajectory + Lanes) ---
    
    # 1. Plot Lanes (Background) - The most important fix
    if lane_vectors is not None and len(lane_vectors) > 0:
        lx = lane_vectors[:, 0]
        ly = lane_vectors[:, 1]
        # Plot as small grey dots to show road geometry
        ax[0].scatter(lx, ly, c='#888888', s=1.5, alpha=0.5, label='Map Lanes')
    else:
        print("Warning: No lane vectors found for plotting!")

    # 2. Plot Predicted Trajectory
    traj = trajectory.cpu().numpy()
    
    # Heuristic: Check if direction matches text (Right vs Left)
    is_right_turn = traj[-1, 1] < -1.0
    pred_right = "right" in pred_text.lower()
    
    # Orange line if mismatch, Blue if match
    line_color = 'b-'
    
    ax[0].plot(traj[:, 0], traj[:, 1], line_color, linewidth=4, label="Predicted Path")
    ax[0].plot(traj[0, 0], traj[0, 1], 'go', markersize=10, label="Start")
    ax[0].plot(traj[-1, 0], traj[-1, 1], 'ro', markersize=10, label="End")
    
    # Text Box
    ax[0].text(0, 0, f"GT: {gt_text}\nPred: {pred_text}", fontsize=12, 
               bbox=dict(facecolor='white', alpha=0.9, edgecolor='black'))
    
    ax[0].set_title(f"Scene BEV (End Y={traj[-1, 1]:.2f}m)", fontsize=14)
    ax[0].legend(loc='upper right')
    ax[0].axis('equal') # CRITICAL: Keeps geometry proportional
    ax[0].grid(True, alpha=0.4)

    # --- PLOT 2: Attention Heatmap ---
    if attn_weights.shape[0] > 0:
        cax = ax[1].imshow(attn_weights.cpu().numpy(), aspect='auto', cmap='viridis')
        
        ax[1].set_xticks(np.arange(0, 30, 5))
        ax[1].set_xticklabels(np.arange(0, 30, 5))
        ax[1].set_xlabel("Trajectory Time Step (0=Now, 30=3s Future)", fontsize=12)
        
        ax[1].set_yticks(np.arange(len(caption_words)))
        ax[1].set_yticklabels(caption_words, fontsize=12)
        ax[1].set_title("Visual Attention Heatmap", fontsize=14)
        
        fig.colorbar(cax, ax=ax[1], label="Attention Weight")
    
    plt.tight_layout()
    plt.savefig(save_name, dpi=150)
    print(f"Saved {save_name}")
    plt.close()

def visualize():
    # 1. Load Model
    with open("vocab.json") as f: vocab = json.load(f)
    print(f"Loading Lane-Aware Model...")
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
    
    print("Searching for samples...")
    count = 0
    
    for batch in loader:
        if count >= 5: break 
        
        data = batch.to(model.device)
        gt_ids = batch.caption_ids[0]
        gt_text = model.tokenizer.decode(gt_ids)
        
        # Filter for "right" turns or lane changes to test your theory
        if "right" not in gt_text:
            continue

        # Extract Lane Vectors (CPU for plotting)
        lane_vectors = None
        if hasattr(data, 'lane_vectors'):
            # Check if it's not empty
            if data.lane_vectors.numel() > 0:
                lane_vectors = data.lane_vectors.detach().cpu().numpy()
            
        with torch.no_grad():
            global_embed, _ = model._get_ego_features(data)
            traj_input = data.y[0].unsqueeze(0)
            
            # Generate
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
                
                print(f"Sample {count} | Lanes: {lane_vectors.shape if lane_vectors is not None else 'None'}")
                plot_attention(
                    traj_input[0], 
                    words, 
                    relevant_attn, 
                    f"lane_viz_{count}.png",
                    gt_text,
                    pred_text,
                    lane_vectors=lane_vectors # Pass the lanes!
                )
                count += 1

if __name__ == "__main__":
    visualize()