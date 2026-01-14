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
    fig, ax = plt.subplots(1, 2, figsize=(22, 8))
    
    # --- PLOT 1: Trajectory + MAP ---
    # 1. Plot Map/Lanes (Background)
    if lane_vectors is not None:
        # lane_vectors is typically [N, 2]. Plot as small grey dots.
        lx, ly = lane_vectors[:, 0].cpu().numpy(), lane_vectors[:, 1].cpu().numpy()
        ax[0].scatter(lx, ly, c='gray', s=1, alpha=0.3, label='Map Lanes')

    # 2. Plot Trajectory
    traj = trajectory.cpu().numpy()
    is_right_turn = traj[-1, 1] < -1.0 # Simple heuristic check
    pred_right = "right" in pred_text.lower()
    
    # Logic: If text says right but traj goes left, line is ORANGE (Check Map!)
    line_color = 'b-'
    
    ax[0].plot(traj[:, 0], traj[:, 1], line_color, linewidth=4, label="Predicted Path")
    ax[0].plot(traj[0, 0], traj[0, 1], 'go', markersize=10, label="Start")
    ax[0].plot(traj[-1, 0], traj[-1, 1], 'ro', markersize=10, label="End")
    
    ax[0].set_title(f"Scene BEV\nGT: {gt_text}", fontsize=12)
    ax[0].legend()
    ax[0].axis('equal')
    ax[0].grid(True, alpha=0.4)

    # --- PLOT 2: Attention ---
    # (Same as before)
    if attn_weights.shape[0] > 0:
        cax = ax[1].imshow(attn_weights.cpu().numpy(), aspect='auto', cmap='viridis')
        ax[1].set_xticks(np.arange(0, 30, 5))
        ax[1].set_xticklabels(np.arange(0, 30, 5))
        ax[1].set_xlabel("Time Step", fontsize=12)
        ax[1].set_yticks(np.arange(len(caption_words)))
        ax[1].set_yticklabels(caption_words, fontsize=12)
        ax[1].set_title("Attention Heatmap", fontsize=14)
        
    plt.tight_layout()
    plt.savefig(save_name)
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
    
    # 3. Search for RIGHT TURNS (The hardest case)
    print("Searching for Right Turns to verify the fix...")
    count = 0
    
    for batch in loader:
        if count >= 5: break 
        
        data = batch.to(model.device)
        gt_ids = batch.caption_ids[0]
        gt_text = model.tokenizer.decode(gt_ids)
        
        # Filter: We want samples where the ground truth says "Right"
        if "right" not in gt_text:
            continue
            
        with torch.no_grad():
            global_embed, _ = model._get_ego_features(data) # [1, 128]
            traj_input = data.y[0].unsqueeze(0) # [1, 30, 2]
            
            # Generate
            logits, attn_weights = model.captioner(
                global_embed, traj_input, captions=None, return_attn=True
            )
            
            pred_ids = logits.argmax(dim=-1)[0]
            pred_text = model.tokenizer.decode(pred_ids)
            
            # Words for plotting
            words = []
            for idx in pred_ids:
                word = model.tokenizer.idx2word[idx.item()]
                if word == "<EOS>": break
                if word not in ["<PAD>", "<SOS>"]:
                    words.append(word)
            
            # Check length match
            if len(words) > 0:
                relevant_attn = attn_weights[0, :len(words), :]
                
                print(f"Sample {count} | GT: {gt_text} | Pred: {pred_text}")
                plot_attention(
                    traj_input[0], 
                    words, 
                    relevant_attn, 
                    f"lane_aware_viz_{count}.png",
                    gt_text,
                    pred_text
                )
                count += 1

if __name__ == "__main__":
    visualize()