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

def plot_attention(trajectory, caption_words, attn_weights, save_name):
    """
    Plots the trajectory and the attention heatmap side-by-side.
    attn_weights: [Seq_Len, 30]
    """
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    
    # --- PLOT 1: Trajectory ---
    traj = trajectory.cpu().numpy()
    ax[0].plot(traj[:, 0], traj[:, 1], 'b-', linewidth=3, label="Predicted Path")
    ax[0].plot(traj[0, 0], traj[0, 1], 'go', markersize=10, label="Start")
    ax[0].plot(traj[-1, 0], traj[-1, 1], 'ro', markersize=10, label="End")
    ax[0].set_title("Trajectory (30 Steps)", fontsize=14)
    ax[0].grid(True)
    ax[0].legend()
    ax[0].axis('equal')

    # --- PLOT 2: Attention Heatmap ---
    # X-axis: Time Steps (0-30)
    # Y-axis: Words generated
    cax = ax[1].imshow(attn_weights.cpu().numpy(), aspect='auto', cmap='viridis')
    
    # Set Labels
    ax[1].set_xticks(np.arange(0, 30, 5))
    ax[1].set_xticklabels(np.arange(0, 30, 5))
    ax[1].set_xlabel("Trajectory Time Step (0=Now, 30=3s Future)", fontsize=12)
    
    ax[1].set_yticks(np.arange(len(caption_words)))
    ax[1].set_yticklabels(caption_words, fontsize=12)
    ax[1].set_title("Visual Attention: What the Model Looked At", fontsize=14)
    
    fig.colorbar(cax, ax=ax[1], label="Attention Weight")
    plt.tight_layout()
    plt.savefig(save_name)
    print(f"Saved {save_name}")
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
    
    # 3. Find interesting samples
    print("Searching for HIGH VELOCITY turns/lane changes...")
    count = 0
    
    for batch in loader:
        if count >= 3: break 
        
        # Move to device first to check values
        data = batch.to(model.device)
        
        # --- NEW FILTERING LOGIC ---
        # 1. Get Ground Truth Trajectory of Ego (Node 0)
        gt_traj = data.y[0] # [30, 2]
        
        # 2. Calculate Displacement (Distance between start and end)
        displacement = torch.norm(gt_traj[-1] - gt_traj[0]).item()
        
        # 3. Filter: Must move at least 10 meters (to avoid stationary cars)
        if displacement < 10.0:
            continue
            
        # 4. Check Caption for "turn" or "change"
        gt_ids = batch.caption_ids[0]
        gt_text = model.tokenizer.decode(gt_ids)
        
        if "turn" not in gt_text and "change" not in gt_text:
            continue

        # If we reach here, we have a Fast Moving Car doing a Turn!
        print(f"Found Sample: {gt_text} | Displacement: {displacement:.2f}m")
            
        with torch.no_grad():
            # A. Get Context
            global_embed, _ = model._get_ego_features(data) # [1, 128]
            
            # B. Generate Trajectory (Single Mode for clarity in attention map)
            # We force the CVAE to give us the BEST reconstruction (z=0 or Mean)
            # For visualization, let's just use the Ground Truth trajectory to see 
            # if the captioner understands the PERFECT path.
            traj_input = data.y[0].unsqueeze(0) # [1, 30, 2]
            
            # C. Generate Caption + Attention
            # Note: We added return_attn=True to the captioner forward method!
            logits, attn_weights = model.captioner(
                global_embed, traj_input, captions=None, return_attn=True
            )
            
            # Decode Words
            pred_ids = logits.argmax(dim=-1)[0]
            pred_text = model.tokenizer.decode(pred_ids)
            
            # Clean up words list for plotting
            words = []
            for idx in pred_ids:
                word = model.tokenizer.idx2word[idx.item()]
                if word == "<EOS>": break
                if word not in ["<PAD>", "<SOS>"]:
                    words.append(word)
                    
            # Extract relevant attention weights [Len_Words, 30]
            # attn_weights is [1, Seq, 30] -> [Seq, 30]
            relevant_attn = attn_weights[0, :len(words), :]

            print(f"Sample {count}: {pred_text}")
            plot_attention(
                traj_input[0], 
                words, 
                relevant_attn, 
                f"attention_viz_{count}.png"
            )
            count += 1

if __name__ == "__main__":
    visualize()