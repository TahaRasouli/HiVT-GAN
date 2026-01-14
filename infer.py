import torch
import numpy as np
import json
import os
import random
from models.hivt_x import HiVTX
from datamodules.nuscenes_datamodule import NuScenesHiVTDataModule

# --- CONFIGURATION ---
CKPT_PATH = "/mount/arbeitsdaten/studenten4/rasoulta/HiVT-GAN/lightning_logs/version_54/checkpoints/epoch=29-step=8040.ckpt"
BACKBONE_PATH = "/mount/studenten/projects/rasoulta/checkpoints/vae-gan-baseline/checkpoints/epoch=45-step=60812.ckpt"
DATA_ROOT = "/mount/studenten/projects/rasoulta/dataset"
OUTPUT_FILE = "inference_results.json"

def run_inference():
    # 1. Load Model
    print("Loading Model...")
    with open("vocab.json") as f: vocab = json.load(f)
    model = HiVTX.load_from_checkpoint(CKPT_PATH, cvae_gan_ckpt=BACKBONE_PATH, vocab_size=len(vocab), strict=False)
    model.eval().cuda()

    # 2. Load Data
    print("Loading Data...")
    datamodule = NuScenesHiVTDataModule(root=DATA_ROOT, split_file="balanced_splits.json", val_batch_size=1, shuffle=True)
    datamodule.setup()
    loader = datamodule.val_dataloader()

    results = []
    target_count = 5
    print(f"Searching for {target_count} non-straight, non-stationary samples...")

    for batch_idx, batch in enumerate(loader):
        if len(results) >= target_count:
            break

        # Move to device
        data = batch.to(model.device)
        
        # 3. Filter Logic
        # A. Check Text (Must not contain "straight")
        gt_ids = batch.caption_ids[0]
        gt_text = model.tokenizer.decode(gt_ids)
        
        if "straight" in gt_text.lower():
            continue

        # B. Check Motion (Must move at least 2 meters)
        traj_input = data.y[0] # [30, 2]
        displacement = torch.norm(traj_input[-1] - traj_input[0]).item()
        
        if displacement < 2.0: # Stationary check
            continue

        # 4. Run Inference
        with torch.no_grad():
            global_embed, _ = model._get_ego_features(data)
            traj_input_batch = traj_input.unsqueeze(0)
            
            # Generate Caption
            logits = model.captioner(global_embed, traj_input_batch, captions=None, return_attn=False)
            pred_ids = logits.argmax(dim=-1)[0]
            pred_text = model.tokenizer.decode(pred_ids)

        # 5. Format Data for JSON
        sample_data = {
            "sample_idx": batch_idx,
            "city": batch.city[0],
            "ground_truth_text": gt_text,
            "predicted_text": pred_text,
            # Convert tensors to simple lists for JSON serialization
            "origin": batch.origin[0].cpu().numpy().tolist(),
            "theta": batch.theta[0].item(),
            "trajectory": traj_input.cpu().numpy().tolist() # The actual path taken
        }
        
        results.append(sample_data)
        print(f"Collected Sample {len(results)}: {gt_text}")

    # 6. Save to File
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nSuccess! Saved {len(results)} samples to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_inference()