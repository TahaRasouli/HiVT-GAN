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

# Targets: How many samples we want for each category
TARGET_COUNTS = {
    "lane_change": 10,
    "turn": 10,
    "u_turn": 10,
    "intersection": 10
}

def get_category(text):
    """
    Classifies the text into one of the target categories.
    Returns None if it doesn't fit or is 'straight'.
    """
    text = text.lower()
    
    # 1. Exclude Straight / Stationary logic handled in main loop
    if "straight" in text:
        return None
        
    # 2. Check for U-Turn (Specific type of turn)
    if "u-turn" in text:
        return "u_turn"
        
    # 3. Check for Lane Change
    if "change lane" in text or "lane change" in text:
        return "lane_change"
        
    # 4. Check for Turns (Left/Right) - Excludes U-turn because we checked it above
    if "turn" in text or "left" in text or "right" in text:
        return "turn"
        
    # 5. Check for Intersection / Juncture
    if "intersection" in text or "junction" in text or "juncture" in text:
        return "intersection"
        
    return None

def run_inference():
    print("Loading Model...")
    with open("vocab.json") as f: vocab = json.load(f)
    model = HiVTX.load_from_checkpoint(CKPT_PATH, cvae_gan_ckpt=BACKBONE_PATH, vocab_size=len(vocab), strict=False)
    model.eval().cuda()

    print("Loading Data...")
    datamodule = NuScenesHiVTDataModule(root=DATA_ROOT, split_file="balanced_splits.json", val_batch_size=1, shuffle=True)
    datamodule.setup()
    loader = datamodule.val_dataloader()

    # Trackers
    collected_samples = []
    current_counts = {k: 0 for k in TARGET_COUNTS.keys()}
    
    print(f"Starting Search. Targets: {TARGET_COUNTS}")

    for batch_idx, batch in enumerate(loader):
        # Stop if all targets met
        if all(current_counts[k] >= TARGET_COUNTS[k] for k in TARGET_COUNTS):
            print("\nAll targets met!")
            break

        gt_ids = batch.caption_ids[0]
        gt_text = model.tokenizer.decode(gt_ids)
        
        # 1. Check Stationary (Motion < 2m)
        traj_input = batch.y[0]
        displacement = torch.norm(traj_input[-1] - traj_input[0]).item()
        if displacement < 2.0:
            continue

        # 2. Determine Category
        category = get_category(gt_text)
        if category is None:
            continue
            
        # 3. Check if we need more of this category
        if current_counts.get(category, 0) >= TARGET_COUNTS[category]:
            continue

        # 4. Run Inference
        data = batch.to(model.device)
        with torch.no_grad():
            global_embed, _ = model._get_ego_features(data)
            traj_input_batch = data.y[0].unsqueeze(0)
            
            logits = model.captioner(global_embed, traj_input_batch, captions=None, return_attn=False)
            pred_ids = logits.argmax(dim=-1)[0]
            pred_text = model.tokenizer.decode(pred_ids)

        # 5. Save Data (Same structure + added 'category' for clarity)
        sample_data = {
            "sample_idx": batch_idx,
            "city": batch.city[0],
            "category_tag": category, # Added helper tag (does not break structure, just adds info)
            "ground_truth_text": gt_text,
            "predicted_text": pred_text,
            "origin": batch.origin[0].cpu().numpy().tolist(),
            "theta": batch.theta[0].item(),
            "trajectory": traj_input.cpu().numpy().tolist()
        }
        
        collected_samples.append(sample_data)
        current_counts[category] += 1
        
        print(f"[{category.upper()}] Found {current_counts[category]}/{TARGET_COUNTS[category]}: {gt_text}")

    # 6. Save JSON
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(collected_samples, f, indent=4)
    
    print(f"\nSaved {len(collected_samples)} samples to {OUTPUT_FILE}")
    print("Final Counts:", current_counts)

if __name__ == "__main__":
    run_inference()