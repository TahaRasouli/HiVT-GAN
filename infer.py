import torch
import numpy as np
import json
import os
import torch.nn.functional as F
from transformers import AutoTokenizer
from models.hivt_x import HiVTX
from datamodules.nuscenes_datamodule import NuScenesHiVTDataModule

# ==========================================
# CONFIGURATION
# ==========================================
# Update these paths to match your exact file locations
CKPT_PATH = "/mount/arbeitsdaten/studenten4/rasoulta/HiVT-GAN/lightning_logs/version_62/checkpoints/epoch=17-step=1296.ckpt"
BACKBONE_PATH = "/mount/studenten/projects/rasoulta/checkpoints/vae-gan-baseline/checkpoints/epoch=45-step=60812.ckpt"
DATA_ROOT = "/mount/studenten/projects/rasoulta/dataset"
OUTPUT_FILE = "inference_results.json"

# Targets: How many samples we want to find for each category
TARGET_COUNTS = {
    "lane_change": 10,
    "turn": 10,
    "u_turn": 10,
    "intersection": 10
}

# The "Menu" of descriptions the model can choose from
CANDIDATE_TEXTS = [
    "The vehicle drives straight.",
    "The vehicle turns left.",
    "The vehicle turns right.",
    "The vehicle changes lane to the left.",
    "The vehicle changes lane to the right.",
    "The vehicle performs a U-turn.",
    "The vehicle stops.",
    "The vehicle moves slowly at an intersection."
]

# ==========================================
# UTILITIES
# ==========================================
def get_category_from_gt(text):
    """Filters Ground Truth to find specific examples."""
    text = text.lower()
    if "u-turn" in text: return "u_turn"
    if "lane change" in text or "change lane" in text: return "lane_change"
    if "turn" in text: return "turn" 
    if "intersection" in text: return "intersection"
    return None

def run_inference():
    print(f"--- Starting Inference ---")
    
    # 1. Load Model
    print(f"Loading Model from: {CKPT_PATH}")
    # We must pass the backbone path as it was required in __init__
    model = HiVTX.load_from_checkpoint(CKPT_PATH, cvae_gan_ckpt=BACKBONE_PATH, strict=False)
    model.eval().cuda()

    # 2. Load Data
    print("Setting up DataModule...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    datamodule = NuScenesHiVTDataModule(
        root=DATA_ROOT, 
        split_file="balanced_splits.json", 
        train_batch_size=1, # Dummy value to satisfy __init__
        val_batch_size=1,   # Process 1 at a time for analysis
        shuffle=True,       # Randomize to find variety
        tokenizer=tokenizer
    )
    datamodule.setup()
    loader = datamodule.val_dataloader()

    # 3. Pre-compute Text Embeddings (The "Anchors")
    print("Encoding Candidate Texts...")
    with torch.no_grad():
        inputs = tokenizer(CANDIDATE_TEXTS, return_tensors="pt", padding=True, truncation=True).to(model.device)
        text_feats = model._encode_text(inputs.input_ids, inputs.attention_mask)
        z_candidates = F.normalize(model.proj_text(text_feats), dim=1)

    # 4. Search Loop
    collected_samples = []
    current_counts = {k: 0 for k in TARGET_COUNTS.keys()}
    
    print(f"Searching for samples... Targets: {TARGET_COUNTS}")

    for batch_idx, batch in enumerate(loader):
        # Stop if we found enough of everything
        if all(current_counts[k] >= TARGET_COUNTS[k] for k in TARGET_COUNTS):
            print("\nAll targets met!")
            break

        # Get Raw Ground Truth Text (roughly decoded)
        gt_text = tokenizer.decode(batch.input_ids[0], skip_special_tokens=True)

        # Categorize this sample based on GT
        category = get_category_from_gt(gt_text)
        
        # Skip if it's 'straight' (None) or if we already have enough of this category
        if category is None or current_counts.get(category, 0) >= TARGET_COUNTS[category]:
            continue
        
        # --- MODEL PREDICTION ---
        batch = batch.to(model.device)
        with torch.no_grad():
            # Get Trajectory Embedding
            traj_feat = model._get_ego_features(batch)
            z_traj = F.normalize(model.proj_traj(traj_feat), dim=1)
            
            # Compare vs Candidates (Dot Product)
            scores = (z_traj @ z_candidates.T).squeeze()
            
            # Pick Winner
            best_idx = scores.argmax().item()
            pred_text = CANDIDATE_TEXTS[best_idx]
            confidence = scores[best_idx].item()

        # 5. Save Data (Ensure 'city' and 'theta' are included!)
        traj_input = batch.y[0].cpu().numpy().tolist()
        
        sample_data = {
            "sample_idx": batch_idx,
            "city": batch.city[0],           # <--- CRITICAL FOR VISUALIZATION
            "theta": batch.theta[0].item(),  # <--- CRITICAL FOR ROTATION
            "origin": batch.origin[0].cpu().numpy().tolist(),
            "category": category,
            "ground_truth_text": gt_text,
            "predicted_text": pred_text,
            "confidence": round(confidence, 4),
            "trajectory": traj_input
        }
        
        collected_samples.append(sample_data)
        current_counts[category] += 1
        
        print(f"[{category.upper()}] Conf: {confidence:.2f} | GT: {gt_text[:40]}... -> PRED: {pred_text}")

    # 6. Dump to JSON
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(collected_samples, f, indent=4)
    
    print(f"\nSaved {len(collected_samples)} samples to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_inference()