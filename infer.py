import torch
import numpy as np
import json
import os
from transformers import AutoTokenizer
from models.hivt_x import HiVTX
from datamodules.nuscenes_datamodule import NuScenesHiVTDataModule
import torch.nn.functional as F

# --- CONFIGURATION ---
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

# --- 1. DEFINE CANDIDATE CAPTIONS (The "Vocabulary" for Retrieval) ---
# Since the model selects the best match, we define standard descriptions.
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

def get_category_from_gt(text):
    """Filters Ground Truth to find specific examples."""
    text = text.lower()
    if "u-turn" in text: return "u_turn"
    if "lane change" in text or "change lane" in text: return "lane_change"
    if "turn" in text: return "turn" # Covers left/right
    if "intersection" in text: return "intersection"
    return None

def run_inference():
    print("Loading Model...")
    # Load Contrastive Model
    model = HiVTX.load_from_checkpoint(CKPT_PATH, cvae_gan_ckpt=BACKBONE_PATH)
    model.eval().cuda()

    print("Loading Data...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # We use batch_size=1 to process samples one by one
    datamodule = NuScenesHiVTDataModule(
        root=DATA_ROOT, 
        split_file="balanced_splits.json", 
        val_batch_size=1, 
        shuffle=True, # Randomize to find variety
        tokenizer=tokenizer
    )
    datamodule.setup()
    loader = datamodule.val_dataloader()

    # --- 2. PRE-COMPUTE TEXT EMBEDDINGS (The "Dictionary") ---
    print("Encoding Candidate Texts...")
    with torch.no_grad():
        inputs = tokenizer(CANDIDATE_TEXTS, return_tensors="pt", padding=True, truncation=True).to(model.device)
        # Encode all candidates once
        text_feats = model._encode_text(inputs.input_ids, inputs.attention_mask)
        z_candidates = F.normalize(model.proj_text(text_feats), dim=1)

    collected_samples = []
    current_counts = {k: 0 for k in TARGET_COUNTS.keys()}
    
    print(f"Starting Search...")

    for batch_idx, batch in enumerate(loader):
        if all(current_counts[k] >= TARGET_COUNTS[k] for k in TARGET_COUNTS):
            print("\nAll targets met!")
            break

        # 1. Get Ground Truth Text (for filtering)
        # Note: batch.input_ids is tokenized, we can't easily read it back to raw text perfectly.
        # But we stored raw attributes in the dataset if available, or we decode roughly.
        gt_text = tokenizer.decode(batch.input_ids[0], skip_special_tokens=True)

        # 2. Determine Category (Is this a sample we want?)
        category = get_category_from_gt(gt_text)
        
        # Skip straight drives or categories we filled up
        if category is None or current_counts.get(category, 0) >= TARGET_COUNTS[category]:
            continue
        
        # 3. RUN INFERENCE (Retrieval)
        batch = batch.to(model.device)
        with torch.no_grad():
            # Embed Trajectory
            traj_feat = model._get_ego_features(batch)
            z_traj = F.normalize(model.proj_traj(traj_feat), dim=1)
            
            # Compare Trajectory vs All Candidates
            # Dot product: [1, 128] @ [8, 128].T -> [1, 8] scores
            scores = (z_traj @ z_candidates.T).squeeze()
            
            # Pick best match
            best_idx = scores.argmax().item()
            pred_text = CANDIDATE_TEXTS[best_idx]
            confidence = scores[best_idx].item()

        # 4. Save
        # Extract raw trajectory for visualization
        traj_input = batch.y[0].cpu().numpy().tolist()
        
        sample_data = {
            "sample_idx": batch_idx,
            "category": category,
            "ground_truth_text": gt_text,
            "predicted_text": pred_text,
            "confidence": round(confidence, 4),
            "origin": batch.origin[0].cpu().numpy().tolist(),
            "trajectory": traj_input
        }
        
        collected_samples.append(sample_data)
        current_counts[category] += 1
        
        print(f"[{category.upper()}] GT: '{gt_text[:30]}...' -> PRED: '{pred_text}' ({confidence:.2f})")

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(collected_samples, f, indent=4)
    
    print(f"\nSaved results to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_inference()