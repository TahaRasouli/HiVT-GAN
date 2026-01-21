import torch
import torch.nn.functional as F
import argparse
from models.hivt_x import HiVTX
from transformers import AutoTokenizer

# ============================
# CONFIGURATION
# ============================
# The "Menu" of descriptions the model can choose from.
# You can add as many as you want!
CANDIDATE_CAPTIONS = [
    "The vehicle drives straight.",
    "The vehicle executes a left turn.",
    "The vehicle executes a right turn.",
    "The vehicle performs a U-turn.",
    "The vehicle changes lanes to the left.",
    "The vehicle changes lanes to the right.",
    "The vehicle remains stationary.",
    "The vehicle is waiting at the intersection."
]

# Map class indices (0-6) to text for the Aux Head
AUX_CLASS_MAP = {
    0: "Straight Drive",
    1: "Left Turn",
    2: "Right Turn",
    3: "U-Turn",
    4: "Lane Change Left",
    5: "Lane Change Right",
    6: "Stationary Stop"
}

def load_model(ckpt_path):
    print(f"Loading model from {ckpt_path}...")
    # strict=False ensures we can load even if there are minor mismatches
    model = HiVTX.load_from_checkpoint(ckpt_path, strict=False)
    model.eval()
    model.cuda()
    return model

def prepare_text_embeddings(model, tokenizer, texts):
    """
    Pre-computes the embeddings for all candidate captions.
    """
    print("Encoding candidate texts...")
    encoded_input = tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(model.device)
    
    with torch.no_grad():
        # 1. BERT Encoding
        bert_output = model.bert(**encoded_input)
        text_features = bert_output.last_hidden_state[:, 0, :] # CLS token
        
        # 2. Projection to Joint Space
        z_text = model.proj_text(text_features)
        z_text = F.normalize(z_text, dim=1)
        
    return z_text

def run_inference(model, z_text_candidates, sample_path):
    # 1. Load Data
    try:
        data = torch.load(sample_path)
    except:
        # Fallback for weights_only=False requirement
        data = torch.load(sample_path, weights_only=False)
        
    # 2. Fix Batch Attributes (Simulate a batch of size 1)
    if not hasattr(data, 'ptr'):
        # PyG usually handles this, but manual loading needs help
        data.ptr = torch.tensor([0, data.num_nodes], dtype=torch.long)
    if not hasattr(data, 'batch'):
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
        
    data = data.to(model.device)
    
    # 3. Ground Truth Info (Just for checking)
    gt_category = getattr(data, 'maneuver_category', "Unknown")
    
    # 4. Model Forward Pass
    with torch.no_grad():
        # A. Trajectory Embedding
        traj_feat = model._get_ego_features(data) # [1, 128]
        z_traj = model.proj_traj(traj_feat)
        z_traj = F.normalize(z_traj, dim=1)
        
        # B. Aux Head Classification
        aux_logits = model.maneuver_classifier(traj_feat)
        aux_probs = F.softmax(aux_logits, dim=1) # [1, 7]

    # 5. Calculate Similarity (Contrastive Score)
    # Cosine similarity is just the dot product of normalized vectors
    similarity_scores = (z_traj @ z_text_candidates.T).squeeze() # [Num_Candidates]
    
    # ============================
    # PRINT RESULTS
    # ============================
    print(f"\n{'-'*40}")
    print(f"Analyzing: {sample_path.split('/')[-1]}")
    print(f"Ground Truth: {gt_category}")
    print(f"{'-'*40}")

    print("\n[Method 1] Contrastive Retrieval (Text Matching):")
    # Sort by score
    sorted_indices = torch.argsort(similarity_scores, descending=True)
    
    for i in sorted_indices:
        score = similarity_scores[i].item()
        text = CANDIDATE_CAPTIONS[i]
        # Highlight the winner
        prefix = ">>" if score == similarity_scores[sorted_indices[0]] else "  "
        print(f"{prefix} [{score:.4f}] {text}")

    print("\n[Method 2] Aux Classifier Head:")
    top_aux_id = torch.argmax(aux_probs).item()
    top_aux_conf = aux_probs[0, top_aux_id].item()
    print(f">> Predicted: {AUX_CLASS_MAP.get(top_aux_id, 'Unknown')} ({top_aux_conf*100:.1f}%)")
    
    print("\nRaw Probabilities:")
    for cls_id, cls_name in AUX_CLASS_MAP.items():
        if cls_id < aux_probs.shape[1]:
            print(f"   {cls_name:<18}: {aux_probs[0, cls_id].item():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to trained .ckpt")
    parser.add_argument("--sample_path", type=str, required=True, help="Path to a .pt file")
    args = parser.parse_args()

    # Init
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = load_model(args.ckpt_path)
    
    # Pre-compute text candidates once
    z_candidates = prepare_text_embeddings(model, tokenizer, CANDIDATE_CAPTIONS)
    
    # Run
    run_inference(model, z_candidates, args.sample_path)