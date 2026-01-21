import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from models.cvae_gan import CVAE_GAN

# Map for Aux Loss (Ensure this matches your dataset!)
MANEUVER_MAP = {
    "Straight Drive": 0, "Left Turn": 1, "Right Turn": 2, "U-Turn": 3,
    "Lane Change Left": 4, "Lane Change Right": 5, "Stationary Stop": 6, "Unknown": -1
}

class HiVTX(pl.LightningModule):
    def __init__(self, cvae_gan_ckpt, embed_dim=128, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        # 1. Backbone
        self.backbone = CVAE_GAN.load_from_checkpoint(cvae_gan_ckpt)
        self.backbone.train() 
        for param in self.backbone.parameters():
            param.requires_grad = True
            
        # 2. Text Encoder
        self.bert = AutoModel.from_pretrained("distilbert-base-uncased")
        
        # 3. Heads
        self.proj_traj = nn.Linear(128, embed_dim)
        self.proj_text = nn.Linear(768, embed_dim)
        self.maneuver_classifier = nn.Linear(128, 7) # Aux Head
        
        # 4. Config
        self.temp = nn.Parameter(torch.tensor(0.07))
        self.ce_loss = nn.CrossEntropyLoss()
        
        # Storage for validation epoch end
        self.validation_step_outputs = []

    def _get_ego_features(self, data):
        all_global_embed = self.backbone(data).reshape(-1, 128)
        ego_indices = data.ptr[:-1]
        return all_global_embed[ego_indices]

    def _encode_text(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]

    def training_step(self, data, batch_idx):
        traj_feat = self._get_ego_features(data) 
        input_ids = data.input_ids.view(data.num_graphs, -1)
        mask = data.attention_mask.view(data.num_graphs, -1)
        text_feat = self._encode_text(input_ids, mask) 
        
        # Projections
        z_traj = F.normalize(self.proj_traj(traj_feat), dim=1)
        z_text = F.normalize(self.proj_text(text_feat), dim=1)
        
        # --- SUPERVISED CONTRASTIVE LOSS ---
        m_ids = data.maneuver_id.view(-1)
        
        # Create mask: 1 if samples have same label, 0 otherwise
        label_mask = torch.eq(m_ids.unsqueeze(1), m_ids.unsqueeze(0)).float()
        
        temp = torch.clamp(self.temp, min=0.01)
        logits = (z_traj @ z_text.T) / temp
        
        # LogSumExp trick for numerical stability
        exp_logits = torch.exp(logits)
        denoms = exp_logits.sum(dim=1, keepdim=True) 
        log_probs = logits - torch.log(denoms + 1e-6)
        
        # Compute mean log-prob of positive pairs
        mask_sum = label_mask.sum(dim=1)
        mean_log_prob_pos = (label_mask * log_probs).sum(dim=1) / (mask_sum + 1e-6)
        
        loss_supcon = - mean_log_prob_pos.mean()
        
        # Aux Loss
        maneuver_logits = self.maneuver_classifier(traj_feat)
        loss_aux = self.ce_loss(maneuver_logits, m_ids)
            
        total_loss = loss_supcon + (0.5 * loss_aux)
        
        self.log("train_loss", total_loss, prog_bar=True, batch_size=data.num_graphs)
        return total_loss

    def validation_step(self, data, batch_idx):
        traj_feat = self._get_ego_features(data)
        input_ids = data.input_ids.view(data.num_graphs, -1)
        mask = data.attention_mask.view(data.num_graphs, -1)
        text_feat = self._encode_text(input_ids, mask)
        
        z_traj = F.normalize(self.proj_traj(traj_feat), dim=1)
        z_text = F.normalize(self.proj_text(text_feat), dim=1)
        
        # Store for epoch-level metric calculation
        self.validation_step_outputs.append({
            "z_traj": z_traj.cpu(), 
            "z_text": z_text.cpu(),
            "labels": data.maneuver_id.view(-1).cpu() # Store class labels too!
        })
        
        # Val loss (approximate for logging)
        logits = (z_traj @ z_text.T) / 0.07
        loss = self.ce_loss(logits, torch.arange(logits.shape[0], device=self.device))
        self.log("val_loss", loss, prog_bar=True, batch_size=data.num_graphs)
        return loss

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs: return
        
        all_traj = torch.cat([x["z_traj"] for x in self.validation_step_outputs])
        all_text = torch.cat([x["z_text"] for x in self.validation_step_outputs])
        all_labels = torch.cat([x["labels"] for x in self.validation_step_outputs])
        
        # Calculate Similarity Matrix (N x N)
        # Note: Doing this on CPU for the whole validation set might be memory intensive.
        # If OOM occurs, we compute in chunks or just use the last batch.
        # For ~4k val samples, it's a 4k x 4k matrix (16M floats = 64MB), which is fine.
        
        similarity = all_traj @ all_text.T 
        
        # --- METRIC 1: Instance Retrieval (Standard CLIP) ---
        # "Did I find the EXACT index?"
        num_samples = similarity.size(0)
        topk_indices = torch.topk(similarity, k=5, dim=1).indices 
        correct_indices = torch.arange(num_samples).view(-1, 1)
        
        r1_instance = (topk_indices[:, :1] == correct_indices).float().mean().item()
        r5_instance = (topk_indices[:, :5] == correct_indices).any(dim=1).float().mean().item()
        
        # --- METRIC 2: Semantic Retrieval (The Real Metric) ---
        # "Did I find a sample with the SAME CLASS?"
        
        # Get the class label of the retrieved item
        # topk_indices is [N, 5]. We use it to index into all_labels [N]
        retrieved_labels_top1 = all_labels[topk_indices[:, 0]] 
        
        # Check equality with query's own label
        semantic_acc_1 = (retrieved_labels_top1 == all_labels).float().mean().item()
        
        self.log("val_R1_inst", r1_instance, prog_bar=True)
        self.log("val_R5_inst", r5_instance, prog_bar=True)
        self.log("val_Sem_Acc", semantic_acc_1, prog_bar=True) # <--- Watch this one!
        
        if self.global_rank == 0:
            print(f"\nEpoch {self.current_epoch} | Inst R@1: {r1_instance:.4f} | Semantic Acc: {semantic_acc_1:.4f}")
            
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4)