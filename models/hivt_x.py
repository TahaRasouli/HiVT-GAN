import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from models.cvae_gan import CVAE_GAN

# Map your string categories to integers for the Aux Loss
MANEUVER_MAP = {
    "Straight Drive": 0,
    "Left Turn": 1,
    "Right Turn": 2,
    "U-Turn": 3,
    "Lane Change Left": 4,
    "Lane Change Right": 5,
    "Stationary Stop": 6
}

class HiVTX(pl.LightningModule):
    def __init__(self, cvae_gan_ckpt, embed_dim=128, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        # 1. Backbone (HiVT) - Loads Pretrained Weights
        self.backbone = CVAE_GAN.load_from_checkpoint(cvae_gan_ckpt)
        self.backbone.train() # Ensure gradients flow
        for param in self.backbone.parameters():
            param.requires_grad = True
            
        # 2. Text Encoder (DistilBERT)
        self.bert = AutoModel.from_pretrained("distilbert-base-uncased")
        
        # 3. Heads
        self.proj_traj = nn.Linear(128, embed_dim)
        self.proj_text = nn.Linear(768, embed_dim)
        
        # AUXILIARY HEAD: 7 Classes (geometric maneuvers)
        self.maneuver_classifier = nn.Linear(128, 7) 
        
        # 4. Loss Config
        self.temp = nn.Parameter(torch.tensor(0.07))
        self.ce_loss = nn.CrossEntropyLoss()
        self.validation_step_outputs = []

    def _get_ego_features(self, data):
        # Extract the embedding for the Ego Vehicle (Index 0)
        all_global_embed = self.backbone(data).reshape(-1, 128)
        ego_indices = data.ptr[:-1] # PyG batching trick to get first node of every graph
        return all_global_embed[ego_indices]

    def _encode_text(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :] # CLS Token

    def forward(self, data):
        # For inference: just return trajectory embedding
        global_embed = self._get_ego_features(data)
        return self.proj_traj(global_embed)

    def training_step(self, data, batch_idx):
        # 1. Trajectory Features [Batch, 128]
        traj_feat = self._get_ego_features(data) 
        
        # 2. Text Features [Batch, 768]
        # Reshape to handle PyG batching quirks
        input_ids = data.input_ids.view(data.num_graphs, -1)
        mask = data.attention_mask.view(data.num_graphs, -1)
        text_feat = self._encode_text(input_ids, mask) 
        
        # 3. Projections
        z_traj = F.normalize(self.proj_traj(traj_feat), dim=1)
        z_text = F.normalize(self.proj_text(text_feat), dim=1)
        
        # 4. Contrastive Loss (Symmetric)
        temp = torch.clamp(self.temp, min=0.01) # Stability Fix
        logits = (z_traj @ z_text.T) / temp
        labels = torch.arange(logits.shape[0], device=self.device)
        loss_contrastive = (self.ce_loss(logits, labels) + self.ce_loss(logits.T, labels)) / 2
        
        # 5. Aux Loss (Maneuver Classification)
        # Convert string list to tensor indices
        maneuver_logits = self.maneuver_classifier(traj_feat)
        
        # Handle string collation from PyG
        raw_cats = data.maneuver_category
        # Flatten list of lists if necessary (depends on PyG version)
        if isinstance(raw_cats[0], list): 
            raw_cats = [item for sublist in raw_cats for item in sublist]
            
        target_indices = torch.tensor(
            [MANEUVER_MAP.get(c, 0) for c in raw_cats], 
            device=self.device
        )
        
        loss_aux = self.ce_loss(maneuver_logits, target_indices)
            
        # Weighted Sum
        total_loss = loss_contrastive + (0.5 * loss_aux)
        
        self.log("train_loss", total_loss, prog_bar=True, batch_size=data.num_graphs)
        self.log("temp", temp, prog_bar=False)
        return total_loss

    def validation_step(self, data, batch_idx):
        traj_feat = self._get_ego_features(data)
        
        input_ids = data.input_ids.view(data.num_graphs, -1)
        mask = data.attention_mask.view(data.num_graphs, -1)
        text_feat = self._encode_text(input_ids, mask)
        
        z_traj = F.normalize(self.proj_traj(traj_feat), dim=1)
        z_text = F.normalize(self.proj_text(text_feat), dim=1)
        
        logits = (z_traj @ z_text.T) / torch.clamp(self.temp, min=0.01)
        labels = torch.arange(logits.shape[0], device=self.device)
        loss = self.ce_loss(logits, labels)
        
        self.log("val_loss", loss, prog_bar=True, batch_size=data.num_graphs)
        self.validation_step_outputs.append({"z_traj": z_traj.cpu(), "z_text": z_text.cpu()})
        return loss

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs: return
        all_traj = torch.cat([x["z_traj"] for x in self.validation_step_outputs])
        all_text = torch.cat([x["z_text"] for x in self.validation_step_outputs])
        
        # Calculate Recall@K
        similarity = all_traj @ all_text.T 
        num_samples = similarity.size(0)
        # Top-K retrieval
        topk_indices = torch.topk(similarity, k=5, dim=1).indices 
        correct_indices = torch.arange(num_samples).view(-1, 1)
        
        r1 = (topk_indices[:, :1] == correct_indices).float().mean().item()
        r5 = (topk_indices[:, :5] == correct_indices).any(dim=1).float().mean().item()
        
        self.log("val_R1", r1, prog_bar=True)
        self.log("val_R5", r5, prog_bar=True)
        
        if self.global_rank == 0:
            print(f"\nEpoch {self.current_epoch} | R@1: {r1:.4f} | R@5: {r5:.4f}")
            
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=5e-5, weight_decay=1e-4)