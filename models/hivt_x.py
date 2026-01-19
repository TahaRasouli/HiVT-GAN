import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from models.cvae_gan import CVAE_GAN

class HiVTX(pl.LightningModule):
    def __init__(self, cvae_gan_ckpt, embed_dim=128, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        # 1. Backbone (HiVT)
        self.backbone = CVAE_GAN.load_from_checkpoint(cvae_gan_ckpt)
        # Unfreeze backbone for fine-tuning
        self.backbone.train()
        for param in self.backbone.parameters():
            param.requires_grad = True
            
        # 2. Text Encoder (DistilBERT)
        # 768 is the hidden size of DistilBERT
        self.bert = AutoModel.from_pretrained("distilbert-base-uncased")
        
        # Optional: Freeze lower layers of BERT to save memory/time
        # for param in self.bert.parameters():
        #     param.requires_grad = False
        
        # 3. Heads
        self.proj_traj = nn.Linear(128, embed_dim)
        self.proj_text = nn.Linear(768, embed_dim) # BERT outputs 768
        self.lane_classifier = nn.Linear(128, 5) 
        
        # 4. Loss
        self.temp = nn.Parameter(torch.tensor(0.07))
        self.ce_loss = nn.CrossEntropyLoss()
        self.validation_step_outputs = []

    def _get_ego_features(self, data):
        all_global_embed = self.backbone(data).reshape(-1, 128)
        ego_indices = data.ptr[:-1]
        return all_global_embed[ego_indices]

    def _encode_text(self, input_ids, attention_mask):
        # Run BERT
        # input_ids: [Batch, Seq]
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get CLS token (first token) as sentence summary
        # shape: [Batch, 768]
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding

    def forward(self, data):
        global_embed = self._get_ego_features(data)
        return self.proj_traj(global_embed)

    def training_step(self, data, batch_idx):
        # 1. Trajectory Features
        traj_feat = self._get_ego_features(data) # [B, 128]
        
        # 2. Text Features (BERT)
        # PyG batches squeeze dimensions sometimes, ensure [B, Seq]
        input_ids = data.input_ids.view(data.num_graphs, -1)
        mask = data.attention_mask.view(data.num_graphs, -1)
        text_feat = self._encode_text(input_ids, mask) # [B, 768]
        
        # 3. Projections
        z_traj = F.normalize(self.proj_traj(traj_feat), dim=1)
        z_text = F.normalize(self.proj_text(text_feat), dim=1)
        
        # --- STABILITY FIX: CLAMP TEMPERATURE ---
        # Ensure temp never goes below 0.01
        temp = torch.clamp(self.temp, min=0.01)

        # 4. Contrastive Loss
        logits = (z_traj @ z_text.T) / temp
        labels = torch.arange(logits.shape[0], device=self.device)
        loss = (self.ce_loss(logits, labels) + self.ce_loss(logits.T, labels)) / 2
        
        # 5. Aux Loss
        lane_logits = self.lane_classifier(traj_feat)
        valid_mask = data.lane_type_id.squeeze() != -1
        aux_loss = 0.0
        if valid_mask.sum() > 0:
            aux_loss = self.ce_loss(lane_logits[valid_mask], data.lane_type_id.squeeze()[valid_mask])
            
        total_loss = loss + (0.5 * aux_loss)
        
        self.log("train_loss", total_loss, prog_bar=True, batch_size=data.num_graphs)
        return total_loss

    def validation_step(self, data, batch_idx):
        traj_feat = self._get_ego_features(data)
        
        input_ids = data.input_ids.view(data.num_graphs, -1)
        mask = data.attention_mask.view(data.num_graphs, -1)
        text_feat = self._encode_text(input_ids, mask)
        
        z_traj = F.normalize(self.proj_traj(traj_feat), dim=1)
        z_text = F.normalize(self.proj_text(text_feat), dim=1)
        
        logits = (z_traj @ z_text.T) / self.temp
        labels = torch.arange(logits.shape[0], device=self.device)
        loss = self.ce_loss(logits, labels)
        
        self.log("val_loss", loss, prog_bar=True, batch_size=data.num_graphs)
        self.validation_step_outputs.append({"z_traj": z_traj.cpu(), "z_text": z_text.cpu()})
        return loss

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs: return
        all_traj = torch.cat([x["z_traj"] for x in self.validation_step_outputs])
        all_text = torch.cat([x["z_text"] for x in self.validation_step_outputs])
        
        similarity = all_traj @ all_text.T 
        num_samples = similarity.size(0)
        topk_indices = torch.topk(similarity, k=10, dim=1).indices 
        correct_indices = torch.arange(num_samples).view(-1, 1)
        
        r1 = (topk_indices[:, :1] == correct_indices).float().mean().item()
        r5 = (topk_indices[:, :5] == correct_indices).any(dim=1).float().mean().item()
        
        self.log("val_R1", r1, prog_bar=True)
        self.log("val_R5", r5, prog_bar=True)
        
        if self.global_rank == 0:
            val_loss = self.trainer.callback_metrics.get('val_loss', 0)
            print(f"\nEpoch {self.current_epoch:03d} | Loss: {val_loss:.4f} | R@1: {r1:.4f} | R@5: {r5:.4f}")
            
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        # We can use standard LRs now because BERT is stable
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5, weight_decay=1e-4)
        return optimizer