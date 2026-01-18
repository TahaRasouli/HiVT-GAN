import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cvae_gan import CVAE_GAN
from utils import SimpleTokenizer

class HiVTX(pl.LightningModule):
    def __init__(self, cvae_gan_ckpt, vocab_size, embed_dim=128, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        # ------------------------------------------------------------------
        # 1. ARCHITECTURE
        # ------------------------------------------------------------------
        
        # A. Backbone (UNFROZEN)
        self.backbone = CVAE_GAN.load_from_checkpoint(cvae_gan_ckpt)
        
        # --- CRITICAL CHANGE: UNFREEZE ---
        self.backbone.train() 
        for param in self.backbone.parameters():
            param.requires_grad = True
        # ---------------------------------

        # B. Text Encoder
        self.text_embedding = nn.Embedding(vocab_size, 256)
        self.text_encoder = nn.GRU(256, 256, batch_first=True)
        
        # C. Heads
        self.proj_traj = nn.Linear(128, embed_dim) 
        self.proj_text = nn.Linear(256, embed_dim) 
        self.lane_classifier = nn.Linear(128, 5) 
        
        # D. Training Components
        self.temp = nn.Parameter(torch.tensor(0.07)) 
        self.ce_loss = nn.CrossEntropyLoss()
        self.tokenizer = SimpleTokenizer(vocab_file="vocab.json") 
        self.validation_step_outputs = []
        
        # ------------------------------------------------------------------
        # 2. TRAINING COMPONENTS
        # ------------------------------------------------------------------
        self.temp = nn.Parameter(torch.tensor(0.07)) # Learnable temperature for InfoNCE
        self.ce_loss = nn.CrossEntropyLoss()
        self.tokenizer = SimpleTokenizer(vocab_file="vocab.json") 
        
        # Storage for calculating Recall@K at the end of every epoch
        self.validation_step_outputs = []

    def _get_ego_features(self, data):
        """Extracts only the Ego Vehicle (Node 0) features from the graph batch."""
        with torch.no_grad():
            all_global_embed = self.backbone(data).reshape(-1, 128)
        
        # data.ptr points to the start of each graph. Ego is always the first node.
        ego_indices = data.ptr[:-1]
        return all_global_embed[ego_indices] # [Batch_Size, 128]

    def _encode_text(self, caption_ids):
        """Encodes text tokens into a single vector."""
        x = self.text_embedding(caption_ids) # [B, Seq_Len, 256]
        _, h_n = self.text_encoder(x)        # h_n is [1, B, 256] (Last hidden state)
        return h_n.squeeze(0)                # [B, 256]

    def forward(self, data):
        """Inference: Returns the projected Trajectory Embedding."""
        global_embed = self._get_ego_features(data)
        return self.proj_traj(global_embed)

    def training_step(self, data, batch_idx):
        # 1. Feature Extraction
        traj_feat = self._get_ego_features(data)        # [B, 128]
        text_feat = self._encode_text(data.caption_ids) # [B, 256]
        
        # 2. Projection & Normalization (Crucial for Contrastive Learning)
        z_traj = F.normalize(self.proj_traj(traj_feat), dim=1)
        z_text = F.normalize(self.proj_text(text_feat), dim=1)
        
        # 3. Contrastive Loss (InfoNCE)
        # logits: [B, B] similarity matrix
        logits = (z_traj @ z_text.T) / self.temp
        labels = torch.arange(logits.shape[0], device=self.device)
        
        # Symmetric Loss (Image->Text and Text->Image)
        loss_i = self.ce_loss(logits, labels)
        loss_t = self.ce_loss(logits.T, labels)
        contrastive_loss = (loss_i + loss_t) / 2
        
        # 4. Auxiliary Loss (Lane Type Classification)
        lane_logits = self.lane_classifier(traj_feat)
        valid_mask = data.lane_type_id.squeeze() != -1
        
        if valid_mask.sum() > 0:
            aux_loss = self.ce_loss(lane_logits[valid_mask], data.lane_type_id.squeeze()[valid_mask])
        else:
            aux_loss = 0.0
            
        total_loss = contrastive_loss + (0.5 * aux_loss)
        
        # Logging
        self.log("train_loss", total_loss, prog_bar=True, batch_size=data.num_graphs)
        self.log("train_cont_loss", contrastive_loss, batch_size=data.num_graphs)
        self.log("temp", self.temp, batch_size=data.num_graphs)
        
        return total_loss

    def validation_step(self, data, batch_idx):
        """
        Calculates loss and stores embeddings for global Recall@K calculation.
        """
        # 1. Get Embeddings
        traj_feat = self._get_ego_features(data)
        text_feat = self._encode_text(data.caption_ids)
        
        z_traj = F.normalize(self.proj_traj(traj_feat), dim=1)
        z_text = F.normalize(self.proj_text(text_feat), dim=1)
        
        # 2. Calculate Val Loss (Proxy for convergence)
        logits = (z_traj @ z_text.T) / self.temp
        labels = torch.arange(logits.shape[0], device=self.device)
        loss = self.ce_loss(logits, labels)
        
        self.log("val_loss", loss, prog_bar=True, batch_size=data.num_graphs)
        
        # 3. Store for Epoch End
        # Move to CPU to prevent GPU OOM during large validations
        self.validation_step_outputs.append({
            "z_traj": z_traj.cpu(), 
            "z_text": z_text.cpu()
        })
        return loss

    def on_validation_epoch_end(self):
        """
        Computes Retrieval Metrics (R@1, R@5, R@10) across the entire validation set.
        """
        if not self.validation_step_outputs:
            return

        # 1. Concatenate all batches
        all_traj = torch.cat([x["z_traj"] for x in self.validation_step_outputs])
        all_text = torch.cat([x["z_text"] for x in self.validation_step_outputs])
        
        # 2. Compute Global Similarity Matrix
        # [N_val, N_val] matrix
        similarity = all_traj @ all_text.T 
        
        num_samples = similarity.size(0)
        
        # 3. Get Top-K Indices for every sample
        # For every trajectory (row), find the indices of the text (cols) with highest dot product
        topk_indices = torch.topk(similarity, k=10, dim=1).indices 
        
        # The correct index for the i-th trajectory is the i-th text
        correct_indices = torch.arange(num_samples).view(-1, 1)
        
        # 4. Calculate Recall
        # R@1: Is the correct text the #1 match?
        r1 = (topk_indices[:, :1] == correct_indices).float().mean().item()
        
        # R@5: Is the correct text in the top 5?
        r5 = (topk_indices[:, :5] == correct_indices).any(dim=1).float().mean().item()
        
        # R@10: Is the correct text in the top 10?
        r10 = (topk_indices[:, :10] == correct_indices).any(dim=1).float().mean().item()
        
        # 5. Log & Print
        self.log("val_R1", r1, prog_bar=True)
        self.log("val_R5", r5, prog_bar=True)
        self.log("val_R10", r10, prog_bar=False)
        
        if self.global_rank == 0:
            val_loss = self.trainer.callback_metrics.get('val_loss', 0)
            print(f"\nEpoch {self.current_epoch:03d} | Loss: {val_loss:.4f} | R@1: {r1:.4f} | R@5: {r5:.4f}")
            
        # 6. Cleanup
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        # 1. Separate Parameters
        backbone_params = list(self.backbone.parameters())
        head_params = [p for n, p in self.named_parameters() if "backbone" not in n]
        
        # 2. Aggressive Learning Rates
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': 1e-4}, # Boosted from 1e-5
            {'params': head_params, 'lr': 1e-3}      # Boosted from 1e-4
        ], weight_decay=1e-4) # Slightly higher decay to prevent overfitting
        
        # 3. Add a Scheduler (Optional but recommended)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1
            }
        }