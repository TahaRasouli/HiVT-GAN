import pytorch_lightning as pl
import torch
import torch.nn as nn
from models.cvae_gan import CVAE_GAN
from models.captioner import TrajectoryCaptioner

class HiVTX(pl.LightningModule):
    def __init__(self, cvae_gan_ckpt, vocab_size, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        # 1. Load Pretrained Backbone (FROZEN)
        print(f"--- Loading Backbone from {cvae_gan_ckpt} ---")
        self.backbone = CVAE_GAN.load_from_checkpoint(cvae_gan_ckpt)
        self.backbone.freeze() # Freezes all params in CVAE-GAN
        self.backbone.eval()   # Sets dropout/batchnorm to eval mode
        
        # 2. Initialize the Explainer (Trainable)
        self.captioner = TrajectoryCaptioner(vocab_size=vocab_size)
        
        # 3. Loss (Ignore padding index 0)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0) 

    def forward(self, data):
        """Inference Forward Pass"""
        # Extract features from frozen backbone
        with torch.no_grad():
            global_embed = self.backbone(data)
            global_embed = global_embed.reshape(-1, 128) 
            
        # For Inference, we usually pass the Predicted Trajectory, 
        # but in forward() we assume we want to caption the Ground Truth 
        # or handle inputs flexibly.
        return self.captioner(global_embed, data.y)

    def training_step(self, data, batch_idx):
        # 1. Feature Extraction (No Gradients)
        with torch.no_grad():
            global_embed = self.backbone(data)
            global_embed = global_embed.reshape(-1, 128)
            
        # 2. Caption Generation (Gradients ON)
        # data.caption_ids comes from your updated dataloader
        logits = self.captioner(global_embed, data.y, captions=data.caption_ids)
        
        # 3. Calculate Loss
        # Logits: [B, Seq, Vocab] -> [B*Seq, Vocab]
        # Targets: [B, Seq] -> [B*Seq]
        targets = data.caption_ids
        loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        
        self.log("train_cap_loss", loss, prog_bar=True, batch_size=data.num_graphs)
        return loss

    def validation_step(self, data, batch_idx):
        # Same logic as training, but no Teacher Forcing (captions=None inside captioner logic if you handled it)
        # Ideally, we calculate BLEU scores here, but Loss is fine for now.
        with torch.no_grad():
            global_embed = self.backbone(data).reshape(-1, 128)
            logits = self.captioner(global_embed, data.y, captions=data.caption_ids)
            loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), data.caption_ids.reshape(-1))
            self.log("val_cap_loss", loss, prog_bar=True, batch_size=data.num_graphs)

    def configure_optimizers(self):
        # CRITICAL: Only optimize self.captioner
        return torch.optim.AdamW(self.captioner.parameters(), lr=1e-3, weight_decay=1e-5)