import pytorch_lightning as pl
import torch
import torch.nn as nn
from models.cvae_gan import CVAE_GAN
from models import TrajectoryCaptioner

class HiVTX(pl.LightningModule):
    def __init__(self, cvae_gan_ckpt, vocab_size, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        # 1. Load Pretrained Backbone (FROZEN)
        print(f"--- Loading Backbone from {cvae_gan_ckpt} ---")
        self.backbone = CVAE_GAN.load_from_checkpoint(cvae_gan_ckpt)
        self.backbone.freeze() 
        self.backbone.eval()
        
        # 2. Initialize the Explainer
        self.captioner = TrajectoryCaptioner(vocab_size=vocab_size)
        
        # 3. Loss
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0) 

    def forward(self, data):
        with torch.no_grad():
            global_embed = self.backbone(data)
            global_embed = global_embed.reshape(-1, 128) 
        return self.captioner(global_embed, data.y)

    def training_step(self, data, batch_idx):
        with torch.no_grad():
            global_embed = self.backbone(data)
            global_embed = global_embed.reshape(-1, 128)
            
        logits = self.captioner(global_embed, data.y, captions=data.caption_ids)
        loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), data.caption_ids.reshape(-1))
        
        self.log("train_cap_loss", loss, prog_bar=True, batch_size=data.num_graphs)
        return loss

    def validation_step(self, data, batch_idx):
        with torch.no_grad():
            global_embed = self.backbone(data).reshape(-1, 128)
            logits = self.captioner(global_embed, data.y, captions=data.caption_ids)
            loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), data.caption_ids.reshape(-1))
            self.log("val_cap_loss", loss, prog_bar=True, batch_size=data.num_graphs)

    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics
        if self.global_rank == 0:
            val_loss = metrics.get('val_cap_loss', 0.0)
            train_loss = metrics.get('train_cap_loss', 0.0)
            print(f"\nEpoch {self.current_epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.captioner.parameters(), lr=1e-3, weight_decay=1e-5)