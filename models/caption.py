import pytorch_lightning as pl
import torch
import torch.nn as nn
from models.cvae_gan import CVAE_GAN

class CaptionFinetuner(pl.LightningModule):
    def __init__(self, pretrained_ckpt):
        super().__init__()
        self.save_hyperparameters()
        
        # 1. Load Backbone (strict=False to ignore missing caption_head)
        print(f"Loading backbone from {pretrained_ckpt}...")
        self.model = CVAE_GAN.load_from_checkpoint(pretrained_ckpt, strict=False)
        
        # 2. Freeze Backbone
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
            
        # 3. Unfreeze Caption Head
        for param in self.model.decoder.caption_head.parameters():
            param.requires_grad = True
            
        # 4. WEIGHTED LOSS (The Fix)
        # Weights roughly inverse to frequency to prevent "Straight Drive" bias
        weights = torch.tensor([
            1.0,   # 0: Straight
            15.0,  # 1: Left
            15.0,  # 2: Right
            50.0,  # 3: U-Turn
            25.0,  # 4: Lane L
            25.0,  # 5: Lane R
            5.0    # 6: Stop
        ])
        self.register_buffer("class_weights", weights)
        
        self.ce_loss = nn.CrossEntropyLoss(weight=self.class_weights)
        self.validation_step_outputs = []

    def forward(self, data):
        # Inference: Prior -> Z -> Decoder -> Logits
        local_embed = self.model.local_encoder(data)
        global_embed = self.model.global_interactor(data, local_embed)
        _, _, caption_logits = self.model.decoder(global_embed, y_gt=None)
        return caption_logits

    def training_step(self, data, batch_idx):
        logits = self(data)
        target = data.maneuver_label.squeeze()
        
        loss = self.ce_loss(logits, target)
        
        # Log accuracy (non-weighted, just raw correctness)
        acc = (torch.argmax(logits, dim=1) == target).float().mean()
        self.log("train_loss", loss, prog_bar=True, batch_size=data.num_graphs)
        self.log("train_acc", acc, prog_bar=True, batch_size=data.num_graphs)
        return loss

    def validation_step(self, data, batch_idx):
        logits = self(data)
        target = data.maneuver_label.squeeze()
        loss = self.ce_loss(logits, target)
        
        preds = torch.argmax(logits, dim=1)
        correct = (preds == target).float().sum()
        total = torch.tensor(target.numel(), device=self.device)
        
        self.validation_step_outputs.append({
            "loss": loss,
            "correct": correct,
            "total": total
        })
        return loss

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs: return
        
        avg_loss = torch.stack([x["loss"] for x in self.validation_step_outputs]).mean()
        total_correct = torch.stack([x["correct"] for x in self.validation_step_outputs]).sum()
        total_samples = torch.stack([x["total"] for x in self.validation_step_outputs]).sum()
        
        val_acc = total_correct / total_samples
        
        self.log("val_loss", avg_loss, prog_bar=True)
        self.log("val_acc", val_acc, prog_bar=True)
        
        if self.global_rank == 0:
            print(f"\n[Epoch {self.current_epoch}] Val Acc: {val_acc*100:.2f}% | Loss: {avg_loss:.4f}")
            
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.decoder.caption_head.parameters(), lr=1e-3)