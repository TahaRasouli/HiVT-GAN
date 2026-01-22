import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score, ConfusionMatrix
from utils import VariationalCaptionGenerator

class ManeuverClassifier(pl.LightningModule):
    def __init__(self, frozen_backbone, num_classes=7, learning_rate=1e-3, class_weights=None):
        super().__init__()
        self.save_hyperparameters(ignore=['frozen_backbone'])
        
        # 1. The Backbone (Frozen)
        self.backbone = frozen_backbone
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 2. The Head (Trainable)
        # Input: 128 (HiVT embedding) -> Output: 7 (Maneuver Classes)
        self.head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )
        
        # 3. Loss & Metrics
        # Weights are passed from the training script to handle imbalance
        self.class_weights = class_weights
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        
        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        
        # We track F1 score per-class to see if we are failing at U-Turns specifically
        self.val_f1_per_class = F1Score(task="multiclass", num_classes=num_classes, average=None)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        
        # Class Names for pretty printing
        self.class_names = [
            "Straight", "Left Turn", "Right Turn", "U-Turn", 
            "LC Left", "LC Right", "Stationary"
        ]

    def forward(self, batch):
        self.backbone.eval()
        with torch.no_grad():
            # HiVT returns [Total_Nodes, 128]
            global_embed = self.backbone(batch)
        
        # Ego Selection: Ego is always at batch.ptr[:-1]
        ego_indices = batch.ptr[:-1]
        ego_embeds = global_embed[ego_indices]
        
        logits = self.head(ego_embeds)
        return logits

    def training_step(self, batch, batch_idx):
        logits = self(batch)
        targets = batch.maneuver_id.squeeze().long()
        
        loss = self.criterion(logits, targets)
        preds = torch.argmax(logits, dim=1)
        
        self.train_acc(preds, targets)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        targets = batch.maneuver_id.squeeze().long()
        
        loss = self.criterion(logits, targets)
        preds = torch.argmax(logits, dim=1)
        
        # Update metrics
        self.val_acc(preds, targets)
        self.val_f1_per_class(preds, targets)
        
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        """
        Prints a results table to the terminal at the end of every epoch.
        """
        # Get the computed metrics from the accumulator
        f1_scores = self.val_f1_per_class.compute()
        acc = self.val_acc.compute()
        
        # Print to console
        print(f"\n{'='*40}")
        print(f"Epoch {self.current_epoch} Results")
        print(f"{'-'*40}")
        print(f"Overall Accuracy: {acc:.4f}")
        print(f"{'-'*40}")
        print(f"{'Class':<15} | {'F1 Score':<10}")
        print(f"{'-'*40}")
        
        for i, name in enumerate(self.class_names):
            score = f1_scores[i].item()
            print(f"{name:<15} | {score:.4f}")
            
        print(f"{'='*40}\n")
        
        # Reset for next epoch
        self.val_f1_per_class.reset()
        self.val_acc.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.head.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }