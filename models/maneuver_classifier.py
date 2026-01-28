import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score

class ManeuverClassifier(pl.LightningModule):
    """
    Ego-centric maneuver classifier with Context Injection and Fine-Tuning.
    
    Fixed: Manually runs backbone encoders to extract embeddings, ignoring the trajectory decoder.
    """

    def __init__(
        self,
        frozen_backbone,
        num_classes: int = 7,
        learning_rate: float = 1e-3,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["frozen_backbone"])

        # 1. BACKBONE
        self.backbone = frozen_backbone
        # No freezing loop -> We allow fine-tuning

        # 2. CLASSIFICATION HEAD
        # Input: 128 (Trajectory) + 3 (Map Hint) = 131
        self.head = nn.Sequential(
            nn.Linear(128 + 3, 128), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

        # 3. METRICS
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1_per_class = F1Score(task="multiclass", num_classes=num_classes, average=None)

        self.class_names = [
            "Straight", "Left Turn", "Right Turn", "U-Turn",
            "LC Left", "LC Right", "Stationary",
        ]

    def forward(self, batch):
        # ----------------------------------------------------------
        # A. BACKBONE BYPASS (The Fix)
        # ----------------------------------------------------------
        # We cannot call self.backbone(batch) because it returns a tuple (y_hat, pi).
        # We must manually run the encoder steps to get the embedding.
        
        # 1. Generate Rotation Matrix (Required by LocalEncoder)
        # Check if backbone wants rotation (default to True)
        rotate = getattr(self.backbone.hparams, 'rotate', True)
        if rotate:
            rotate_mat = torch.empty(batch.num_nodes, 2, 2, device=self.device)
            sin_vals = torch.sin(batch['rotate_angles'])
            cos_vals = torch.cos(batch['rotate_angles'])
            rotate_mat[:, 0, 0] = cos_vals
            rotate_mat[:, 0, 1] = -sin_vals
            rotate_mat[:, 1, 0] = sin_vals
            rotate_mat[:, 1, 1] = cos_vals
            batch['rotate_mat'] = rotate_mat
        else:
            batch['rotate_mat'] = None

        # 2. Run Encoders Manually
        # This extracts the [Batch, Nodes, 128] embedding
        local_embed = self.backbone.local_encoder(data=batch)
        out = self.backbone.global_interactor(data=batch, local_embed=local_embed)

        # ----------------------------------------------------------
        # B. EGO EXTRACTION
        # ----------------------------------------------------------
        assert hasattr(batch, "ego_index"), "Batch missing ego_index"
        ego_idx = batch.ego_index.long()
        
        # Check output shape
        if not torch.is_tensor(out):
             raise RuntimeError(f"Backbone encoders returned {type(out)} instead of Tensor")
             
        # Extract Ego: [Batch, 128]
        ego_embeds = out[0, ego_idx, :]

        # ----------------------------------------------------------
        # C. CONTEXT INJECTION (Map Hints)
        # ----------------------------------------------------------
        if hasattr(batch, 'turn_directions'):
            ego_turn_dirs = batch.turn_directions[ego_idx]
            ego_turn_dirs = torch.clamp(ego_turn_dirs, 0, 2).long()
            map_hint = F.one_hot(ego_turn_dirs, num_classes=3).float()
            input_vector = torch.cat([ego_embeds, map_hint], dim=1)
        else:
            zeros = torch.zeros(ego_embeds.size(0), 3, device=self.device)
            input_vector = torch.cat([ego_embeds, zeros], dim=1)

        # ----------------------------------------------------------
        # D. CLASSIFICATION
        # ----------------------------------------------------------
        logits = self.head(input_vector)
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.backbone.parameters(), 'lr': 1e-5}, # Slow backbone
            {'params': self.head.parameters(), 'lr': self.hparams.learning_rate} # Fast head
        ])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }

    def training_step(self, batch, batch_idx):
        targets = batch.maneuver_id.view(-1).long()
        logits = self(batch)
        loss = self.criterion(logits, targets)
        preds = torch.argmax(logits, dim=1)

        self.train_acc(preds, targets)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, batch_size=targets.size(0))
        self.log("train_acc", self.train_acc, on_epoch=True, prog_bar=True, batch_size=targets.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        targets = batch.maneuver_id.view(-1)
        loss = self.criterion(logits, targets)
        preds = torch.argmax(logits, dim=1)

        self.val_acc.update(preds, targets)
        self.val_f1_per_class.update(preds, targets)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=targets.size(0))
        return loss

    def on_validation_epoch_end(self):
        f1_scores = self.val_f1_per_class.compute()
        acc = self.val_acc.compute()
        macro_f1 = f1_scores.mean()

        self.log("val_f1_macro", macro_f1, prog_bar=True)
        self.log("val_acc_epoch", acc, prog_bar=False)

        if self.trainer.is_global_zero:
            print(f"\n{'='*60}")
            print(f"Epoch {self.current_epoch} Results | Macro F1: {macro_f1:.4f} | Acc: {acc:.2%}")
            print(f"{'-'*60}")
            print(f"{'Class':<15} | {'F1 Score':<10} | {'Status'}")
            print(f"{'-'*60}")
            for i, name in enumerate(self.class_names):
                score = f1_scores[i].item()
                status = "⚠️ LOW" if score < 0.5 else "✅ OK"
                print(f"{name:<15} | {score:.4f}     | {status}")
            print(f"{'='*60}\n")

        self.val_f1_per_class.reset()
        self.val_acc.reset()