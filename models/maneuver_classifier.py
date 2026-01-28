import torch
import torch.nn as nn
import torch.nn.functional as F  # Needed for One-Hot Encoding
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score


class ManeuverClassifier(pl.LightningModule):
    """
    Ego-centric maneuver classifier with Context Injection and Fine-Tuning.

    Input:  HiVT Embedding [128] + Map Turn Hint [3]
    Output: Maneuver Class [7]
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

        # ----------------------------------------------------------
        # 1. BACKBONE (Suggestion C: Unfrozen for Fine-Tuning)
        # ----------------------------------------------------------
        self.backbone = frozen_backbone
        
        # We REMOVE the "requires_grad = False" loop.
        # The backbone is now trainable, but constrained by a low LR.

        # ----------------------------------------------------------
        # 2. CLASSIFICATION HEAD (Suggestion B: Context Injection)
        # ----------------------------------------------------------
        # Input Dimension: 128 (Trajectory) + 3 (Map Turn Direction) = 131
        self.head = nn.Sequential(
            nn.Linear(128 + 3, 128), 
            nn.ReLU(),
            nn.Dropout(0.5),  # Suggestion A: High Regularization
            nn.Linear(128, num_classes),
        )

        # ----------------------------------------------------------
        # 3. LOSS + METRICS
        # ----------------------------------------------------------
        # Label Smoothing helps with overfitting
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1_per_class = F1Score(
            task="multiclass", num_classes=num_classes, average=None
        )

        self.class_names = [
            "Straight", "Left Turn", "Right Turn", "U-Turn",
            "LC Left", "LC Right", "Stationary",
        ]

    # --------------------------------------------------------------
    # FORWARD
    # --------------------------------------------------------------
    def forward(self, batch):
        # We allow gradients to flow through the backbone now (Fine-Tuning)
        out = self.backbone(batch)  # [1, total_nodes, 128]

        # Basic Checks
        if not torch.is_tensor(out):
            raise RuntimeError(f"Backbone returned non-tensor: {type(out)}")

        assert hasattr(batch, "ego_index"), "Batch missing ego_index"
        ego_idx = batch.ego_index.long().to(out.device)
        
        # Extract Ego Embedding
        ego_embeds = out[0, ego_idx, :]  # [Batch, 128]

        # ----------------------------------------------------------
        # SUGGESTION B: CONTEXT INJECTION (Map Hints)
        # ----------------------------------------------------------
        # We extract the 'turn_directions' map feature for the ego vehicle.
        # Values: 0 (Straight), 1 (Left), 2 (Right)
        if hasattr(batch, 'turn_directions'):
            # Get raw indices
            ego_turn_dirs = batch.turn_directions[ego_idx] # [Batch]
            
            # Safety Clamp (0-2) just to be sure
            ego_turn_dirs = torch.clamp(ego_turn_dirs, 0, 2).long()
            
            # One-Hot Encode -> [Batch, 3] (e.g., [0, 1, 0] for Left)
            map_hint = F.one_hot(ego_turn_dirs, num_classes=3).float()
            
            # Concatenate: [Batch, 128] + [Batch, 3] -> [Batch, 131]
            input_vector = torch.cat([ego_embeds, map_hint], dim=1)
        else:
            # Fallback (Zeros) if map data missing
            zeros = torch.zeros(ego_embeds.size(0), 3, device=ego_embeds.device)
            input_vector = torch.cat([ego_embeds, zeros], dim=1)

        # Pass combined vector to Head
        logits = self.head(input_vector)
        return logits

    # --------------------------------------------------------------
    # OPTIMIZER (Suggestion C: Differential Learning Rates)
    # --------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            # Group 1: Backbone (Train very slowly to refine physics)
            {
                'params': self.backbone.parameters(), 
                'lr': 1e-5 
            },
            # Group 2: Head (Train normally)
            {
                'params': self.head.parameters(), 
                'lr': self.hparams.learning_rate
            }
        ])
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    # --------------------------------------------------------------
    # STANDARD TRAINING STEPS
    # --------------------------------------------------------------
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
            # 1. Compute the per-class metrics
            f1_scores = self.val_f1_per_class.compute()
            acc = self.val_acc.compute()

            # 2. Compute Macro F1 (Average of all class F1s)
            # This is critical for ModelCheckpoint to verify overall performance
            macro_f1 = f1_scores.mean()

            # 3. Log main metrics for the Trainer
            self.log("val_f1_macro", macro_f1, prog_bar=True)
            self.log("val_acc_epoch", acc, prog_bar=False)

            # 4. Log per-class F1s (good for TensorBoard, hidden from progress bar)
            for i, name in enumerate(self.class_names):
                self.log(f"val_f1_{name}", f1_scores[i], prog_bar=False)

            # 5. Print readable table to Terminal (Main process only)
            if self.trainer.is_global_zero:
                print(f"\n{'='*60}")
                print(f"Epoch {self.current_epoch} Results | Macro F1: {macro_f1:.4f} | Acc: {acc:.2%}")
                print(f"{'-'*60}")
                print(f"{'Class':<15} | {'F1 Score':<10} | {'Status'}")
                print(f"{'-'*60}")
                
                for i, name in enumerate(self.class_names):
                    score = f1_scores[i].item()
                    # Visual flag for poor performance
                    status = "⚠️ LOW" if score < 0.5 else "✅ OK"
                    print(f"{name:<15} | {score:.4f}     | {status}")
                print(f"{'='*60}\n")

            # 6. Reset metrics for the next epoch
            self.val_f1_per_class.reset()
            self.val_acc.reset()