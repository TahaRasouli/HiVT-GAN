import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score


class ManeuverClassifier(pl.LightningModule):
    """
    Graph-level maneuver classifier on top of a frozen HiVT / CVAE backbone.
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
        # 1. BACKBONE (FROZEN)
        # ----------------------------------------------------------
        self.backbone = frozen_backbone
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        # ----------------------------------------------------------
        # 2. CLASSIFICATION HEAD
        # ----------------------------------------------------------
        self.head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

        # ----------------------------------------------------------
        # 3. LOSS + METRICS
        # ----------------------------------------------------------
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1_per_class = F1Score(
            task="multiclass", num_classes=num_classes, average=None
        )

        self.class_names = [
            "Straight",
            "Left Turn",
            "Right Turn",
            "U-Turn",
            "LC Left",
            "LC Right",
            "Stationary",
        ]

    # --------------------------------------------------------------
    # FORWARD
    # --------------------------------------------------------------
    def forward(self, batch):
        """
        Returns logits of shape [B, num_classes].
        """
        self.backbone.eval()
        with torch.no_grad():
            global_embed = self.backbone(batch)

        # Normalize backbone output shape
        # Acceptable:
        #   [B, D]
        #   [B, 1, D] -> squeeze
        if global_embed.dim() == 3 and global_embed.size(1) == 1:
            global_embed = global_embed.squeeze(1)

        # Final safety
        assert global_embed.dim() == 2, (
            f"Backbone output must be [B,D], got {global_embed.shape}"
        )

        logits = self.head(global_embed)
        return logits

    # --------------------------------------------------------------
    # TRAINING STEP
    # --------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        targets = batch.maneuver_id.view(-1).long()

        # Skip empty batches (Lightning + PyG edge case)
        if targets.numel() == 0:
            return None

        logits = self(batch)

        # Shape hygiene
        logits = logits.view(logits.size(0), -1)
        targets = targets.view(-1)

        assert logits.size(0) == targets.size(0), (
            f"Train N mismatch: logits {logits.shape}, targets {targets.shape}"
        )

        loss = self.criterion(logits, targets)

        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, targets)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_epoch=True, prog_bar=True)

        return loss

    # --------------------------------------------------------------
    # VALIDATION STEP
    # --------------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        targets = batch.maneuver_id.view(-1).long()

        # Skip empty batches
        if targets.numel() == 0:
            return None

        logits = self(batch)

        # Shape hygiene
        logits = logits.view(logits.size(0), -1)
        targets = targets.view(-1)

        assert logits.size(0) == targets.size(0), (
            f"Val N mismatch: logits {logits.shape}, targets {targets.shape}"
        )

        loss = self.criterion(logits, targets)

        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, targets)
        self.val_f1_per_class(preds, targets)

        self.log("val_loss", loss, prog_bar=True)
        return loss

    # --------------------------------------------------------------
    # VALIDATION EPOCH END
    # --------------------------------------------------------------
    def on_validation_epoch_end(self):
        f1_scores = self.val_f1_per_class.compute()
        acc = self.val_acc.compute()

        print("\n" + "=" * 40)
        print(f"Epoch {self.current_epoch} Results")
        print("-" * 40)
        print(f"Overall Accuracy: {acc:.4f}")
        print("-" * 40)
        print(f"{'Class':<15} | {'F1 Score':<10}")
        print("-" * 40)

        for i, name in enumerate(self.class_names):
            print(f"{name:<15} | {f1_scores[i].item():.4f}")

        print("=" * 40 + "\n")

        self.val_f1_per_class.reset()
        self.val_acc.reset()

    # --------------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.head.parameters(), lr=self.hparams.learning_rate)
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
