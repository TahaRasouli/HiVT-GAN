import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score


class ManeuverClassifier(pl.LightningModule):
    """
    Ego-centric maneuver classifier on top of a frozen HiVT / CVAE backbone.

    Backbone output observed: [B, N, 128]
    We select the ego embedding (prefer batch.ego_index, otherwise default to 0).
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
        # 1. FROZEN BACKBONE
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
        self.val_f1_per_class = F1Score(task="multiclass", num_classes=num_classes, average=None)

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
    # FORWARD (EGO POOLING WITH FALLBACK)
    # --------------------------------------------------------------
    def forward(self, batch):
        """
        Returns logits of shape [B, num_classes].
        """
        self.backbone.eval()
        with torch.no_grad():
            node_embeddings = self.backbone(batch)
            # Expected: [B, N, 128]

        if node_embeddings.dim() != 3:
            raise RuntimeError(f"Expected backbone output [B,N,D], got {node_embeddings.shape}")

        B, N, D = node_embeddings.shape
        if B == 0 or N == 0:
            # Empty batch / empty graph case
            return torch.empty((0, self.head[-1].out_features), device=node_embeddings.device)

        # Prefer ego_index if present; otherwise default to 0
        if hasattr(batch, "ego_index"):
            ego_idx = batch.ego_index.view(-1).long()
            if ego_idx.numel() != B:
                # If ego_index exists but doesn't match, fall back
                ego_idx = torch.zeros((B,), dtype=torch.long, device=node_embeddings.device)
        else:
            ego_idx = torch.zeros((B,), dtype=torch.long, device=node_embeddings.device)

        # Bound check
        ego_idx = torch.clamp(ego_idx, 0, N - 1)

        ego_embed = node_embeddings[torch.arange(B, device=node_embeddings.device), ego_idx]  # [B, D]
        logits = self.head(ego_embed)  # [B, num_classes]
        return logits

    # --------------------------------------------------------------
    # TRAINING STEP
    # --------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        targets = batch.maneuver_id.view(-1).long()
        if targets.numel() == 0:
            return None

        logits = self(batch)
        if logits.numel() == 0:
            return None

        # Ensure [B, C] and [B]
        assert logits.dim() == 2, f"logits must be [B,C], got {logits.shape}"
        assert targets.dim() == 1, f"targets must be [B], got {targets.shape}"
        assert logits.size(0) == targets.size(0), f"N mismatch: logits {logits.shape}, targets {targets.shape}"

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
        if targets.numel() == 0:
            return None

        logits = self(batch)
        if logits.numel() == 0:
            return None

        assert logits.dim() == 2, f"logits must be [B,C], got {logits.shape}"
        assert targets.dim() == 1, f"targets must be [B], got {targets.shape}"
        assert logits.size(0) == targets.size(0), f"N mismatch: logits {logits.shape}, targets {targets.shape}"

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
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }
