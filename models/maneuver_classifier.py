import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score


class ManeuverClassifier(pl.LightningModule):
    """
    Ego-centric maneuver classifier on top of a frozen HiVT / CVAE backbone.

    Backbone output: [B, N, 128]
    Ego pooling via batch.ego_index → [B, 128]
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
        self.backbone.eval()
        with torch.no_grad():
            global_embed = self.backbone(batch)

        num_graphs = batch.num_graphs
        assert hasattr(batch, "ego_index"), "Batch missing ego_index"

        # ego_index should be one per graph
        assert batch.ego_index.numel() == num_graphs, (
            f"ego_index must have length num_graphs={num_graphs}, "
            f"got {batch.ego_index.numel()}"
        )

        # ------------------------------------------------------------
        # Case 1: [num_graphs, N, D]  (graph-batched)
        # ------------------------------------------------------------
        if global_embed.dim() == 3 and global_embed.size(0) == num_graphs:
            ego_embeds = global_embed[
                torch.arange(num_graphs, device=global_embed.device),
                batch.ego_index
            ]  # [num_graphs, D]

        # ------------------------------------------------------------
        # Case 2: [1, total_nodes, D] (single-batch)
        # ------------------------------------------------------------
        elif global_embed.dim() == 3 and global_embed.size(0) == 1:
            assert hasattr(batch, "ptr"), "Batch missing ptr (required for node offsets)"
            # ptr: [num_graphs + 1] node offsets
            ego_global = batch.ptr[:-1].to(global_embed.device) + batch.ego_index.to(global_embed.device)
            ego_embeds = global_embed[0, ego_global, :]  # [num_graphs, D]

        # ------------------------------------------------------------
        # Case 3: [total_nodes, D] (fully flattened)
        # ------------------------------------------------------------
        elif global_embed.dim() == 2:
            assert hasattr(batch, "ptr"), "Batch missing ptr (required for node offsets)"
            ego_global = batch.ptr[:-1].to(global_embed.device) + batch.ego_index.to(global_embed.device)
            ego_embeds = global_embed[ego_global, :]  # [num_graphs, D]

        else:
            raise RuntimeError(
                f"Unsupported backbone output shape {tuple(global_embed.shape)} "
                f"for num_graphs={num_graphs}"
            )

        logits = self.head(ego_embeds)  # [num_graphs, num_classes]
        return logits


    # --------------------------------------------------------------
    # TRAINING STEP
    # --------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        targets = batch.maneuver_id.view(-1).long()
        logits = self(batch)

        loss = self.criterion(logits, targets)
        preds = torch.argmax(logits, dim=1)

        self.train_acc(preds, targets)

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=targets.size(0),
        )
        self.log(
            "train_acc",
            self.train_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=targets.size(0),
        )

        return loss

    # --------------------------------------------------------------
    # VALIDATION STEP
    # --------------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            print("ego_index shape:", batch.ego_index.shape)

        logits = self(batch)
        targets = batch.maneuver_id.view(-1)

        loss = self.criterion(logits, targets)
        preds = torch.argmax(logits, dim=1)

        self.val_acc.update(preds, targets)
        self.val_f1_per_class.update(preds, targets)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=targets.size(0),
        )

        return loss

    # --------------------------------------------------------------
    # VALIDATION EPOCH END
    # --------------------------------------------------------------
    def on_validation_epoch_end(self):
        self.val_acc.reset()
        self.val_f1_per_class.reset()

    # --------------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.head.parameters(),
            lr=self.hparams.learning_rate,
        )
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
